# Triton-Fused-RMSNorm-Kernel

A high-performance Fused RMSNorm operator built with OpenAI Triton. The goal is to fuse the RMSNorm computation into a single kernel on a consumer GPU (this repo's primary dev/test box is an **RTX 3060 Laptop, 6GB VRAM**) and approach the HBM bandwidth ceiling — used as a stepping-stone project before tackling more complex kernels like FlashAttention.

---

## 1. Background & Goals

### 1.1 Core Pain Point

RMSNorm is a memory-bound operation: a forward pass needs to read the input once, compute a statistic, read the input again to normalize, then write the output — one full round trip per intermediate step. A naive PyTorch implementation stitched together from multiple ops (`pow` → `mean` → `rsqrt` → `mul` → `mul`) materializes several intermediate tensors, pushing the number of HBM round trips well above the theoretical minimum of 1 read + 1 write.

### 1.2 Mathematical Definition

For an input row vector $x \in \mathbb{R}^d$ and weight $g \in \mathbb{R}^d$:

$$
y_i = \frac{x_i}{\sqrt{\dfrac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot g_i
$$

Backward pass (if included in scope — see the trade-off note in Phase 2):

$$
\text{let } r = \dfrac{1}{\sqrt{\frac{1}{d}\sum x_j^2+\epsilon}}, \quad
\frac{\partial L}{\partial x_i} = r \cdot g_i \cdot \frac{\partial L}{\partial y_i} - \frac{r^3}{d}\, x_i \sum_j x_j g_j \frac{\partial L}{\partial y_j}
$$

$$
\frac{\partial L}{\partial g_i} = \sum_{\text{rows}} \frac{\partial L}{\partial y_i}\cdot(x_i \cdot r)
$$

`dg` requires a reduction across rows — this is the extra complexity the backward kernel carries relative to forward, and it's handled separately in the plan.

### 1.3 Quantified Acceptance Criteria

Instead of a vague "faster" target, here are measurable, baseline-relative numbers:

| Metric | Baseline (`torch.nn.RMSNorm`, eager) | Target |
|---|---|---|
| Forward latency (d=4096, rows=8192, fp16) | measured, recorded | ≤ 60% of baseline latency |
| Effective memory bandwidth utilization | measured, recorded | ≥ 80% of nominal device bandwidth (estimated via `2*M*N*dtype_bytes / latency`, with nominal bandwidth taken from `nvidia-smi` / `torch.cuda.get_device_properties`) |
| Correctness | — | fp32: `atol=1e-5` vs. PyTorch reference; fp16/bf16: `atol=1e-2, rtol=1e-3` (intermediate accumulation is upcast to fp32) |

> Note: the original plan referenced a desktop RTX 3060/4060, but this machine is actually an **RTX 3060 Laptop (6GB)**, with nominal bandwidth ~336GB/s (slightly below the desktop 3060's 360GB/s). This means the benchmark shape matrix needs to respect VRAM limits (e.g. `batch=128, seq=2048, dim=4096` in fp32 with multiple live intermediates can OOM) — concrete limits are given in Phase 3 below.

## 2. Tech Stack & Repo Layout

### 2.1 Tech Stack

- Python 3.10+ (this machine has 3.12.3, no downgrade needed)
- PyTorch 2.x + OpenAI Triton (**neither is currently installed** — top priority for Phase 0)
- `pytest` + `torch.testing.assert_close`
- `torch.profiler` + TensorBoard trace viewer
- `triton.testing.do_bench` / `perf_report`

### 2.2 Planned Repo Layout

```
Triton-Fused-RMSNorm-Kernel/
├── README.md                  # this planning doc
├── LICENSE
├── pyproject.toml / requirements.txt
├── src/
│   └── triton_rmsnorm/
│       ├── __init__.py
│       ├── reference.py       # PyTorch baseline implementation
│       ├── kernel_fwd.py      # Triton forward kernel
│       ├── kernel_bwd.py      # Triton backward kernel (optional scope)
│       └── ops.py             # torch.autograd.Function wrapper + nn.Module
├── tests/
│   ├── test_correctness.py    # shape/dtype matrix tests
│   └── test_edge_cases.py     # non-power-of-2 d, d=1, extreme eps, etc.
├── benchmarks/
│   ├── bench_latency.py       # perf_report line charts
│   └── bench_profile.py       # torch.profiler trace export
├── reports/
│   └── performance_report.md  # charts and conclusions from Phase 3/4
└── .github/workflows/ci.yml   # optional: run correctness tests (skip if no GPU)
```

## 3. Milestones (Phase 0 → 4)

### Phase 0: Environment Setup (Day 0 — missing from the original plan, but a hard prerequisite)

Current state of this machine: `nvidia-smi` detects an RTX 3060 Laptop (driver 610.53, CUDA UMD 13.3), but `python3 -c "import torch"` / `import triton` both raise `ModuleNotFoundError`.

Tasks:
1. Create a fresh `conda` environment (miniconda is already installed) rather than polluting the system Python 3.12 — Triton/PyTorch wheel support for 3.12 tends to lag behind 3.10/3.11, so a 3.11 env lowers the odds of hitting compatibility issues.
2. Install a PyTorch build matching the CUDA 13.x driver (`pip install torch` pulls in Triton as a dependency by default, so a separate install usually isn't needed; install `triton` directly only if you need bleeding-edge features).
3. Verify with a 3-line smoke test: `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`, and running the official Triton `vector-add` tutorial kernel.

Exit criteria: `pytest -q tests/test_env.py` passes (asserting exactly the three checks above).

### Phase 1: Baseline & Test Scaffolding (Day 1-3)

- **Task 1**: Implement a pure-PyTorch mathematically-equivalent version in `reference.py`, while also keeping a call to `torch.nn.RMSNorm` for cross-checking (the two should match exactly — this validates that the reference implementation itself is correct).
- **Task 2**: `test_correctness.py` uses `pytest.mark.parametrize` to cover the cross product of `batch_size ∈ {1,8,32,128}`, `seq_len ∈ {128,512,2048}`, `dim ∈ {768,4096}`, `dtype ∈ {fp32,fp16,bf16}` — trim to a representative subset rather than the full cartesian product, prioritizing boundary values.
- **Task 3**: Get `torch.profiler` running in `bench_profile.py`, export a `chrome_trace`, and record the baseline's kernel latency/bandwidth estimate as the reference numbers for Phase 3, saved under `reports/`.

Exit criteria: all baseline tests pass; a table/trace file recording baseline latency numbers is archived.

### Phase 2: Core Triton Development (Day 4-7)

- **Task 1 (Forward)**: `grid=(M,)`, one program per row; use `tl.program_id(0)` for the row offset, `tl.arange(0, BLOCK_SIZE)` + masking to handle non-power-of-2 `d`; upcast to fp32 before the sum-of-squares (`tl.sum(x.to(tl.float32) * x.to(tl.float32))`) to avoid fp16 overflow; use `other=0.0` on `tl.load` for out-of-bounds positions so they don't pollute the sum.
- **Task 2 (Wrapper)**: `torch.autograd.Function.forward` invokes the kernel; ship a "correct but not necessarily fast" `backward` first (even a pure-PyTorch-autograd fallback is fine) to keep the op differentiable, then decide in Phase 3 whether a dedicated backward kernel is worth writing.
- **Scope trade-off (needs your call)**: the original plan is titled "Fused RMSNorm" but only ever describes the forward kernel. If the target use case is **inference** (LLM inference, mirroring how vLLM/SGLang use it), forward-only is sufficient and the `nn.Module` can just wrap calls in `torch.no_grad()`. If you want training support / to present this as a "complete operator implementation" on your resume, add a backward kernel at the end of Phase 2 (it's more work than forward — `dg` needs a cross-row reduction, doable as a two-stage kernel: compute per-row partial gradients, then either a separate reduction kernel or `tl.atomic_add`). This decision materially changes the Day 4-7 workload, so it's worth deciding only after forward + correctness tests are done, not before.

Exit criteria: Triton forward output matches the PyTorch baseline numerically across the full Phase 1 test matrix; the `nn.Module` wrapper is a drop-in replacement.

### Phase 3: Autotuning & Performance Tuning (Day 8-10)

- **Task 1**: `@triton.autotune` with, for Ampere, `num_warps ∈ {2,4,8}`, `num_stages ∈ {1,2,3,4}`, `BLOCK_SIZE` set to the nearest power of 2 ≥ `dim`; `key=["dim"]` to trigger re-tuning. Since this is a 6GB laptop GPU, the autotune sweep itself repeatedly allocates/frees memory — get the sweep working on small batches first to avoid OOMing during autotuning itself.
- **Task 2**: `benchmarks/bench_latency.py` with `triton.testing.perf_report`, x-axis `dim` from 512 to 8192, y-axis latency/bandwidth, output to `reports/*.png`. **VRAM budget warning**: at `batch=128, seq=2048, dim=4096, fp32`, the input tensor alone is `128*2048*4096*4B ≈ 4.3GB`; add the output and intermediates and you're near the 6GB ceiling — either drop to fp16 or cap the largest shape around `batch*seq ≤ 65536`, and verify empirically with headroom to spare.
- **Task 3**: Compare `torch.profiler` traces before/after fusion, quantifying the drop in kernel launch count (N → 1) and the reduction in HBM bytes read/written.

Exit criteria: a clear perf_report chart, plus a written conclusion stating "at shape X, we hit Y GB/s, Z% of nominal bandwidth."

### Phase 4: Open-Sourcing & Resume Asset (Day 11-14)

- **Task 1**: Pull the *measured* (not estimated) numbers already produced into a benchmark summary table at the top of the README; `reports/performance_report.md` holds the full charts and methodology so others can reproduce the results.
- **Task 2**: Phrase resume bullets with **ranges + explicit conditions** rather than a bare multiplier, e.g. "On an RTX 3060 Laptop (6GB), d=4096, fp16: reduced forward latency by X% vs. PyTorch eager, raising HBM bandwidth utilization from Y% to Z%" — this avoids getting caught out when an interviewer asks "which GPU, which shape?"

Exit criteria: the repo has a one-command `pytest` and `python benchmarks/bench_latency.py`; the README summary matches the raw data under `reports/`.

## 4. Key Challenges & Design Decisions (additions to the original plan)

1. **`BLOCK_SIZE` must be a power of 2**: this is a hard requirement of `tl.arange(0, BLOCK_SIZE)` (not a "recommendation") — it comes from how Triton allocates vectorized registers under the hood. Non-power-of-2 `d` is handled via masking, not by making `BLOCK_SIZE` itself non-power-of-2 — worth spelling out clearly in code comments so it isn't forgotten later.
2. **Single-pass vs. two-pass reduction**: when `d` fits within one block (typical LLM hidden dims like 4096 usually do), a single program can `tl.load` the whole row and `tl.sum` it in one pass — the most bandwidth-efficient option. A chunked two-pass accumulation only matters when `d` is huge (far beyond SRAM capacity, e.g. 32768+). Implement single-pass first to cover mainstream LLM hidden dims; treat two-pass as a stretch goal rather than over-engineering it in Phase 2.
3. **Stride handling**: if the upstream tensor is non-contiguous (e.g. a view after `permute`), `row_start_ptr = x_ptr + row_idx * stride_row` must take an explicit `stride_row` rather than assuming `stride_row == dim` — otherwise non-contiguous input silently reads the wrong memory (no error, just wrong numbers). Add a non-contiguous-input case to the test matrix.
4. **Precision**: upcast to `tl.float32` before `tl.sum(x*x)`; cast back to the output dtype on write-back — already noted in the original plan, kept here.
5. **Backward's `dg` cross-row reduction**: if backward is in scope, `dg` is summed over all rows — a genuine reduction. The naive approach (`tl.atomic_add` per program) has atomic-contention overhead at high row counts; worth a dedicated look during Phase 3 profiling, but not worth over-engineering upfront.

## 5. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| 6GB VRAM OOMs on large benchmark shapes | Phase 3 Task 2 can't complete the full shape matrix | Estimate with `torch.cuda.mem_get_info()` before laying out the benchmark matrix; drop precision or batch the sweep if needed |
| Triton/PyTorch version mismatch with this machine's CUDA 13.x driver | Phase 0 gets stuck | Prefer a fresh conda env + the latest stable build from the official pip index; check the PyTorch/Triton release notes' CUDA compatibility table on error |
| Backward-kernel scope creep eats into Phase 3/4 time | Two-week schedule slips | Use the trade-off criteria in Section 3 — ship the inference-oriented forward-only version first, treat backward as a stretch goal if time allows |

---

*Next step: decide the Phase 2 scope trade-off (whether to include a backward kernel), then start Phase 0 environment setup.*
