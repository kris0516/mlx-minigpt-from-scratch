# Benchmark Report Template — Awaiting Runtime Measurements

**Status: NOT MEASURED.** This page is a schema for a future opt-in study, not a result.

## Required fields

- Hardware model, SoC/GPU configuration, unified-memory capacity, macOS version, ambient/thermal setup, power state.
- Python version, MLX 0.32.2, git commit SHA, dtype and compile mode.
- Model V/C/H/N/T, batch size B, prompt length, generated-token count, actual RNG seed.
- Procedure: number of warmups/repeats, wall-time start/end with mx.eval boundary, independent process per configuration.
- MLX active/cache/raw peak bytes and window peak calculation; macOS Memory Pressure/Swap observations **separately**, if genuinely observed.
- Pre-fill positions/s and autoregressive generated tokens/s as separate metrics, because no-cache generation recomputes the window.

## Unmeasured README table

| Hardware | C/H/N | T/B | MLX active/peak | Prefill positions/s | Generate tokens/s | Swap/thermal |
|---|---|---|---|---|---|---|
| Not recorded | 32/4/1 | 16/1 | Not measured | Not measured | Not measured | Not measured |
| Not recorded | 64/4/2 | 32/1 | Not measured | Not measured | Not measured | Not measured |

No row may be changed to zero or PASS without a real observation. Save raw JSON separately, preserve code and commit SHA, and disclose thermal sensors used. The optional `scripts/benchmark_memory.py` and `scripts/benchmark_performance.py` are not automatically run or triggered in CI. One small run is not evidence of sustained low-heat behavior.
