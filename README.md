# MiniGPT from Scratch — Apple Silicon × MLX

> **Teaching materials prepared through Step 25.3.** No training, tests, Notebook execution, sampling, benchmark, smoke test or fresh-clone audit was performed as part of this publication. Prepared does **not** mean runtime validated or learner accepted. [Progress](docs/course-progress.md).

从最小矩阵运算出发，沿 **MLP → RNN → LSTM/GRU → Seq2Seq → Attention → Transformer → Decoder-only GPT** 的历史路线，在 Apple Silicon 上构建字符级 MiniGPT 教学项目。默认核心实现保持可解释、保守的小模型；**不是**经过性能调优的生产大语言模型。

## Environment

- Apple Silicon / macOS, native arm64; Python 3.12 with project `.venv`.
- **Apple MLX exactly 0.32.2** in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).
- VS Code/Jupyter; track only master `notebooks/*.ipynb`; use ignored work copies for experiments.
- No CUDA/PyTorch training backend, no automatically downloaded corpus, no bundled weights.

## Architecture (current baseline, not Step 22 optional variants)

```text
token IDs (B,T)                 learned positions (T,C)
     |                                |
     v                                v
Embedding (B,T,C) -----------------> add
                                      |
                 +--------------------+-------------------+
                 |  TransformerBlock repeated N times    |
                 | RMSNorm -> causal MHA -> residual     |
                 | RMSNorm -> GELU MLP  -> residual      |
                 +--------------------+-------------------+
                                      |
                               final RMSNorm
                                      |
                             untied LM head
                                      v
                             raw logits (B,T,V)
```

Inside MHA: a `C→3C` projection splits Q/K/V, head-major tensors have shape `(B,H,T,d)` with `d=C/H`, then dense scores `(B,H,T,T)` are causally masked, softmaxed and multiplied by V. The model uses learned absolute position embeddings and **does not implement KV cache, RoPE, GQA, SwiGLU, or Flash-style attention in the baseline**. Step 22 teaches these as optional concepts.

## Quick start — install only (NOT performed for this release)

```bash
git clone https://github.com/kris0516/mlx-minigpt-from-scratch.git
cd mlx-minigpt-from-scratch
uv sync --frozen
```

Confirm the VS Code interpreter / Notebook kernel points to this project's `.venv/bin/python` (Python 3.12). Run the curriculum notebooks **only after copying** a master, for example:

```bash
git pull
```

```bash
cp -n notebooks/102_step23_1_data_module.ipynb notebooks/work/102_step23_1_data_module_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

This preserves original masters and prior personal notes. The `notebooks/work/*.ipynb` pattern is Git-ignored.

## Source package and future explicit commands

`src/minigpt/data.py` handles character vocabulary, split and shifted windows; `model.py` preserves the Step 15.4 baseline; `train.py` keeps loss/Adam state boundaries; `generate.py` performs no-cache decoding; `config.py` and `cli.py` provide configuration and opt-in commands.

The repository currently uses a src layout **without an installed console-script entry point**. For future explicit opt-in commands use `PYTHONPATH=src`:

```bash
# Future choice only; requires locally prepared UTF-8 corpus at this exact path.
PYTHONPATH=src uv run python -m minigpt.cli train --corpus data/tinyshakespeare/input.txt --output runs/unique-run-001

# Future choice only; requires a real completed training command and its files.
PYTHONPATH=src uv run python -m minigpt.cli generate --checkpoint runs/unique-run-001 --prompt "To be" --new-tokens 20

# Future choice only; currently NOT executed and NOT passing by assumption.
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_*.py' -v
```

The CLI's `model.safetensors` + `run.json` artifact is **weights-only for inference**. It is **not** a complete Adam/RNG checkpoint for exact training resumption. See [Step 20](docs/step-20-checkpointing-and-recovery.md) for the separate resume protocol.

## Learning map and notebooks

| Stage | Materials |
|---|---|
| 1–3 Environment / tensors | [Roadmap](docs/course-roadmap.md), [Step 03](docs/step-03-tensor-basics.md) |
| 4–10 MLP → recurrent networks → Seq2Seq | [Step 04](docs/step-04-neural-network-basics.md), [Step 10](docs/step-10-seq2seq-attention.md) |
| 11–15 Self-attention → full GPT | [Step 11](docs/step-11-self-attention.md), [Step 15](docs/step-15-transformer-to-gpt.md) |
| 16–20 Dataset / loss / MLX execution / training / recovery | [Step 16](docs/step-16-tiny-shakespeare-data.md), [Step 20](docs/step-20-checkpointing-and-recovery.md) |
| 21–22 Generation / modern attention variants | [Step 21](docs/step-21-autoregressive-generation.md), [Step 22](docs/step-22-modern-gpt-improvements.md) |
| 23 Source refactor | [Step 23](docs/step-23-source-refactor.md), [Master 102](notebooks/102_step23_1_data_module.ipynb) |
| 24 Tests / benchmark protocol | [Step 24](docs/step-24-testing-benchmarking.md), [unmeasured report template](docs/benchmark-report-template.md) |
| 25 GitHub-ready release preparation | [Step 25](docs/step-25-release.md) |

See the [static reproducibility audit and pending runtime gates](docs/REPRODUCIBILITY.md). The [complete notebook index](docs/INDEX.md) links the actual masters. The [full roadmap](docs/course-roadmap.md) describes every subsection, mathematical prerequisite and optional extension. [Progress](docs/course-progress.md) distinguishes published materials from exercise completion and runtime acceptance.

## Benchmark overview — not measured

| Hardware | C/H/N | T/B | MLX active/peak | Prefill positions/s | Generate tokens/s | Swap/thermal |
|---|---|---|---|---|---|---|
| Not recorded | 32/4/1 | 16/1 | Not measured | Not measured | Not measured | Not measured |
| Not recorded | 64/4/2 | 32/1 | Not measured | Not measured | Not measured | Not measured |

**These are placeholders, not zeros.** No memory/throughput/thermal advantage or zero-Swap claim is established. Future measurements must include hardware, dtype, synchronization, warmups, versions and raw records. See [benchmark methodology](docs/benchmark-report-template.md).

## License and sources

Code is under [MIT LICENSE](LICENSE). See the eventual [Credits and references](docs/CREDITS.md) and source documents. Tiny Shakespeare is not downloaded or bundled by publication. Keeping the baseline separate from optional architecture experiments protects the pedagogical checkpoint contract.
