# Credits, Papers, Data Source and License

This educational repository's original code and documentation are published by **kris0516** under its own [MIT LICENSE](../LICENSE), which currently begins “MIT License” and records “Copyright (c) 2026 kris0516”. The license does not imply ownership of third-party frameworks, papers or datasets.

## Main framework and software

- [Apple MLX](https://github.com/ml-explore/mlx), pinned here to **v0.32.2**. Apple MLX offers array, autodiff, compilation, Metal execution, and `nn.Module` APIs used by the lessons. External MLX source remains under its **own** upstream license and copyright.
- [Python](https://www.python.org/), Python 3.12; [uv](https://docs.astral.sh/uv/) for environment management; [Jupyter](https://jupyter.org/) and VS Code for work-copy notebooks. Their terms are not overwritten by this repository's MIT notice.
- The prior authored project lessons and code in this repository remain linked in [complete INDEX](INDEX.md) and [roadmap](course-roadmap.md).

## Foundational research milestones

| Learning stage | Reference |
|---|---|
| LSTM and gated recurrence | Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997), [Neural Computation](https://doi.org/10.1162/neco.1997.9.8.1735) |
| Attention and encoder–decoder | Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) |
| Transformer | Vaswani et al., *Attention Is All You Need* (2017), [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| GPT decoder-only language modeling | Radford et al., *Improving Language Understanding by Generative Pre-Training* (2018), [OpenAI publication](https://openai.com/research/language-unsupervised) |
| RoPE | Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| Multi-query attention | Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| Grouped-query attention | Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| IO-aware exact attention | Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |

References identify conceptual origins, not claims that the full upstream architectures or published fused kernels are integrated into this project's baseline. The default implementation remains an untied, learned-position MiniGPT using dense causal MHA; Step 22 treats modern variants separately.

## Tiny Shakespeare / character data

The curriculum discusses the **Tiny Shakespeare** text accessible in Andrej Karpathy's [char-rnn example dataset](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), with an optional explicit fetch path documented in earlier Step 16 lessons. The present source package/CLI **does not automatically download** or bundle that dataset. Readers must supply a local UTF-8 file if they choose to run training.

The literary works of Shakespeare are historically public-domain texts in many jurisdictions, but a particular compiled data artifact, repository and any accompanying materials can have separate provenance or terms. Do not treat a general public-domain statement as permission to copy every third-party dataset without checking its actual provenance.

## License consistency and limitations

- The tracked [LICENSE](../LICENSE) is MIT and attributes the repository to kris0516 (2026). This Step **did not modify** its terms.
- No third-party source files were copied into the new `src/minigpt/` module from external libraries. Educational MLX API usage and external research citations are not relicensing of MLX or paper content.
- `pyproject.toml` currently does not declare a distribution-specific license metadata field or a build-system. We do not claim published package-wheel compliance; **PYTHONPATH=src** remains the documented import route.
- Runtime tests, benchmark measurements, external source license audits, and final release validation have not been executed in this publication pass.

Please use the upstream links and paper identifiers when reusing this project for research or substantial derivative teaching materials.
