# v0.1.0 Educational Materials — Draft Release Notes

> **Material-complete through Step 25.5. Not a tested software release.** No release tag or GitHub Release has been created in this session; no Notebook, test, training, smoke, benchmark, sample generation, or fresh-clone check was run.

## What is now on main

- Curriculum master Notebooks numbered **01–116**, with the new Step 23–25 sequence **102–116**.
- A `src/minigpt/` package: `data.py`, `model.py`, `train.py`, `generate.py`, `config.py`, `cli.py`, and `__main__.py`.
- Four `tests/test_*.py` sources and `scripts/tiny_overfit.py`; these are **prepared, not executed**.
- MLX allocator/throughput protocol scripts `scripts/benchmark_memory.py` and `scripts/benchmark_performance.py`; no observed measurements.
- README, complete [course index](INDEX.md), [reproducibility gates](REPRODUCIBILITY.md), [credits](CREDITS.md), [MIT LICENSE](../LICENSE), and [unmeasured benchmark template](benchmark-report-template.md).

## Full Step 23–25 material/commit ledger

| Subsection | Master notebook | Material commit |
|---|---|---|
| 23.1 | [102_step23_1_data_module](../notebooks/102_step23_1_data_module.ipynb) | [43e960a14d04](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/43e960a14d047e0839da9f04fce0e4f01d9cacea) |
| 23.2 | [103_step23_2_model_module](../notebooks/103_step23_2_model_module.ipynb) | [1baad6a0cb06](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/1baad6a0cb0697c12ccbeba305a4c9cb52e4bafd) |
| 23.3 | [104_step23_3_training_module](../notebooks/104_step23_3_training_module.ipynb) | [6d4528c75dea](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/6d4528c75dea3fc22343787716c823d09184839b) |
| 23.4 | [105_step23_4_generation_module](../notebooks/105_step23_4_generation_module.ipynb) | [472942e7dc82](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/472942e7dc826b49da6da12873b3b42035625dfa) |
| 23.5 | [106_step23_5_configuration_cli](../notebooks/106_step23_5_configuration_cli.ipynb) | [5a14fc48436b](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/5a14fc48436bd71948fe788fa02768ac37fe4351) |
| 24.1 | [107_step24_1_unit_tests](../notebooks/107_step24_1_unit_tests.ipynb) | [b0f9a91d67c2](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/b0f9a91d67c229e81c4e880796f91c5f54ddf1fc) |
| 24.2 | [108_step24_2_tiny_overfit_test](../notebooks/108_step24_2_tiny_overfit_test.ipynb) | [9380232c180e](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/9380232c180ef905ae5c93fdd1a8f99bc9c92f44) |
| 24.3 | [109_step24_3_memory_benchmark](../notebooks/109_step24_3_memory_benchmark.ipynb) | [5192f89760b7](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/5192f89760b7bdf80c58b54226c28b51cd210f43) |
| 24.4 | [110_step24_4_performance_thermal_benchmark](../notebooks/110_step24_4_performance_thermal_benchmark.ipynb) | [81391f8886c4](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/81391f8886c4e8292c11b57fd01d324875b50756) |
| 24.5 | [111_step24_5_readme_benchmark_table](../notebooks/111_step24_5_readme_benchmark_table.ipynb) | [7a69a4e67a20](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/7a69a4e67a202091408b291356ed37b34fa110cf) |
| 25.1 | [112_step25_1_readme_rewrite](../notebooks/112_step25_1_readme_rewrite.ipynb) | [c7969294a1e3](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/c7969294a1e3ba5f1c984743d318dc4072b32fdb) |
| 25.2 | [113_step25_2_documentation_index](../notebooks/113_step25_2_documentation_index.ipynb) | [1f0001ad2c40](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/1f0001ad2c40877496fa3a2bcbebb1c78252012d) |
| 25.3 | [114_step25_3_reproducibility_audit](../notebooks/114_step25_3_reproducibility_audit.ipynb) | [4fd8fa24d257](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/4fd8fa24d257a5f9a6b00fa1b953842b374d8fae) |
| 25.4 | [115_step25_4_credits_references_license](../notebooks/115_step25_4_credits_references_license.ipynb) | [25f28ca6f704](https://github.com/kris0516/mlx-minigpt-from-scratch/commit/25f28ca6f704a86e4caa4bf4639369adff34750d) |
| 25.5 | [116_step25_5_final_release](../notebooks/116_step25_5_final_release.ipynb) | Current main HEAD (this final material commit) |

## Baseline design preserved

The default MiniGPT remains untied and learned-position with dense causal multi-head attention, Pre-Norm RMSNorm and GELU MLP. Step 22 variants are reference lessons, not automatic model changes. The CLI stores inference-only weights/metadata, not exact optimizer/RNG resume state.

## Why no stable release tag yet?

A stable version tag would imply a specific commit has undergone the required acceptance gates. The user prohibited runtime tests and benchmark execution, so it would be misleading to claim such verification. The available connected GitHub actions in this session also do **not** expose creating annotated tags or a GitHub Release. Accordingly:

- **Remote `main`**: source and teaching materials published, per subsection.
- **Proposed release identity**: `v0.1.0` *materials candidate*; no tag ref created.
- **GitHub Release object**: NOT created.
- **Technical signoff**: pending actual clean-clone environment check, unit tests, CLI smoke, optional performance measurements and an independent acceptance review.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [course-progress.md](course-progress.md). Do not infer numerical correctness, low heat, zero Swap, performance or hardware support merely from source availability.
