# Course-material progress — Step 25.5 published (authoritative)

**Current material checkpoint — 2026-11-18:** Step 23.1–25.5 的 **15 份新母版（102–116）已发布到 GitHub main**，仓库共有 **116 份教学母版**。这只表示教材、源码、测试/测量脚本与文档已准备；**不表示学员已经学完、独立审查此次修订通过，或运行测试/训练/采样/benchmark 成功**。没有访问用户 `notebooks/work/`，也没有创建 Git tag / GitHub Release。

**独立复核集中修订（本轮首次修正，按既有上限累计）：** 23.1–23.5、24.1–24.2、25.3–25.5 各 **1/2**；未修改的 24.3–24.5、25.1–25.2 保持 **0/2**。Step 22.5 及更早的记录、修正次数、独立验收均原样保留。此轮只做静态源码/文稿检查与 Git 发布，未运行课程 Notebook、测试、训练、生成、评测或 benchmark。完整历史材料提交表和最新静态范围参见 [RELEASE_NOTES](RELEASE_NOTES.md)、[REPRODUCIBILITY](REPRODUCIBILITY.md) 和 [RELEASE_CHECKLIST](RELEASE_CHECKLIST.md)。

**第二次教学补充后的累计台账（最终）：** 仅 **Step 23.3、Step 23.5 为 2/2**；其余第一轮已修订小节继续为原 **1/2**，未修订小节保持原 **0/2**。上方“首次修正 1/2”保留为历史快照，不被覆盖。此轮只增加 104、106 母版与同一 Step 23 讲义的 Markdown，未执行课程代码，旧验收与 Step 22.5 及更早记录不变。

## Historical snapshot — Step 22.5 material publication (superseded as current)

当前已发布至 **Step 22.5**（10/25 个新小节）。最新授权允许按原路线自主补齐 Step 21–25；尚未发布的后续小节不计为完成。教材完成不代表学员掌握或运行通过。

只进行材料编写、静态审阅和 Git 发布。没有执行 Notebook/训练/验证/测试/采样/benchmark/checkpoint，没有读取或下载语料，没有执行 work copies。Release tag/GitHub Release 未创建。
## New masters published

- [92_step21_1_generation_loop](../notebooks/92_step21_1_generation_loop.ipynb)
- [93_step21_2_greedy_vs_sampling](../notebooks/93_step21_2_greedy_vs_sampling.ipynb)
- [94_step21_3_temperature](../notebooks/94_step21_3_temperature.ipynb)
- [95_step21_4_top_k_sampling](../notebooks/95_step21_4_top_k_sampling.ipynb)
- [96_step21_5_context_cropping](../notebooks/96_step21_5_context_cropping.ipynb)
- [97_step22_1_weight_tying](../notebooks/97_step22_1_weight_tying.ipynb)
- [98_step22_2_rotary_position_embeddings](../notebooks/98_step22_2_rotary_position_embeddings.ipynb)
- [99_step22_3_swiglu_gated_mlp](../notebooks/99_step22_3_swiglu_gated_mlp.ipynb)
- [100_step22_4_mqa_gqa_attention](../notebooks/100_step22_4_mqa_gqa_attention.ipynb)
- [101_step22_5_flash_style_attention](../notebooks/101_step22_5_flash_style_attention.ipynb)

## Step 23.1 — Data module
- [Master](../notebooks/102_step23_1_data_module.ipynb) · [src/minigpt/data.py](../src/minigpt/data.py) · [Step 23 notes](step-23-source-refactor.md)
- Published material only; static check performed, not executed. Initial substantive corrections 0/2. Next: 23.2.

## Step 23.2 — Model module
- [Master](../notebooks/103_step23_2_model_module.ipynb) · [src/minigpt/model.py](../src/minigpt/model.py) · [Step 23 notes](step-23-source-refactor.md)
- Static code extraction, no execution, no checkpoint load. Initial corrections 0/2. Next: 23.3.

## Step 23.3 — Training module
- [Master](../notebooks/104_step23_3_training_module.ipynb) · [Source](../src/minigpt/train.py) · [Step 23 notes](step-23-source-refactor.md)
- Source and teaching material published; static-only, not executed. Initial corrections 0/2.

## Step 23.4 — Generation module
- [Master](../notebooks/105_step23_4_generation_module.ipynb) · [Source](../src/minigpt/generate.py) · [Step 23 notes](step-23-source-refactor.md)
- Source and teaching material published; static-only, not executed. Initial corrections 0/2.

## Step 23.5 — Configuration and CLI
- [Master](../notebooks/106_step23_5_configuration_cli.ipynb) · [config.py](../src/minigpt/config.py) · [cli.py](../src/minigpt/cli.py) · [Step 23 notes](step-23-source-refactor.md)
- Step 23.1–23.5 teaching/source files published; NOT run. No runtime test/packaging certification. Next 24.1; initial corrections 0/2.

## Step 24.1 — Unit tests
- [Master](../notebooks/107_step24_1_unit_tests.ipynb) · [tests/test_data.py](../tests/test_data.py) · [tests/test_model.py](../tests/test_model.py) · [tests/test_checkpoint.py](../tests/test_checkpoint.py) · [tests/test_generate.py](../tests/test_generate.py) · [Step 24 notes](step-24-testing-benchmarking.md)
- Published as static artifacts, NOT executed; initial corrections 0/2.

## Step 24.2 — Tiny overfit test
- [Master](../notebooks/108_step24_2_tiny_overfit_test.ipynb) · [scripts/tiny_overfit.py](../scripts/tiny_overfit.py) · [Step 24 notes](step-24-testing-benchmarking.md)
- Published as static artifacts, NOT executed; initial corrections 0/2.

## Step 24.3 — Memory benchmark
- [Master](../notebooks/109_step24_3_memory_benchmark.ipynb) · [scripts/benchmark_memory.py](../scripts/benchmark_memory.py) · [Step 24 notes](step-24-testing-benchmarking.md)
- Static materials only; no measured results, corrections 0/2.

## Step 24.4 — Performance / thermal protocol
- [Master](../notebooks/110_step24_4_performance_thermal_benchmark.ipynb) · [scripts/benchmark_performance.py](../scripts/benchmark_performance.py) · [Step 24 notes](step-24-testing-benchmarking.md)
- Static materials only; no measured results, corrections 0/2.

## Step 24.5 — Benchmark table for README
- [Master](../notebooks/111_step24_5_readme_benchmark_table.ipynb) · [docs/benchmark-report-template.md](../docs/benchmark-report-template.md) · [Step 24 notes](step-24-testing-benchmarking.md)
- Static materials only; no measured results, corrections 0/2.

## Step 25.1 — Rewrite README
- [Master](../notebooks/112_step25_1_readme_rewrite.ipynb) · [README](../README.md) · [Step 25 notes](step-25-release.md)
- Comprehensive README revised, benchmark values remain Not measured. Initial corrections 0/2; next 25.2.

## Step 25.2 — Documentation index
- [Master](../notebooks/113_step25_2_documentation_index.ipynb) · [Complete INDEX](INDEX.md) · [Step 25 notes](step-25-release.md)
- Step1–25 docs and actual published masters indexed; no runtime, initial corrections 0/2. Next 25.3.

## Step 25.3 — Reproducibility audit (static evidence)
- [Master](../notebooks/114_step25_3_reproducibility_audit.ipynb) · [Audit guide](REPRODUCIBILITY.md) · [Static audit script](../scripts/static_release_audit.py) · [Step 25 notes](step-25-release.md)
- Pin/link/ignore inspected from remote; audit script and all runtime gates NOT run. Corrections 0/2; next 25.4.

## Step 25.4 — Credits, references, license
- [Master](../notebooks/115_step25_4_credits_references_license.ipynb) · [Credits](CREDITS.md) · [MIT LICENSE](../LICENSE) · [Step 25 notes](step-25-release.md)
- Static attribution check only, no tests/data downloads. Corrections 0/2; next 25.5.

## Step 25.5 — Final materials / release-preparation closure
- [Master](../notebooks/116_step25_5_final_release.ipynb) · [Release Notes](RELEASE_NOTES.md) · [Release Checklist](RELEASE_CHECKLIST.md) · [Step 25 notes](step-25-release.md)
- All 15 Step 23–25 masters published on main, initial corrections 0/2. **Runtime, fresh clone, tag and GitHub Release NOT performed.**

## Compact Step 23–25 commit ledger

| Step | Master | Verified material commit |
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
