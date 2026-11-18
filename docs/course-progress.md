# Course-material progress — Step 22.5

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
