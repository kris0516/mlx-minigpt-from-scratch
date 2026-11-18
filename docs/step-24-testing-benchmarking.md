# Step 24 — Testing, Benchmarks and Profiling

> All tests and benchmarks in this section are **source code only**. They were not executed and have no observed numerical results. Run only in independently controlled work sessions.


## Step 24.1 — Unit tests

# Step 24.1 — Unit tests

**命令 1：**

~~~bash
git pull
~~~

**命令 2（新时间戳工作副本，cp -n 不覆盖）：**

~~~bash
cp -n notebooks/107_step24_1_unit_tests.ipynb notebooks/work/107_step24_1_unit_tests_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

Apple Silicon / Python 3.12 / MLX v0.32.2。母版只供阅读和复制；所有课程代码/测试保持未执行。

## 1. 为什么新增本节（Motivation）

Step 23 已有可导入的 src 模块，本节把可验证的接口写成独立 unittest 文件。**测试代码已编写**不等于测试通过。本节禁止在材料制作时运行测试，也不新增自动触发的 CI 工作流。优先检验 next-token shift、dense causal mask、模型形状、weights-only save/load，以及生成历史与随机 key。

本节完整可运行的源代码单独存放于 tests/test_data.py、tests/test_model.py、tests/test_checkpoint.py、tests/test_generate.py，而不是把整段复用代码复制进讲解。

## 2. 输入输出、shape 与预期不变量

Data: (N,) → (B,T) 的 X/Y；Model: (B,T) → (B,T,V)；mask: (T,T)；save/load 的同一输入应得到相同 (B,T,V) 输出。该测试选择 B=2,T=3,V=8,C=16,H=4，head_dim=4。

## 3. 纸面例子（不是实测）

人工指定 scores 是 3×3 全零矩阵，causal 过滤后第一行只能看 key0，第二行看 key0/key1，第三行看全部。未来用 mx.softmax 得到上三角严格为 0。这个推导只说明期望，当前没有测试输出。

## 4. 限制、验证规则与练习

Python 标准库 unittest 不增加 pytest 依赖，uv.lock 不修改。save/load 测试将来会写入自动清理的临时目录，**当前没有任何写入**。测试基于小数组、短序列、单头/多头条件；通过也不代表所有 dtype、长上下文、性能、恢复训练状态均正确。

**练习（无答案）：** ① 解释 shape 测试为何不能独自证明 causal mask 正确；② 设计一个 non-square causal mask 反例并说明不属于当前等长参考；③ 哪个测试应拒绝长度超过位置表容量？

**状态：** 本节仅静态编写、检查和发布，初始修正 0/2；不声称独立验收、运行通过、时间/内存/温度读数。继续下一小节。

Code files: [tests/test_data.py](../tests/test_data.py), [tests/test_model.py](../tests/test_model.py), [tests/test_checkpoint.py](../tests/test_checkpoint.py), [tests/test_generate.py](../tests/test_generate.py).


## Step 24.2 — Tiny overfit test

# Step 24.2 — Tiny overfit test

**命令 1：**

~~~bash
git pull
~~~

**命令 2（新时间戳工作副本，cp -n 不覆盖）：**

~~~bash
cp -n notebooks/108_step24_2_tiny_overfit_test.ipynb notebooks/work/108_step24_2_tiny_overfit_test_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

Apple Silicon / Python 3.12 / MLX v0.32.2。母版只供阅读和复制；所有课程代码/测试保持未执行。

## 1. 为什么新增本节（Motivation）

Tiny overfit（极小数据过拟合）是一种 end-to-end sanity check：如果连固定小序列都学不动，先排查数据 shift、loss、gradient、Adam、mask，不应扩大训练规模。新文件是 scripts/tiny_overfit.py，只有未来明确运行入口，发布时不会训练。

本节完整可运行的源代码单独存放于 scripts/tiny_overfit.py，而不是把整段复用代码复制进讲解。

## 2. 输入输出、shape 与预期不变量

固定 X=(1,4)、Y=(1,4)，V=3,C=16,H=4,N=1,Tmax=4。logits (1,4,3)，loss ()；每一次调用对应一个 Adam 更新，返回的训练 loss 来自更新前参数。脚本训练后另算当前参数上的 final_loss，才与 initial_loss 对照。

## 3. 纸面例子（不是实测）

初始未学习时若 3 个 logits 相同，则每个位置交叉熵为 ln3，四位置均值仍为 ln3。训练的目标是让 loss 对这组固定样本下降，但**实际起点不是上述假想均匀分数**，它取决于随机初始化，不能伪填 ln3 或某条曲线。

## 4. 限制、验证规则与练习

样本只有 4 个监督位置，50 updates 是谨慎的小规模候选，不是普适收敛保证。脚本最后根据真实运行结果报告 initial/final loss、loss_decreased，并在未下降时抛错误；不预先填结果。一个 toy 能过拟合不能证明泛化、无 Swap、低热或大模型稳定。

**练习（无答案）：** ① 列出第一次更新前和第 50 次更新后 loss 的参数版本；② 若 loss 不下降，应优先审查哪些数据/梯度接口？③ 为什么不能把过拟合成功当作验证集表现好？

**状态：** 本节仅静态编写、检查和发布，初始修正 0/2；不声称独立验收、运行通过、时间/内存/温度读数。继续下一小节。

Code files: [scripts/tiny_overfit.py](../scripts/tiny_overfit.py).


## Step 24.3 — Memory benchmark

# Step 24.3 — Memory benchmark

**命令一：**

~~~bash
git pull
~~~

**命令二（cp -n 不覆盖工作笔记）：**

~~~bash
cp -n notebooks/109_step24_3_memory_benchmark.ipynb notebooks/work/109_step24_3_memory_benchmark_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

Apple Silicon / Python 3.12 / MLX v0.32.2；本轮未运行任何 Notebook、测试、训练、采样、benchmark 或代码验证。

## 1. Motivation / 本节新增概念

本节新增显式 MLX allocator memory 统计协议，而非复述 18.5。源码只测量**将来某次**小模型 forward 期间 active/cache/peak，使用 one process per configuration 规避旧进程缓存污染。脚本在 import 阶段不构造模型、读取系统或计时。

## 2. Input, output 与 shape

输入 ModelConfig: B=1、T=16、C=32、H=4、N=1、V=32；token_ids shape (1,16)，输出 raw logits (1,16,32)。手工逻辑 score shape (1,4,16,16)，如果 float32 为 4096 bytes；它只是理论单一中间张量的尺寸，不是实测分配。

## 3. 纸算小例子（不是实测）

如果 baseline active=1000 bytes，reset 后 raw peak=1500 bytes，则推导的 window peak=max(1000,1500)=1500，above-baseline=500 bytes。若 raw peak=0，则同样要用起点 active 做区间高点。以上数字是演算假设，与 Apple Silicon 设备无关，**不是测量数据**。

## 4. 实现边界与未来验证

mx.get_active_memory/mx.get_cache_memory/mx.get_peak_memory 返回分配器 bytes，不等于整机 RAM、物理驻留或 macOS Swap；mx.reset_peak_memory 在固定版重置 raw peak 为 0，不是起点 active。统计前后显式 mx.eval，输出 JSON 在未来调用时才产生；默认 stdout，不写文件。避免假称零 Swap 或低热。

完整可选程序/表格在 scripts/benchmark_memory.py；此 Notebook 只标出本节新增的指标，不复制长源码。

**练习（无答案）：** ① 计算 B=2,H=4,T=64 的 fp32 score 逻辑字节数；② 解释 raw peak 为0 但当前 active>0 的情况；③ 为什么两种配置应分别在干净进程观察？

**状态：** 静态材料初稿 0/2；没有运行输出或性能结论。

Full artifacts: [scripts/benchmark_memory.py](../scripts/benchmark_memory.py).


## Step 24.4 — Performance / thermal protocol

# Step 24.4 — Performance / thermal protocol

**命令一：**

~~~bash
git pull
~~~

**命令二（cp -n 不覆盖工作笔记）：**

~~~bash
cp -n notebooks/110_step24_4_performance_thermal_benchmark.ipynb notebooks/work/110_step24_4_performance_thermal_benchmark_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

Apple Silicon / Python 3.12 / MLX v0.32.2；本轮未运行任何 Notebook、测试、训练、采样、benchmark 或代码验证。

## 1. Motivation / 本节新增概念

本节将 Step 19.5 的测量意识与 Step 21 的生成路径分开：prefill forward throughput 和无 KV-cache 生成 throughput 是两种指标。代码仅为未来短时、谨慎 benchmark 准备，**没有真实计时、温度或功耗观测**。

## 2. Input, output 与 shape

未来输入 token IDs (1,T)，MiniGPT logits (1,T,V)；一次 forward latency 的计时窗口从 perf_counter 开始，到 mx.eval(output) 完成。生成从一项 ID 开始，追加 n 项，返回 (1,1+n)。输出分别是 positions/second=T/median_forward_seconds 和 generated_tokens/second=n/median_generate_seconds。

## 3. 纸算小例子（不是实测）

假设（纯纸算）T=16，某次同步 forward 用 0.008 s，则这次 16/0.008=2000 positions/s；若生成 5 个 token 用 0.05 s，则 100 generated tokens/s。两种单位不同，不能把 2000 与 100 简单当成同任务加速比。假设值不得进入 README 实测表。

## 4. 实现边界与未来验证

脚本仅报告 wall-clock，非 GPU kernel-only 时间；warmups 不保证稳定态。生成仍每步重新计算窗口，未实现 KV cache。没有读取温度传感器、功率计或系统 Swap，因此相关 JSON 字段保留 null。Apple Silicon unified memory 与 allocator 统计不可冒充热测数据；不要用长时间满载实验作为本轮检查。

完整可选程序/表格在 scripts/benchmark_performance.py；此 Notebook 只标出本节新增的指标，不复制长源码。

**练习（无答案）：** ① 解释为什么若把 mx.eval 放在计时窗口外，MLX lazy evaluation 会污染结果；② 编写 forward 与 decode 的不同吞吐单位；③ 需要什么额外仪器才可以宣称实际设备温度曲线？

**状态：** 静态材料初稿 0/2；没有运行输出或性能结论。

Full artifacts: [scripts/benchmark_performance.py](../scripts/benchmark_performance.py).


## Step 24.5 — Benchmark table for README

# Step 24.5 — Benchmark table for README

**命令一：**

~~~bash
git pull
~~~

**命令二（cp -n 不覆盖工作笔记）：**

~~~bash
cp -n notebooks/111_step24_5_readme_benchmark_table.ipynb notebooks/work/111_step24_5_readme_benchmark_table_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

Apple Silicon / Python 3.12 / MLX v0.32.2；本轮未运行任何 Notebook、测试、训练、采样、benchmark 或代码验证。

## 1. Motivation / 本节新增概念

前两节只准备统计程序，本节把未来报告所需字段整理为固定 README 表格与 docs/benchmark-report-template.md。由于本轮禁止 runtime benchmark，所有数值栏必须写 Not measured，而不是 0、不填虚构 tokens/s、也不宣称无 Swap。

## 2. Input, output 与 shape

每行至少包含硬件与环境、C/H/N、T/B、dtype/MLX 版本、MLX active/peak bytes、prefill positions/s、generation tokens/s、macOS Swap/thermal 来源。README 先展示摘要列，完整 JSON/设备设置留作后续可追溯记录。

## 3. 纸算小例子（不是实测）

配置 A 为 C=32,H=4,N=1,T=16,B=1，仅列出形状与候选参数，不预测 active 或吞吐；配置 B 是 C=64,H=4,N=2,T=32,B=1。二者不能据理论 score 大小直接填实测 peak。

## 4. 实现边界与未来验证

模板本身没有数据来源，任何将来填值都应同步记录 Git SHA、macOS/SoC/MLX、输入 shape/dtype、预热/同步边界、测量次数。温度与系统 Swap 必须来自对应工具，不能从 MLX allocator counters 推断。

完整可选程序/表格在 docs/benchmark-report-template.md；此 Notebook 只标出本节新增的指标，不复制长源码。

**练习（无答案）：** ① 说明为什么 Not measured 不等于 0；② 区分 peak_active 与 end_active 两个列名；③ 列出最少需要记录的三个可复现环境字段。

**状态：** 静态材料初稿 0/2；没有运行输出或性能结论。

Full artifacts: [docs/benchmark-report-template.md](../docs/benchmark-report-template.md).
