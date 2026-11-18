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
