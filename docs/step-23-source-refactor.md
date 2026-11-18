# Step 23 — Production-style Source Refactor

> Step 23.1–23.5 publish one subsection per commit. No data access, model construction, notebook execution or testing occurs during material publication.

## Step 23.1 — Data module

**工作副本的导入位置（Working-directory discovery）**：下列代码在仓库 root、`notebooks/` 或 `notebooks/work/` 启动 kernel 时，均从 `Path.cwd()` 向上查找同时含 `pyproject.toml` 与 `src/minigpt/__init__.py` 的目录，再将其 `src` 加入 `sys.path`。这不是依赖单一 cwd 的硬编码路径；如果 kernel 在仓库树外启动，将明确抛出 `FileNotFoundError`。这段逻辑只负责定位代码，**没有运行课程代码，也不是自动安装**。

**五份母版共用的导入设置（在各自 code cell 内）：**

```python
from pathlib import Path
import sys

repo_root = next(
    (p for p in (Path.cwd(), *Path.cwd().parents)
     if (p / "pyproject.toml").is_file()
     and (p / "src" / "minigpt" / "__init__.py").is_file()),
    None,
)
if repo_root is None:
    raise FileNotFoundError("MiniGPT repository not found above kernel working directory")
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
```


# Step 23.1 — Data module：把数据处理从 Notebook 迁移为可复用模块

**本节新增：** 将 Step 16.1–16.3 的字符词表、连续 train/validation 切分和 shifted-window sampler 放入 `src/minigpt/data.py`。不是新 tokenizer，也不是重新获取 Tiny Shakespeare。导入模块没有文件访问或数据下载。

**终端命令一（在仓库根目录，独立执行）：**
```bash
git pull
```

**终端命令二（独立执行、不覆盖 work 笔记）：**
```bash
cp -n notebooks/102_step23_1_data_module.ipynb notebooks/work/102_step23_1_data_module_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

固定 Apple Silicon、Python 3.12、MLX v0.32.2；只编辑 work 副本，不运行母版。本节代码与所有练习均未执行，静态编写不等于运行通过。

## 1. 为什么要抽成模块？（module / separation of concerns）

Step 16 的 Notebook 已教会数据来源、词表映射、切分与采样，但若每次训练或验证都复制同一段函数，容易产生多个漂移版本。**Refactor（重构）**强调在保持输入输出契约的前提下改变代码组织，不在本节改变字符级方案或训练/验证边界。

新文件 `src/minigpt/data.py` 提供六个独立入口：`load_text`（仅显式调用时读本地 UTF-8）、`build_vocabulary`、`encode`、`decode`、`split_train_validation`、`get_batch`。

`src/minigpt/__init__.py` 只定义包版本；不是在 import 时悄悄创建模型或访问语料。完整源码属于配套模块，Notebook 只演示新增公共契约。

## 2. 数据流与 shape（Input / output contract）

设原始字符数为 N，实际字符词表大小为 V；文本先用已固定的 `stoi` 映射得到整数 IDs，`tokens` shape 为 `(N,)`，dtype `int32`。

连续切分：`tokens[:cut]` → `train_tokens: (N_train,)`，`tokens[cut:]` → `val_tokens: (N_val,)`；**两个 split 的样本窗口绝不跨边界**。调用 `get_batch(one_split,B,T,rng)` 在每一行取 T+1 个 ID，组成：

| 对象 | Shape | 内容 |
|---|---|---|
| X | `(B,T)` | 每段窗口的前 T 个 token |
| Y | `(B,T)` | 同一段向右偏移一格的 T 个目标 |
| starts | Python `list[int]` 长 B | 每段在*当前 split 内*的起点 |

`randrange(N_split - T)` 上界不包含自身；若 N_split=T+1，唯一合法起点是 0。必须保留显式传入的 `random.Random` 对象，才能管理训练采样状态。

## 3. 手算最小例子

人为指定字符串 `"ababa"`，排序后的词表为 `{"a":0,"b":1}`，token 流 `[0,1,0,1,0]`。取 block_size=2、某一合法起点 index=1，窗口是 `[1,0,1]`，故 X=`[1,0]`、Y=`[0,1]`。这是先验数字演示，不是本次随机采样或 MLX 运行结果。

`get_batch` 实际使用传入 RNG 选择起点，因此不能把上述 index=1 当作某个 seed 已经观察到的值。

## 4. 代码位置与验证边界

本节新增代码在 `src/minigpt/data.py`；上面的 16 行仅说明如何 import 和调用，不重复抄写整个模块。以后 src 模块需要在 `PYTHONPATH=src`、项目安装或本页明确的 import-path 条件下被发现。

**可静态预测：** 词表 V=2，原始 IDs shape (5,)，切分 (3,)/(2,)，批次 X/Y 为 (1,2)；Y 比 X 向右移一位。具体 starts、X/Y 元素需实际运行才有结果，本轮没有。

**使用限制：** 非空语料；未知字符拒绝；切分两端都非空；batch/T 必须是正整数；split 长度至少 T+1。没有做并行数据加载、token packing、BPE、streaming、自动下载或隐式训练。

**练习（不附答案）：** ① 手列 `"cabca"` 的 stoi/itos 与 token IDs；② 在 N=7,T=3 时列合法起点范围；③ 说明为什么 train/val 分开调用 sampler 比在总流采样后再给窗口贴标签更安全。

## 5. 完成状态

文件只做静态审阅；Notebook 所有 code cells 的 execution_count=null、outputs=[]。没有读语料、执行采样、导入 MLX、跑测试或改动 work 副本。当前源码属于已准备、未运行的教学材料；继续顺序进入 Step 23.2。

Module: [src/minigpt/data.py](../src/minigpt/data.py). Newly introduced public functions are tested only in a future user-authorized runtime session, never inferred as PASS here.


## Step 23.2 — Model module

# Step 23.2 — Model module：将已验收 MiniGPT 从 Notebook 抽取为源码

**命令 1（仓库根目录）：**

~~~bash
git pull
~~~

**命令 2（独立执行、保护 work 副本）：**

~~~bash
cp -n notebooks/103_step23_2_model_module.ipynb notebooks/work/103_step23_2_model_module_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

只修改新的 work 副本；母版与源码均未执行。环境保持 Apple Silicon、Python 3.12、MLX v0.32.2。

## 1. 为什么要把模型放进单独模块（Refactor, not redesign）？

旧母版 notebooks/65_step15_4_implement_minigpt.ipynb 已有四个核心类。后续脚本反复复制相同类定义，会导致版本分叉，进而改变权重路径或 checkpoint 的含义。本节新建 src/minigpt/model.py，**直接复用原母版的类代码**，而不是引入新的 attention 算法。

保留 CausalMultiHeadAttention、FeedForwardMLP、TransformerBlock、MiniGPT 四个类；仍使用 learned absolute position embedding、untied LM head、dense causal attention、GELU 和 Pre-Norm RMSNorm。22.1–22.5 中的 RoPE、SwiGLU、GQA、Flash-style、KV cache 都没有混入默认架构。

## 2. 前向 shape（Input/output contract）

取 B=2、T=3、C=16、H=4、D=4、V=11、N=2 作为纸面例子：

| 层或中间量 | shape |
|---|---|
| token IDs | (2,3) int32 |
| token embeddings / position embeddings | (2,3,16) / (3,16) |
| 单次 QKV projection | (2,3,48)，切成三个 (2,3,16) |
| Q/K/V head-major | (2,4,3,4) |
| dense scores / attention weights | (2,4,3,3) |
| merged head output / Block | (2,3,16) |
| raw logits | (2,3,11) |

约束 C % H = 0；1 ≤ T ≤ max_seq_len。Position IDs 是 forward-local 的 arange(T)，causal mask 在 forward 内构造，不会进入模型参数树。输出为 raw logits，不是分类概率。

## 3. 小例子与源码职责

本节新知识是**模块边界和兼容性**。长的四个类定义完整放在 src/minigpt/model.py，不在 Notebook 当作新的 150 行教学内容重复粘贴。下段仅演示包导入与未来的 shape 检查。模型参数都是随机新初始化，不能被误称为已恢复训练权重。

## 4. 边界、预期验证与练习

静态审查核对了原 15.4 代码中的 qkv_proj/c_proj 参数布局、forward-local mask、Block 清单与 untied lm_head。**源码抽取不等于 checkpoint 加载验证通过**。新模块也没有加入 fused kernel、混合精度或 cache。

未来预期 shape 为 (2,3,11)；本轮没有任何 forward 输出、速度、显存或执行成功记录。其兼容性需要在后续 Step 24 中实际运行测试才能确定。

**练习（不附答案）：** ① C=24,H=6,T=5 时写出 Q、score、head output 的 shape；② 解释更换 RoPE 为何可能影响旧 checkpoint；③ 找到 mask 不被登记为 trainable 参数的代码位置。

静态材料初稿 0/2；下一节是 23.3 Training module。

[Source module](../src/minigpt/model.py) is an extraction of [Step 15.4](../notebooks/65_step15_4_implement_minigpt.ipynb), not a runtime-tested replacement.


## Step 23.3 — Training module

# Step 23.3 — Training module

**命令一：**

~~~bash
git pull
~~~

**命令二：**

~~~bash
cp -n notebooks/104_step23_3_training_module.ipynb notebooks/work/104_step23_3_training_module_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

只在独立 work 副本运行/修改；本轮只静态准备，未执行任何 Notebook、MLX、训练、生成或 benchmark。Apple Silicon / Python 3.12 / MLX v0.32.2。

## 1. 动机（Motivation）

前面 17.5、19.2、19.3 已把 loss、gradient、Adam、validation 放在 Notebook。本节将这些接口迁到 src/minigpt/train.py；新增的是函数之间的清晰契约，不是更大的模型或已执行训练。仍使用 uncompiled step，便于看清显式 mx.eval 边界与变化的 optimizer state。

完整运行实现放在 **src/minigpt/train.py**，以下 Notebook 只教学本小节新增接口，不复制旧类的长代码。

## 2. Shape / Input–Output Contract

输入：X/Y 都是 (B,T) int32；MiniGPT 产生 logits (B,T,V)。将 logits 展平为 (B*T,V)，targets 展平为 (B*T,)，cross_entropy 后均值为标量 ()。nn.value_and_grad 返回 model 参数树的梯度；global_grad_norm 把所有梯度叶子平方归约为一个标量。make_train_step 绑定同一 model 和 Adam，返回 (loss_before_update, preclip_norm)，各 shape ()。验证函数只前向，不更新 Adam。

## 3. 手算小例子（Paper example, not runtime output）

令 B=1,T=2,V=3，原 logits 为两个全 0 行，targets=[0,2]。每位置 softmax 为 [1/3,1/3,1/3]，交叉熵平均为 ln(3)。这是纸上数值，不代表随机模型执行输出。该 loss 是当前参数更新**之前**的版本；下一次 validation 用的是更新后的参数。

## 4. 限制、预期验证与静态结论

Validation batches require an independently owned random.Random; training sampling must not reuse validation RNG. max_grad_norm=None disables clipping; a positive finite value enables global clipping. Nonfinite loss/grad norm abort before optimizer.update, but other runtime failures are not promised away. The returned metrics are Python numbers only when explicitly called. No torch, autocast, graph compilation, automatic checkpoint, no fabricated loss curve.

未来可验证的前提是合适的真实模型、正确的数组和显式调用；静态审核无法保证任意设备上通过，也没有任何读数、模型生成或训练记录。

**练习（不附答案）：** ① 列出 B=4,T=8,V=32 时展平前后 shapes；② 解释为什么返回 old-parameter loss 不等于更新后的 validation loss；③ 画出训练 RNG 和验证 RNG 的分开调用路线。

本节材料初稿，实际修正 0/2；按用户新授权自动继续下一小节。

[Complete source](../src/minigpt/train.py) is an implementation intended for later execution, not proof of runtime acceptance.


### 23.3 真实 API / 时间线（教学调用，不执行）
假设调用前已经分别准备好同一组 `model`、`optimizer`、`train_tokens`、`val_tokens`；以下为与 `src/minigpt/train.py` 函数签名对齐的**未来调用片段**，不是本 Notebook 自带模型或训练入口：

```python
from random import Random
train_rng, val_rng = Random(2026), Random(2027)  # 两套独立 Python RNG 状态
train_step = make_train_step(model, optimizer, max_grad_norm=1.0)
history = run_training(
    model, train_step, train_tokens,
    batch_size=4, block_size=64, updates=4, train_rng=train_rng,
    val_tokens=val_tokens, val_rng=val_rng,
    val_every=2, val_batches=2,
)
```

| 循环内 index | `record["update"]` | 假设进入循环前 Adam step=s | loss / validation 使用哪版参数 |
|---|---|---|---|
| 1 | 1 | 更新后 s+1 | 训练 loss：`θ_s`；本轮无 validation |
| 2 | 2 | 更新后 s+2 | 训练 loss：`θ_(s+1)`；validation：`θ_(s+2)` |

`record["update"]` 来自本次 `range(1, updates+1)`，再次调用 `run_training` 会从 1 重新编号；**Adam 的 `optimizer.state["step"]` 则在原对象中继续累计**，两者不可混为同一个进度字段。`train_rng` 与 `val_rng` 是两个独立的 `random.Random` 对象，**不是 MLX random key**；验证采样不会消费训练 RNG。

实际源码先调用 `train_step` 完成 Adam 更新，再生成当前 `record`，然后视 `val_every` 执行验证；验证成功后才 `records.append(record)`。若验证途中报错，`estimate_validation_loss` 的 `finally` 恢复训练模式，但**此前 Adam 更新不会自动回滚**；当次 record 尚未 append，整个函数可能直接抛异常而不返回 history。此示例没有运行，不声称上述数值是观测结果。

## Step 23.4 — Generation module

# Step 23.4 — Generation module

**命令一：**

~~~bash
git pull
~~~

**命令二：**

~~~bash
cp -n notebooks/105_step23_4_generation_module.ipynb notebooks/work/105_step23_4_generation_module_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

只在独立 work 副本运行/修改；本轮只静态准备，未执行任何 Notebook、MLX、训练、生成或 benchmark。Apple Silicon / Python 3.12 / MLX v0.32.2。

## 1. 动机（Motivation）

Step 21.1/21.5 的生成循环仍留在 Notebook 中。本节把 greedy 与按温度的 categorical sampling 变成可导入的两个函数，保持已验收模型无 cache 的行为；新的职责是封装输入检查、窗口切片、显式随机 key 与推理模式恢复，不是训练或发明新解码架构。

完整运行实现放在 **src/minigpt/generate.py**，以下 Notebook 只教学本小节新增接口，不复制旧类的长代码。

## 2. Shape / Input–Output Contract

模型读取 context (1,min(L+r,W))，返回 raw logits (1,T,V)。取 logits[:, -1, :] -> last (1,V)，greedy argmax 或 sample 选择 token (1,)，token[:,None] -> (1,1)，沿 axis=1 拼到完整 history (1,L+r+1)。返回的 greedy history (1,L+n)；sampling 额外返回 next_key (2,) uint32（固定 MLX key 约定）。W<=model.max_seq_len；n=0 无 forward。

## 3. 手算小例子（Paper example, not runtime output）

prompt IDs=[0,1]、block_size=3，追加 3 个 token。三轮 forward 的 context 长度为 2、3、3；最终 history 长 5。不预言具体 token，因为真实模型和参数未知。对于已知 logits [0, ln2]，greedy 选 ID 1，按温度 1 抽样则理论概率为 [1/3,2/3]；实际 draw 不填写。

## 4. 限制、预期验证与静态结论

Crop 保留返回历史但每轮重新计算整个窗口，**没有 KV cache**。旧模型 learned position 在新窗口从 0 编号，不能无损复用旧 K/V。该模块目前只处理单条 prompt、固定 n 个 token；没有 EOS、batch、top-k 或 top-p；也未实现 GPU fused sampling。Sampling 每轮 split 显式 key，不使用隐式全局 key；温度必须为正有限。

未来可验证的前提是合适的真实模型、正确的数组和显式调用；静态审核无法保证任意设备上通过，也没有任何读数、模型生成或训练记录。

**练习（不附答案）：** ① 写出 L=5,W=3,n=2 的每轮 context 长度和返回 shape；② 解释为何 sample 的 key 必须随轮次更新；③ 说明 why context cropping 不能替代 KV cache。

本节材料初稿，实际修正 0/2；按用户新授权自动继续下一小节。

[Complete source](../src/minigpt/generate.py) is an implementation intended for later execution, not proof of runtime acceptance.


### 23.4 精确的生成轴与两条选择策略
`_last_logits` 的源码先得到 `logits = model(context)`，shape **`(1,T,V)`**，然后用 **`logits[:, -1, :] → last (1,V)`** 取最后一个 query 位置的全部词表分数；`last` 已经是二维数组，不能再对它执行 `last[:, -1, :]`。

`history` 保存完整 prompt + 已生成 token；每轮只把 `context = history[:, -block_size:]` 交给模型，新的 `token (1,1)` 要沿时间轴追加到 **history**，不是覆盖为短 context。当前只实现 **greedy argmax** 和 **全词表（full-vocabulary）temperature categorical sampling**；**没有 top-k、top-p 或 EOS**。Step 21.4 的 top-k 教材函数没有被集成进 `generate_sample`。

抽样使用 `next_key, draw_key = mx.random.split(key)`：`draw_key` 只供本轮 `mx.random.categorical`，而 `key = next_key` 留给下一轮并随 history 返回，不复用本轮抽样 key。两条生成函数都在 `model.eval()` 后将循环包进 `try/finally`，退出或抛异常时 `model.train(was_training)` 恢复调用前的模式；这不表示已成功生成，也不是发生错误时回滚 history。

## Step 23.5 — Configuration and CLI

# Step 23.5 — Configuration and CLI：可复现参数与显式运行入口

**命令一：**

~~~bash
git pull
~~~

**命令二：**

~~~bash
cp -n notebooks/106_step23_5_configuration_cli.ipynb notebooks/work/106_step23_5_configuration_cli_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

Apple Silicon、Python 3.12、MLX 0.32.2。Notebook 母版未运行，work 文件不覆盖。

## 1. 本节新问题：从 Notebook 的全局变量到可描述的运行配置

Step 23.1–23.4 已把数据、模型、训练、生成的实现拆为独立模块。本节新增 src/minigpt/config.py（dataclass 与校验）和 src/minigpt/cli.py（argparse 入口），让用户**明确选择**何时加载本地语料、开始训练或生成。

配置与 checkpoint 是两种东西：配置说明结构，weights 记录参数数值；训练完全恢复还需要 Adam moments、RNG state 等，不能用一个 JSON 假装包含所有内容。源文件 import 不自动运行 main。

## 2. Shape 与契约

ModelConfig 的 V 是真实词表大小；dims=C=64、heads=H=4、blocks=N=2、max_seq_len=Tmax=64 是候选配置，不是训练测得的最佳值。必须满足 C%H=0。TrainConfig 提供 B=4、block_size=64、updates=20、learning_rate=3e-4 与独立随机种子；必须满足 T<=Tmax。

**数据流（训练命令）**：显式 UTF-8 path → vocab/IDs (L,) → train/val splits → X,Y 各 (B,T) → MiniGPT logits (B,T,V) → scalar loss () → Adam 更新 → 新目录保存 model.safetensors 与 run.json。

**生成命令**：显式 checkpoint 目录 → 读取 run.json 与 model.safetensors → prompt IDs (1,Lp) → 每轮裁剪 context (1,min(Lp+r,Tmax)) → logits (1,T,V) → 追加到 history (1,Lp+r+1)。

## 3. 最小配置例子与严格边界

人为构造 V=3、C=16、H=4、N=1、Tmax=8，对应每头 d=4；若 token prompt 长 2、输出 3 个新 ID，最终 history shape 应是 (1,5)。只给定结构，不假设模型已经学会语言。

CLI 通过明确命令才访问磁盘、构造模型、训练、生成；本次**没有执行**下列命令。

~~~bash
# 仅未来由用户明确决定是否执行
PYTHONPATH=src uv run python -m minigpt.cli --help
PYTHONPATH=src uv run python -m minigpt.cli train --corpus data/tinyshakespeare/input.txt --output runs/example-unique
PYTHONPATH=src uv run python -m minigpt.cli generate --checkpoint runs/example-unique --prompt 'To be' --new-tokens 20
~~~

当前 pyproject.toml / uv.lock 不增加新打包依赖，因此使用 PYTHONPATH=src 而不是声称已安装 console_scripts。

## 4. 文件、练习与预期验证

src/minigpt/config.py 定义只保存 Python 标量的配置树；src/minigpt/cli.py 定义两个明确命令。train 产生的产物**只包含可用于推理的权重和描述**，不保存可恢复训练的 Adam 与精确 RNG 状态。不能把这份 CLI 自动等同于 Step 20.4 的完整恢复协议。

预期配置检查：C/H=4，T<=Tmax；输出模型的理论 logits shape 仍是 (B,T,V)。CLI 的 --help、实际 save/load、训练和生成均未运行，因此没有真实成功声明。

**练习（不附答案）：** ① 若 C=18,H=4，配置检查应如何失败？② 说明为何 `if __name__ == "__main__": main()` 避免 import 时启动 CLI，以及 `--corpus` 的实际用途；③ 比较推理权重文件与完整恢复训练状态需要的额外内容。

23.5 教材初稿，实际修正 0/2；完成大 Step 23 的静态材料准备，下一节 24.1 Unit tests。

Source modules: [config.py](../src/minigpt/config.py), [cli.py](../src/minigpt/cli.py), [__main__.py](../src/minigpt/__main__.py). Python package is imported using PYTHONPATH=src; uv.lock unchanged; no console entry point installed.


### 23.5 命令行与真实模型参数的逐层映射
训练子命令的 `--heads` 由 `argparse` 写到 `args.heads`；`train_command` 执行 `ModelConfig(num_heads=args.heads, ...)`，之后 `model_cfg.as_model_kwargs()` 把 `num_heads` 传入 `MiniGPT(**...)`。即 **`--heads → args.heads → ModelConfig.num_heads → MiniGPT(num_heads=...)`**。例如 `--dims 16 --heads 4` 的每头维度为 4；`dims % num_heads != 0` 会在配置检查时被拒绝。

**避免 import 副作用的是源码末尾 `if __name__ == "__main__": main()`，不是 `--corpus` 的 required 标志。** `--corpus` 只在真正解析 `train` 命令时要求使用者提供显式路径；仅 `import minigpt.cli` 不调用 `main()`，不读语料、构造模型或训练。生成子命令使用另一组 `--checkpoint/--prompt` 参数。

生成过程对 `run.json` 的 `itos` 长度/唯一性及模型 `vocab_size` 做一致性检查，并通过 `load_weights(..., strict=True)` 加载结构兼容的权重；但 **`(B,T,V)` logits shape 相同、参数名/shape 对得上，并不能证明 `model.safetensors` 与 `run.json` 出自同一次训练**，也不能证明词表的 ID→字符语义正确配对。目前没有内容哈希或原子配对验证协议，错误的同形状产物仍可能被误配。
