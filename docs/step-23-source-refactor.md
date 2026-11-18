# Step 23 — Production-style Source Refactor

> Step 23.1–23.5 publish one subsection per commit. No data access, model construction, notebook execution or testing occurs during material publication.

## Step 23.1 — Data module

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
