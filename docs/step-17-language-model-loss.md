# Step 17 — Language-Model Loss

本 Step 沿用现有 roadmap。当前只准备并发布 **17.1 Logits, probabilities, and cross-entropy**，初始生成实际修正计数 **0/2**，完成后停在17.1，不进入17.2。

材料审核方已独立核验大 Step 16 的全部母版、文档与完整回复，核心静态 PASS。这是材料验收，不是学员学习或运行反馈；已通过课节不返工。累计实际修正保留16.1=**1/2**、16.2=**1/2**、16.3至16.5各**0/2**，17.1初版=**0/2**。计数跨轮、跨对话不重置，后续只按请求逐小节至大 Step 20 全部完成后停止。

## Step 17.1 — Logits, probabilities, and cross-entropy

母版：[72_step17_1_logits_probabilities_cross_entropy.ipynb](../notebooks/72_step17_1_logits_probabilities_cross_entropy.ipynb)。本节把 MiniGPT 的 raw logits 接到“模型为正确下一 token 分配了多少概率”这一问题，再由此建立每个位置的 loss。

### Self-contained work copy and existing interfaces

新副本自带 `mlx.core as mx`、`mlx.nn as nn` 导入和手工 toy 数组。无需读取语料缓存、恢复旧kernel变量、调用sampler、构造模型或运行forward，也不 `%run` 旧课。以后使用时，选择项目 Python 3.12 / `.venv` kernel，先运行imports，再按母版次序阅读和使用准备的toy cells；本轮不执行。

衔接的两个既有接口是：

| 对象 | Shape / 含义 |
|---|---|
| MiniGPT raw logits | `(B,T,V)`，每个位置对V个词表类别的未归一化分数 |
| Shifted targets Y | `(B,T)`，每个位置的正确下一token类别ID |

B是batch行数，T是每行输入长度，V是词表类别数。固定一个位置 `(b,t)`，其 `logits[b,t,:]` 是长度V的分数向量，正确类别是单个整数 `Y[b,t]`。Y已经由16.3右移构造，loss处不再额外shift。

### Logits are scores, not probabilities

**Logit（未归一化分数）** 用来比较各类别的相对倾向，可以是负值，也不要求总和为1。MiniGPT的LM head返回这些分数，不在模型末端直接加入softmax。

**Probability（概率）** 用来描述模型当前给各类别分配的概率质量。对某个位置，它应当非负且总和为1。这里的模型概率不是已经验证过的经验正确率。

Token ID是类别标签，例如ID 2只标识某个token；它与ID 1、ID 0之间的数值距离不是语言模型应当最小化的误差。

### Softmax over the vocabulary axis

固定一个位置，记logits为 `z=(z_0,...,z_(V-1))`。v表示某个类别下标，`p_v`表示该类别的模型概率。`exp(a)=e^a` 是指数函数，取正值；求和符号 `sum` / `Σ` 表示将指定类别的项全部相加。

Softmax定义为：

\[
p_v=\frac{\exp(z_v)}{\sum_{u=0}^{V-1}\exp(z_u)}.
\]

分母对同一个位置的全部V个类别求和，所以这些概率合起来为1。代码明确使用：

```python
probabilities = mx.softmax(logits, axis=-1)
```

`axis=-1` 在 `(B,T,V)` 上指词表轴V，输出shape保持 `(B,T,V)`。不要沿B或T归一化。注意attention里的末轴曾表示key positions；API写法相同，轴的含义取决于实际张量。

### One hand-computable example

**Natural logarithm（自然对数）** 写作 `ln`，与指数函数互逆：`exp(ln(a))=a`，其中a>0。手工选择：

\[
z=[\ln1,\ln2,\ln3],\qquad y=2.
\]

y是正确类别ID。因为指数后的三个数是 `[1,2,3]`，总和为6，所以：

\[
p=[1/6,1/3,1/2].
\]

正确类别2得到概率1/2。对数只是为了方便构造可手算logits；这不表示MiniGPT会对token IDs或词频取对数来产生输出。

### Negative log-likelihood and categorical cross-entropy

**Likelihood（似然）** 在这里指：模型给这个已观察到的正确类别分配的概率 `p_y`。**Negative log-likelihood，NLL（负对数似然）** 为：

\[
\ell=-\ln p_y.
\]

对于0<p_y<=1，ln(p_y)<=0，因此负号使loss非负；正确类别概率越大，loss越小。手算例中：

\[
\ell=-\ln(1/2)=\ln2\approx0.6931.
\]

使用自然对数时，这个数采用自然对数单位；它不是一个错误百分比。

为了说明 **cross-entropy（交叉熵）** 的名称，先定义目标类别分布q。一般categorical cross-entropy为：

\[
H(q,p)=-\sum_{v=0}^{V-1}q_v\ln p_v.
\]

对于一个明确的正确类别y，数学上可写one-hot目标：正确位置 `q_y=1`，其他位置为0。于是只有正确类别那一项保留下来，交叉熵恰好等于NLL。本例 `q=[0,0,1]`，得到同一个 `ln2`。

这里的one-hot仅用于解释公式。现有Y继续保存整数类别ID，不在数据接口中改成 `(B,T,V)` one-hot。

### Stable calculation directly from logits

对正数a、b，有 `ln(a/b)=ln(a)-ln(b)`。将softmax代入NLL：

\[
\ell=\ln\left(\sum_v\exp(z_v)\right)-z_y.
\]

**Log-sum-exp** 简写LSE，表示先对各项取指数、求和，再取自然对数。`mx.logsumexp(z,axis=-1)` 提供这一运算，而无需先把所有概率显式算出来。

Softmax不因所有logits加上同一个常数而改变。取 `m=max(z)`、`shifted=z-m`，可写：

\[
\ell=\operatorname{LSE}(shifted)-shifted_y.
\]

这个写法避免对很大的原始logits直接取指数，也避免先形成极小概率再取log。单位置的准备代码核心是：

```python
m = mx.max(z, axis=-1, keepdims=True)
shifted = z - m
loss = mx.logsumexp(shifted, axis=-1) - shifted[y]
```

`keepdims=True` 保留被归约的轴，便于将最大值按原形状广播回去。`mx.log` 则是逐元素自然对数，输入须在其正数定义域内。本课不承诺任何极端浮点输入下都能保持无限精度，也不把 `-log(softmax(z)[y])` 作为一般数值实现方案。

### MLX cross-entropy without flattening

项目固定MLX 0.32.2。本节使用：

```python
position_losses = nn.losses.cross_entropy(
    raw_logits,
    targets,
    axis=-1,
    reduction='none',
)
```

先明确参数含义：

- `raw_logits` 是未经过softmax的浮点分数；不能把概率当logits再传入。
- `targets` 是整数正确类别IDs，每个值满足 `0<=y<V`。
- `axis=-1` 明确最后一轴是类别轴。
- `reduction='none'` 保留每个位置的loss，不在这里做整批求和或平均。

输入 `raw_logits:(B,T,V)`、`targets:(B,T)`，输出 `position_losses:(B,T)`。本节保留原rank，不做整批聚合；flatten B×T留给既定17.2。

### Prepared shape example: already-shifted targets

母版手工定义一个长度3的chunk：

```text
chunk = [[0,2,1]]
X     = [[0,2]]
Y     = [[2,1]]
```

因此B=1、T=2，X/Y shape都是 `(1,2)`。再手工定义V=3的toy logits：

```python
toy_logits = mx.log(mx.array(
    [[[1., 2., 3.], [3., 2., 1.]]],
    dtype=mx.float32,
))
```

该数组的shape是 `(1,2,3)`；它用于模拟MiniGPT输出接口，并非实际模型forward结果。取log的正数只是可手算例子的构造材料。

| 位置 | Softmax概率 | 正确类别Y | 逐位置loss |
|---|---|---:|---|
| `(0,0)` | `[1/6,1/3,1/2]` | 2 | `ln2` |
| `(0,1)` | `[1/2,1/3,1/6]` | 1 | `ln3` |

所以逐位置结果的数学预期是 `[[ln2,ln3]]`、shape `(1,2)`。这些是推导值，不是本轮运行输出；Y不再shift，也不在本节展平或平均。

### Operational order, scope, and ledger

之后实际使用work副本时：安全复制新文件 → 选择项目kernel → imports → 阅读单位置符号与softmax → 手工toy logits/target usage → NLL与cross-entropy对应 → 稳定表达式与MLX API → 自带的 `(1,2,3)` / `(1,2)` shape示意。所有必要数据与依赖已在本母版中。

本轮仅准备并发布17.1，停止在本节，不进入17.2。没有读取或下载语料，没有执行Notebook、测试、训练、实验、采样或用户work副本，没有代答练习。没有模型构造、forward、梯度、optimizer、训练循环或生成实现。材料准备和静态审阅不表示学员掌握或运行通过。

累计实际修正：16.1=**1/2**、16.2=**1/2**、16.3至16.5各**0/2**、17.1初版=**0/2**。每节最多两次实际修正，跨轮与换对话不重置。

### References

- [MLX 0.32.2 loss implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/losses.py)：项目固定版本的cross-entropy实现依据。
- [MLX cross_entropy API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html)：输入、类别轴与reduction。
- [MLX softmax API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.softmax.html)：显式指定词表轴。
- [MLX logsumexp API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logsumexp.html)：稳定表达式所用API。
- [Step15模型接口](step-15-transformer-to-gpt.md)、[Step16数据接口](step-16-tiny-shakespeare-data.md)、[course roadmap](course-roadmap.md)。
