# Step 17 — Language-Model Loss

本 Step 沿用现有 roadmap。当前只准备并发布 **17.2 Flatten B×T supervision**，初始生成实际修正计数 **0/2**，完成后停在17.2，不进入17.3。当前母版为 [73_step17_2_flatten_bt_supervision.ipynb](../notebooks/73_step17_2_flatten_bt_supervision.ipynb)。

材料审核方已独立核验大 Step 16 的全部母版、文档与完整回复，以及17.1的实际母版、文档、固定MLX版本接口与完整回复，核心静态 PASS。这是材料验收，不是学员学习或运行反馈；已通过课节不返工。累计实际修正保留16.1=**1/2**、16.2=**1/2**（包括已记录的超时后完整回复补齐）、16.3至16.5及17.1各**0/2**，17.2初版=**0/2**。计数跨轮、跨对话不重置，后续只按请求逐小节至大 Step 20 全部完成后停止。

## Step 17.1 — Logits, probabilities, and cross-entropy

母版：[72_step17_1_logits_probabilities_cross_entropy.ipynb](../notebooks/72_step17_1_logits_probabilities_cross_entropy.ipynb)。17.1已由材料审核方独立确认核心静态PASS，累计实际修正**0/2**；下文保留本节已通过的教学及当时边界。本节把 MiniGPT 的 raw logits 接到“模型为正确下一 token 分配了多少概率”这一问题，再由此建立每个位置的 loss。

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

17.1当轮仅准备并发布本节，停止在17.1，没有进入17.2。没有读取或下载语料，没有执行Notebook、测试、训练、实验、采样或用户work副本，没有代答练习。没有模型构造、forward、梯度、optimizer、训练循环或生成实现。材料准备和静态审阅不表示学员掌握或运行通过。

17.1验收后的累计实际修正：16.1=**1/2**、16.2=**1/2**、16.3至16.5及17.1各**0/2**。每节最多两次实际修正，跨轮与换对话不重置；当前17.2初版另记**0/2**。

### References

- [MLX 0.32.2 loss implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/losses.py)：项目固定版本的cross-entropy实现依据。
- [MLX cross_entropy API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html)：输入、类别轴与reduction。
- [MLX softmax API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.softmax.html)：显式指定词表轴。
- [MLX logsumexp API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logsumexp.html)：稳定表达式所用API。
- [Step15模型接口](step-15-transformer-to-gpt.md)、[Step16数据接口](step-16-tiny-shakespeare-data.md)、[course roadmap](course-roadmap.md)。

## Step 17.2 — Flatten B×T supervision

母版：[73_step17_2_flatten_bt_supervision.ipynb](../notebooks/73_step17_2_flatten_bt_supervision.ipynb)。本节承接17.1的逐位置交叉熵，把B行、每行T个位置组织成N=B×T条监督项，再说明这些位置损失的等权平均。这里准备的是未来同一训练步骤所用loss的组织，不执行或实现训练步骤。

### Self-contained work copy and the shared interface

新work副本自带 `mlx.core as mx`、`mlx.nn as nn` 导入和全部toy数组，沿用项目Python3.12、`.venv`与固定 `mlx==0.32.2`。之后使用时，选择项目kernel，先运行本节imports，再按本节顺序使用toy setup、reshape、逐位置loss与mean的cells即可。无需旧kernel变量、旧Notebook的 `%run`、语料缓存、sampler、模型构造或forward，没有新依赖；本轮只准备材料。

沿用已经建立的接口：`toy_logits:(B,T,V)` 是每个位置的raw scores，`Y:(B,T)` 是已右移的整数目标。`toy_logits[b,t,:]` 仍对应 `Y[b,t]`，不能在loss处再次shift。B表示batch行数，T表示每行输入位置数，V表示类别数。

本节手工给出两个长度为T+1的chunk，第一行沿用17.1的例子：

```python
import mlx.core as mx
import mlx.nn as nn

toy_chunks = mx.array([[0, 2, 1], [1, 0, 2]], dtype=mx.int32)
X = toy_chunks[:, :-1]  # [[0,2],[1,0]], shape (2,2)
Y = toy_chunks[:, 1:]   # [[2,1],[0,2]], shape (2,2), already shifted

toy_logits = mx.log(mx.array(
    [[[1., 2., 3.], [3., 2., 1.]],
     [[1., 1., 2.], [3., 2., 1.]]],
    dtype=mx.float32,
))  # Shape (2,2,3); manually specified, not model outputs.
```

对这些正数取log只为构造方便手算的分数，不是对token IDs或词频取log的模型算法。这里B=2、T=2、V=3，因此总共有 **N=B×T=4个token-level targets（逐token监督目标）**。X只帮助看清已右移的对应关系，不用来产生这些手工logits。

### Flatten merges the position axes and keeps V

**Flatten B×T supervision（将B、T两个定位轴展平为监督行）** 指把每个位置的V个类别分数作为一整行。我们合并的是B、T，不把类别轴V混进去：

| 对象 | 展平前 | 展平后 |
|---|---|---|
| Raw logits | `(B,T,V)` | `(N,V)`，N=B×T |
| Integer targets | `(B,T)` | `(N,)` |

定义展平后的行下标r。沿原数组的B/T顺序，先走完某行的T个位置，再进入下一行，因此：

\[
r=bT+t,\qquad
\mathrm{flat\_logits}[r,v]=\mathrm{toy\_logits}[b,t,v],\qquad
\mathrm{flat\_targets}[r]=Y[b,t].
\]

b从0到B−1，t从0到T−1，v从0到V−1；r便从0到N−1。这个式子只是为原位置换一个行号，两个数组必须采用同一对应关系。

| 展平行r | 原位置 `(b,t)` | 该行raw logits | 正确类别 |
|---:|---|---|---:|
| 0 | `(0,0)` | `[ln1,ln2,ln3]` | 2 |
| 1 | `(0,1)` | `[ln3,ln2,ln1]` | 1 |
| 2 | `(1,0)` | `[ln1,ln1,ln2]` | 0 |
| 3 | `(1,1)` | `[ln3,ln2,ln1]` | 2 |

展平不会创造新的监督信息，不会改变模型允许看见的因果上下文，也不是把两条输入序列接成一条更长的模型输入。不能只对logits或targets的一侧调换位置，否则正确目标会配到别的位置的分数上。

### Explain reshape before using it

**Reshape（重塑形状）** 在这里保留逻辑元素顺序和元素总数，只改变shape。logits的元素总数仍为B×T×V=N×V；targets的元素总数仍为B×T=N。`mx.reshape(array, shape)` 的第二个参数给出目标形状；不要据此推断底层存储一定不发生复制。

`(N,V)` 表示两个轴；`(N,)` 表示只有一个轴的Python tuple，尾部逗号使它成为单元素tuple，`(N)` 则只是给N加括号。

```python
B, T, V = toy_logits.shape
N = B * T

flat_logits = mx.reshape(toy_logits, (N, V))
flat_targets = mx.reshape(Y, (N,))
```

本例shape分别为 `(4,3)` 和 `(4,)`，`flat_targets` 的数学预期顺序为 `[2,1,0,2]`。以后读到 `mx.reshape(toy_logits, (-1,V))` 或 `mx.reshape(Y, (-1,))` 时，`-1` 表示从元素总数推断该维度；同一个shape里至多使用一处。当前示例显式写N，便于对照监督项的数量，不另开实验。

### Cross-entropy still computes one loss per target

复用17.1的API契约：传入raw logits和整数目标IDs，`axis=-1` 指类别轴，`reduction='none'` 保留每条监督项的loss。每个目标ID仍须满足 `0<=y<V`。

```python
flat_losses = nn.losses.cross_entropy(
    flat_logits,
    flat_targets,
    axis=-1,
    reduction='none',
)  # Shape (N,) = (4,).
```

**MLX本来就支持17.1的三维logits和二维targets。** 展平是理解与组织监督项的一种方式，不是该API只能接收二维logits的限制。同样使用 `reduction='none'` 时，直接对 `toy_logits,Y` 计算会得到 `position_losses:(B,T)`；两种写法在数学上满足 `flat_losses[b*T+t]=position_losses[b,t]`。

四个位置继续用17.1的NLL含义手算：

| r | 正确类别 | 正确类别的概率 | 逐位置loss |
|---:|---:|---|---|
| 0 | 2 | `3/(1+2+3)=1/2` | `ln2` |
| 1 | 1 | `2/(3+2+1)=1/3` | `ln3` |
| 2 | 0 | `1/(1+1+2)=1/4` | `ln4` |
| 3 | 2 | `1/(3+2+1)=1/6` | `ln6` |

所以原位置排列的损失为 `[[ln2,ln3],[ln4,ln6]]`，展平排列为 `[ln2,ln3,ln4,ln6]`。这是数学对应，不是本轮运行输出、测试结果或数值比较。

### Equal-weight mean: many targets, one scalar loss

**Arithmetic mean（算术平均）** 指把所有数相加，再除以数的个数。本例所有位置均有效且等权。用小写 `ell` / \(\ell_r\) 表示第r条监督项的损失，大写L表示它们的平均值：

单层求和用r遍历全部N条展平行；双层求和则先遍历某个b的T个位置，再遍历全部B行。两种下标写法数的是同一组B×T个损失，因此可以写成：

\[
L=\frac{1}{N}\sum_{r=0}^{N-1}\ell_r
 =\frac{1}{BT}\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\ell_{b,t}.
\]

分母是监督位置数N=B×T，不是类别数V。对本例，L的手算表达式为 `(ln2+ln3+ln4+ln6)/4`。

`mx.mean(array)` 未指定axis时计算所有元素的平均值；当前输入是一维 `flat_losses`，因此恰好汇总这N个位置：

```python
loss = mx.mean(flat_losses)  # Scalar, shape ().
```

**Scalar（标量）** 只有一个数，shape写作 `()`，没有B、T或V轴。本节不展开padding、mask或权重等其他设置，也不另加 `reduction='mean'` 捷径。

把三个操作分清：

| 操作 | Shape变化 | 所做的事 |
|---|---|---|
| Reshape | `(B,T,V)`→`(N,V)`；`(B,T)`→`(N,)` | 按相同顺序组织监督位置，保留每个V类向量 |
| Cross-entropy | `(N,V)`与`(N,)`→`(N,)` | 为每个位置的分数与正确类别算一个loss |
| Mean | `(N,)`→`()` | 对这N个位置损失取等权平均 |

一个batch含有N个token-level targets，不表示有N次参数更新；目标ID可以重复，相邻位置也不必是统计独立样本。这里只解释未来同一训练步骤所用loss的组织，不实现更新过程。

### Operational order, scope, and ledger

之后实际使用work副本时：安全复制到新文件名 → 选择项目kernel → imports与自带toy setup → 对照r与 `(b,t)` 的四行表 → reshape logits和targets → 未归约交叉熵 → 等权mean。所有必要依赖与示例数组均在本母版中，不需要执行旧课或读取语料。

本轮仅准备并发布17.2，完成即停，不进入17.3。不执行Notebook、用户work副本、测试、数值实验、采样或训练，不读取或下载语料，不代答练习；不实现 `value_and_grad`、梯度、optimizer、训练循环或生成。材料准备与静态审阅不代表学员掌握或运行验证。

累计实际修正保留：16.1=**1/2**、16.2=**1/2**（包括超时后完整回复补齐）、16.3至16.5及17.1各**0/2**，17.2初版=**0/2**。每小节最多两次实际修正，跨轮与换对话不重置；后续只按请求逐小节至大Step20全部完成后停止。

### References and continuity

- [17.1母版](../notebooks/72_step17_1_logits_probabilities_cross_entropy.ipynb)：raw logits、已shift的整数目标和逐位置交叉熵。
- [MLX 0.32.2 loss implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/losses.py) 与 [cross_entropy API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html)：原rank与展平后的输入契约。
- [Course roadmap](course-roadmap.md) 与 [teaching handoff](teaching-handoff.md)：当前17.2范围、累计计数与后续边界。
