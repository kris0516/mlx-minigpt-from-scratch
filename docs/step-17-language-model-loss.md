# Step 17 — Language-Model Loss

本 Step 沿用现有 roadmap，17.1至17.5的全部材料已获材料审核方独立核心静态 PASS。最新验收包括17.5的实际五文件提交 `3fdec00b251765bda581c2b183e68d039c83c902`、固定MLX v0.32.2接口和完整教学回复，实际修正 **0/2**。末节母版为 [76_step17_5_uncompiled_training_step.ipynb](../notebooks/76_step17_5_uncompiled_training_step.ipynb)。本文件保留已通过的教学与各节当时边界，不返工大Step17；当前只准备[18.1 Lazy evaluation](step-18-mlx-execution-model.md)，完成即停在18.1，不进入18.2。

这是材料验收，不是学员学习或运行反馈，不表示学员完成、掌握、运行通过或完成练习。大Step16已通过的材料与计数同样保留。累计实际修正为16.1=**1/2**、16.2=**1/2**（包括已记录的超时后完整回复补齐）、16.3至16.5及17.1至17.5各**0/2**，当前18.1初始生成=**0/2**。计数跨轮、跨对话不重置，后续只按请求逐小节至大Step20全部完成后停止。

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

17.1验收后的累计实际修正：16.1=**1/2**、16.2=**1/2**、16.3至16.5及17.1各**0/2**。每节最多两次实际修正，跨轮与换对话不重置；随后17.2至17.5验收计数均保持**0/2**。

### References

- [MLX 0.32.2 loss implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/losses.py)：项目固定版本的cross-entropy实现依据。
- [MLX cross_entropy API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html)：输入、类别轴与reduction。
- [MLX softmax API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.softmax.html)：显式指定词表轴。
- [MLX logsumexp API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logsumexp.html)：稳定表达式所用API。
- [Step15模型接口](step-15-transformer-to-gpt.md)、[Step16数据接口](step-16-tiny-shakespeare-data.md)、[course roadmap](course-roadmap.md)。

## Step 17.2 — Flatten B×T supervision

母版：[73_step17_2_flatten_bt_supervision.ipynb](../notebooks/73_step17_2_flatten_bt_supervision.ipynb)。17.2实际母版、文档与完整回复已由材料审核方独立确认核心静态PASS，累计实际修正**0/2**；下文保留本节已通过的教学及当时边界。本节承接17.1的逐位置交叉熵，把B行、每行T个位置组织成N=B×T条监督项，再说明这些位置损失的等权平均。这里准备的是未来同一训练步骤所用loss的组织，不执行或实现训练步骤。

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

17.2当轮仅准备并发布本节，完成即停，没有进入17.3。没有执行Notebook、用户work副本、测试、数值实验、采样或训练，没有读取或下载语料或代答练习；没有实现 `value_and_grad`、梯度、optimizer、训练循环或生成。材料准备与静态审阅不代表学员掌握或运行验证。

17.2验收后的累计实际修正：16.1=**1/2**、16.2=**1/2**（包括超时后完整回复补齐）、16.3至16.5及17.1至17.2各**0/2**。每小节最多两次实际修正，跨轮与换对话不重置；随后17.3至17.5验收均保持**0/2**，后续只按请求逐小节至大Step20全部完成后停止。

### References and continuity

- [17.1母版](../notebooks/72_step17_1_logits_probabilities_cross_entropy.ipynb)：raw logits、已shift的整数目标和逐位置交叉熵。
- [MLX 0.32.2 loss implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/losses.py) 与 [cross_entropy API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html)：原rank与展平后的输入契约。
- [Course roadmap](course-roadmap.md) 与 [teaching handoff](teaching-handoff.md)：本节范围记录、累计计数与当前后续边界。

## Step 17.3 — MLX nn.value_and_grad

母版：[74_step17_3_nn_value_and_grad.ipynb](../notebooks/74_step17_3_nn_value_and_grad.ipynb)。17.3真实母版、文档、MLX v0.32.2接口与完整教学回复已由材料审核方独立确认核心静态PASS，累计实际修正**0/2**；下文保留本节已通过的教学及当时边界。本节把17.2的标量loss接口接到Module的可训练参数树，解释如何由一个loss函数得到“同时返回loss与gradients”的新函数。只到求导接口与梯度对应关系，不进入17.4 Adam或参数学习更新。

### Self-contained work copy and the new dependency on parameters

新work副本自带 `mlx.core as mx`、`mlx.nn as nn`、完整toy Module、toy数据、loss函数与变换调用示例。之后使用时，选择项目Python3.12 / `.venv` kernel，从本节imports开始按顺序使用；沿用 `mlx==0.32.2`，不添加依赖，不需要旧kernel、旧Notebook的 `%run`、语料缓存、sampler或完整MiniGPT定义。本轮只准备材料，所有代码均未执行、无输出。

17.2的手工toy logits用来解释监督组织，并不依赖模型参数。现在要求导的是“loss怎样随参数变化”，所以本节必须在loss函数内部，由模型参数产生logits，再计算同一个reshape → 交叉熵 → mean。不能把上节那组固定logits放到函数外，然后宣称所得loss反映了模型参数的变化。

### A tiny Module and its parameter tree

回顾4.2：`class ... (nn.Module)` 定义模型类，`super().__init__()` 初始化父类；`self` 表示当前对象。实例写成 `model(X)` 时会调用它的 `__call__` 方法。

本例给模型一个公开属性 `head`，它是Python **dict（字典）**。字典通过key查找值，`self.head['bias']` 用字符串key `'bias'` 取得数组。dict是参数树的 **container（容器）**，真正保存浮点参数的数组是 **leaf（叶子）**。

`mx.broadcast_to(array, shape)` 按广播规则把数组扩展为指定shape。本例把一个长度V=3的向量广播到 `(B,T,V)`：每个位置使用同一组3个参数，不是为B×T个位置创建独立参数。

```python
import mlx.core as mx
import mlx.nn as nn

class SharedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = {
            'bias': mx.array([0., 0., 0.], dtype=mx.float32)
        }

    def __call__(self, X):
        B, T = X.shape
        V = self.head['bias'].shape[0]
        return mx.broadcast_to(self.head['bias'], (B, T, V))
```

这只是**参数到loss的接口toy替身**。它只读取X的shape，不读取token内容或上下文，每个位置都输出相同的分数向量；它不是MiniGPT，也不是给MiniGPT推荐全零初始化。

MLX会递归收集Module、dict、list结构中的公开数组参数。`parameters()` 返回参数树，其中也可以包含被冻结的参数；`trainable_parameters()` 只保留可训练的参数。本例没有冻结任何叶子，两棵树相同，其结构示意为：

```text
{'head': {'bias': 一个shape为(3,)的MLX数组}}
```

```python
model = SharedLogitModel()
params = model.parameters()
trainable_params = model.trainable_parameters()
```

路径 `head → bias` 标识同一个参数叶子，例如 `trainable_params['head']['bias']`。这里的树形组织保留名称和层次，并不是17.2合并B、T轴的张量reshape。输入数据X、目标Y放在Module外部，不把它们作为模型参数保存。

### Put the scalar loss in a function

**Function object（函数对象）** 是可被传递的函数本身：`loss_fn` 表示函数；`loss_fn(model,X,Y)` 则表示调用它。Python的 `def` 定义函数，函数体在调用时使用传入的参数，`return` 把计算结果交还调用方。

先恢复17.2相同的toy输入：

```python
toy_chunks = mx.array([[0, 2, 1], [1, 0, 2]], dtype=mx.int32)
X = toy_chunks[:, :-1]  # [[0,2],[1,0]], shape (2,2)
Y = toy_chunks[:, 1:]   # [[2,1],[0,2]], shape (2,2), already shifted
```

下面 `loss_fn` 的第一个普通Python参数叫model，函数内部的 `model(X)` 调用它的forward规则。所得raw logits依赖当前模型参数；再沿用17.2的两侧reshape、逐位置CE和等权mean。返回值保持MLX scalar array，shape为 `()`，不转成 `.item()` 或 `float(...)`。

```python
def loss_fn(model, X, Y):
    logits = model(X)
    B, T, V = logits.shape
    N = B * T
    flat_logits = mx.reshape(logits, (N, V))
    flat_targets = mx.reshape(Y, (N,))
    token_losses = nn.losses.cross_entropy(
        flat_logits,
        flat_targets,
        axis=-1,
        reduction='none',
    )
    return mx.mean(token_losses)
```

X/Y shape为 `(2,2)`，logits为 `(2,2,3)`，N=4。`token_losses:(4,)` 每项仍对应target顺序 `[2,1,0,2]`，但数值由本节共享参数产生，不再是17.2手工指定的四组分数。Y已经shift，不再次右移；交叉熵直接接收raw logits。

### Function transformation: first obtain a callable, then call it

**Function transformation（函数变换）** 接收一个函数并产生另一个函数。5.3已经见过 `mx.grad` 和 `mx.value_and_grad`；这里使用能识别Module可训练参数树的 `nn.value_and_grad`。它的第一个参数model指定“对这个Module的可训练参数求导”，第二个参数接收函数对象 `loss_fn`，返回一个新的 **callable（可调用对象）**，下面命名为g：

```python
g = nn.value_and_grad(model, loss_fn)
```

这里没有把一个预先计算好的loss交给变换。随后调用g时，`model,X,Y` 是转交给原 `loss_fn(model,X,Y)` 的普通实参；包装器不会自动插入model。因此两个地方都出现model是有原因的，而且使用的是同一个实例。

调用g返回两项结果。左侧的 `loss_value, grads` 是 **tuple unpacking（元组解包）**：按顺序接收两个返回项。第一项 `loss_value` 是scalar MLX array；第二项 `grads` 是与可训练参数树对应的梯度结构：

```python
loss_value, grads = g(model, X, Y)
```

`nn.value_and_grad` 不是给通用 `mx.value_and_grad` 换一个缩写，也不在此helper上另加 `argnums`。本节示例按参数树路径读取梯度：

```python
bias_gradient = grads['head']['bias']  # Shape (3,).
```

这个向量的三个分量对应共享参数的三个分量。它不是shape为 `(4,)` 的逐token损失，也不是shape为 `(2,2,3)` 的logits梯度。参数树的叶子shape与相应梯度叶子的shape一致。

当前用显式返回的 `grads` 接收结果，无需假定参数数组上自动附着 `.grad`，也无需添加 `.backward()` 或 `.zero_grad()` 调用。这个例子只准备loss与梯度的取得方式，不进行参数学习更新。

### Hand calculation: one shared vector, four supervised positions

用 \(\theta\) 表示所有可训练参数，本例就是长度3的向量。标量loss写作 \(L(\theta;X,Y)\)：当前X/Y固定，考察L对参数的依赖。

**Partial derivative（偏导数）** \(\partial L/\partial\theta_v\) 表示固定其他参数分量时，第v个分量微小变化引起的局部loss变化率。把三个偏导数按参数位置排列，就是shape `(3,)` 的 **gradient（梯度）**。正负号解释的是局部影响，不是本节要实现的更新算法。

本例 \(\theta=[0,0,0]\)。每个位置的三个logits都相同，因此各类别概率均为1/3，每个位置的损失均为ln3，等权平均仍为：

\[
L=\ln3.
\]

接着推导三个梯度分量。r是17.2的展平位置，\(y_r\) 是它的正确类别；定义one-hot指示量 \(q_{r,v}\)：当v等于\(y_r\)时为1，否则为0。共享的类别概率记作 \(p_v\)。

沿用17.1从logits直接计算NLL的表达式，令 \(s=\sum_u\exp(\theta_u)\)，其中u遍历三个类别，则：

\[
\ell_r=\ln s-\theta_{y_r}.
\]

简短回顾链式法则所需的导数：\(\mathrm{d}\exp(a)/\mathrm{d}a=\exp(a)\)，对s>0有 \(\mathrm{d}\ln s/\mathrm{d}s=1/s\)。固定其他分量时，s中只有 \(\exp(\theta_v)\) 这一项随 \(\theta_v\) 变化；\(\theta_{y_r}\) 对 \(\theta_v\) 的导数就是指示量 \(q_{r,v}\)。因此：

\[
\frac{\partial\ell_r}{\partial\theta_v}
=\frac{\exp(\theta_v)}{s}-q_{r,v}
=p_v-q_{r,v}.
\]

四个位置共用同一个参数向量，平均loss的梯度也要把它们的贡献相加再除以N=4。target顺序 `[2,1,0,2]` 中，类别0、1、2分别出现1、1、2次；one-hot指示量的平均分别为1/4、1/4、1/2。所以：

\[
\nabla_\theta L
=\left[\frac13-\frac14,\frac13-\frac14,\frac13-\frac12\right]
=\left[\frac1{12},\frac1{12},-\frac16\right].
\]

这对应 `grads['head']['bias']` 的三个分量。前两个正分量表示分别小幅增大相应参数时loss局部上升；第三个负分量表示小幅增大第三个参数时loss局部下降。这里只解释敏感度，不选择学习率或执行参数更新。

`loss_value=ln3` 和上述梯度都是按明确输入推导的数学预期，不是本轮运行输出、数值实验或测试结果；没有展开softmax Jacobian。

### Operational order, scope, and ledger

之后使用work副本时：安全复制到新文件名 → 选择项目kernel → imports → 阅读dict与broadcast并定义toy Module → 创建toy模型和外部X/Y → 了解参数树 → 定义内部调用model的loss函数 → 先取得g再调用g → 按树路径读取梯度并对照手算。所需定义全部在新母版中，不依赖旧课执行状态。

17.3当轮仅准备并发布本节，完成即停，没有进入17.4。没有读取或下载语料，没有执行课程代码、Notebook、测试、数值实验、采样、训练或用户work副本，没有代答练习；不实现optimizer、参数学习更新、训练循环或生成，不扩展执行/编译专题。材料准备、静态审阅与发布均不表示学员掌握或运行验证。

17.3验收后的累计实际修正：16.1=**1/2**、16.2=**1/2**（包括超时后的完整回复补齐）、16.3至16.5及17.1至17.3各**0/2**。每小节最多两次实际修正，跨轮与换对话不重置；随后17.4与17.5验收均保持**0/2**，后续仍只按请求逐小节至大Step20全部完成后停止。

### References and continuity

- [MLX v0.32.2 nn/utils.py](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/utils.py)：Module-aware `value_and_grad` 的固定版本调用契约。
- [MLX v0.32.2 Module implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/base.py)：参数树与可训练参数的筛选。
- [Official nn.value_and_grad API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html)：函数变换及返回值。
- [4.2 Module与参数树](../notebooks/09_step04_2_parameters_bias_module.ipynb)、[5.3 自动微分](../notebooks/14_step05_3_gradient_autodiff.ipynb)、[17.2监督与标量loss](../notebooks/73_step17_2_flatten_bt_supervision.ipynb)：本节的课程前置。
- [Course roadmap](course-roadmap.md) 与 [teaching handoff](teaching-handoff.md)：17.3范围、材料状态与累计计数。

## Step 17.4 — Adam optimizer

母版：[75_step17_4_adam_optimizer.ipynb](../notebooks/75_step17_4_adam_optimizer.ipynb)。17.4实际母版、文档、固定MLX接口与完整教学回复已由材料审核方独立确认核心静态PASS，累计实际修正**0/2**；下文保留本节已通过的教学及当时边界。本节承接17.3已经得到的scalar loss与参数梯度树，解释Adam怎样保存历史梯度的摘要、怎样形成每个参数分量的调整，以及这些状态为什么占内存。只准备Adam本身与一次孤立的树接口示例，不组装17.5的训练步骤。

### Self-contained inputs: use the parameter-shaped gradient

17.3中，两个batch行、每行两个位置共同产生一个scalar mean loss。共享参数 `head.bias` 有3个分量；在零参数处，手算得到：

\[
g_1=[1/12,\ 1/12,\ -1/6],\qquad \text{shape }(3,).
\]

这个梯度已经包含四个位置的贡献及mean中的除4，优化器处不再除以B×T。它不是四项位置loss，也不是 `(B,T,V)` 的logits梯度。

新work副本直接自带这个参数树与梯度树示意，使用项目Python3.12 / `.venv` kernel和固定 `mlx==0.32.2`。`mlx.optimizers` 是MLX自带的优化器模块，`as optim` 只是导入后的短名称；不安装新依赖，不恢复旧Notebook变量，不调用模型、loss或自动微分。

```python
import mlx.core as mx
import mlx.optimizers as optim

parameters = {
    "head": {
        "bias": mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
    }
}

example_grads = {
    "head": {
        "bias": mx.array([1 / 12, 1 / 12, -1 / 6], dtype=mx.float32),
    }
}
```

这两个Python字典保持同一路径 `head → bias`，叶子shape都为 `(3,)`。`example_grads` 是前节手算结果的接口材料，不是本轮实际求导所得。以后使用本节work副本时，所有必要导入、数组、优化器构造和调用示例都在本节，无需语料、sampler或MiniGPT定义；本轮所有代码均未执行。

### First and second moments: keep compact history

用k表示第几次优化器更新，避免与序列长度T或位置t混淆。\(\theta_{k-1}\) 是本次更新前的参数，\(g_k\) 是当前scalar loss对这些参数的梯度；\(\alpha\)（alpha）是 **learning rate（学习率）**，控制参数调整的整体尺度。

对于一个正梯度分量，减去一个小的正调整会沿局部下降方向移动；负梯度则相反。梯度只说明当前位置的局部敏感度，不能由此保证任意有限更新都会降低loss。Adam进一步使用历史梯度的方向和尺度信息。

**Exponential moving average，EMA（指数移动平均）** 每次保留旧估计的一部分，再加入新观察的一部分。\(\beta\)（beta）位于0与1之间，越接近1，旧估计保留得越多。Adam维护两组与参数同shape的数组，起点均为0：

\[
m_0=0,\qquad v_0=0,
\]

\[
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,
\]

\[
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2.
\]

**First moment estimate（一阶矩估计）** m对带符号的梯度做平滑，保留近期梯度的方向信息。**Second raw moment estimate（二阶原始矩估计）** v对梯度的平方做平滑，用来描述分量尺度；它没有减去均值，因此不是中心化方差，也不是对loss求二阶导数。

这里的乘法、平方、开平方和除法都是 **element-wise（逐元素）** 运算。例如 `[a,b,c]` 的平方是 `[a²,b²,c²]`，不是向量内积。每个参数分量有自己对应的m和v，叶子shape始终为 `(3,)`。优化器只保存这两个摘要，不是把每一次旧梯度全部存下来。

### Bias correction and the fixed-version implementation choice

从0开始会使早期EMA仍带有零初值的影响。例如第一次更新只有 \((1-\beta_1)g_1\)，而不是完整的 \(g_1\)。若输入梯度始终为同一个g，递推k次可得到 \(m_k=(1-\beta_1^k)g\)；平方梯度的递推也有同样形式。

**Bias correction（偏差修正）** 因此定义：

\[
\hat m_k=\frac{m_k}{1-\beta_1^k},\qquad
\hat v_k=\frac{v_k}{1-\beta_2^k}.
\]

帽子符号表示修正后的量；上标k表示乘方，不是数组下标。这种修正处理零初始化的权重偏差，不会让不断变化的真实梯度变成已知常量，也与模型参数名 `head.bias` 中的bias不是同一含义。

本节选择带修正的公式：

\[
\theta_k=\theta_{k-1}
-\alpha\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
\]

\(\epsilon\)（epsilon）是很小的正数，在本例float32计算中用于避免分母为零；它位于**平方根外面**。分母的平方根把梯度平方的尺度还原，再对各分量进行缩放。它不是对所有数值、dtype或输入都有效的稳定性保证，也不意味着每个分量以后总是调整同样的幅度。

课程固定的MLX v0.32.2需要按真实实现区分“默认值”和“本节选项”：

| Adam构造参数 | MLX v0.32.2默认行为 | 本节明确选择 |
|---|---|---|
| `learning_rate` | 必须提供，没有默认数值 | `1e-3`，即0.001 |
| `betas` | `[0.9, 0.999]` | `(0.9, 0.999)` |
| `eps` | `1e-8` | `1e-8` |
| `bias_correction` | **`False`** | **`True`** |

因此省略 `bias_correction` 并不会使用上面的带帽公式。该版本默认分支直接使用未经修正的m、v：\(\theta_k=\theta_{k-1}-\alpha m_k/(\sqrt{v_k}+\epsilon)\)。两条分支是不同的实现设置，不把其他框架的默认值套到MLX。

下面用 **keyword arguments（关键字实参）** 把所选数值写清楚；`betas` 的两个数构成一个tuple，按顺序表示beta1和beta2。`True` 是Python布尔值，用来开启该选项。`optim.Adam(...)` 构造一个保存配置和状态的优化器对象，并赋给变量optimizer。这组数值是教学选择，不是已调优的MiniGPT配置。

```python
optimizer = optim.Adam(
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    bias_correction=True,
)
```

[固定v0.32.2 Adam与基类实现](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/optimizers/optimizers.py)是本节默认值、公式分支与状态行为的依据；在线文档可能随新版本更新。

### One hand calculation and one isolated tree API

从 \(\theta_0=[0,0,0]\)、\(m_0=v_0=0\) 和上面的g1开始，取beta1=0.9、beta2=0.999：

\[
m_1=0.1g_1=[1/120,\ 1/120,\ -1/60],
\]

\[
v_1=0.001g_1^2=[1/144000,\ 1/144000,\ 1/36000].
\]

第一轮的修正分母分别为0.1、0.001，所以：

\[
\hat m_1=g_1,\qquad \hat v_1=g_1^2.
\]

各分量都非零，忽略展示近似中很小的epsilon时，\(g_1/\sqrt{g_1^2}\) 只留下正负号，得到：

\[
\theta_1\approx[-0.001,\ -0.001,\ +0.001].
\]

这里写“约等于”：有epsilon时调整幅度会略小，不能声称输出恰好等于该数组。若保留其余设置、改为默认 `bias_correction=False`，第一轮比例约为 \(0.1/\sqrt{0.001}=\sqrt{10}\)，调整幅度约0.0031623。这个对照只说明默认分支和带修正分支确实不同，不再增加第二次调用或数值实验。

接下来才使用API。`optimizer.init(parameters)` 是对象的 **method（方法）**，按给定参数树建立状态。在这个全新optimizer上，m/v从0开始，step仍为0，没有参数更新。

```python
optimizer.init(parameters)
```

随后准备一次孤立调用：

```python
parameters_after = optimizer.apply_gradients(example_grads, parameters)
```

第一个实参是梯度树，第二个是当前参数树。方法返回下一组参数树，左侧赋值用新变量 `parameters_after` 接收；原 `parameters` 不会被这次调用替换。与此同时，优化器对象自己的state会前进，因此它不是完全无副作用的函数。这里没有forward、loss、自动微分、model更新或完整训练循环，也没有执行这个调用。

优化器状态沿参数路径继续向下存放m、v：

| 读取路径 | 本例作用与shape |
|---|---|
| `optimizer.state["step"]` | scalar `()`，优化器次数；全新初始化时0，应用一次后为1 |
| `optimizer.state["learning_rate"]` | scalar `()`，当前学习率 |
| `optimizer.state["head"]["bias"]["m"]` | 未修正的一阶矩数组，`(3,)` |
| `optimizer.state["head"]["bias"]["v"]` | 未修正的二阶原始矩数组，`(3,)` |
| `parameters_after["head"]["bias"]` | 返回的新参数叶子，`(3,)` |

实际持久保存的是m和v，不是另存一套hat m/hat v。`betas`、`eps`、`bias_correction` 是优化器对象配置，不能从这个state字典中假定读到这些键。

该版本通过 `zeros_like(parameter)` 初始化moment数组，即按对应参数的shape和dtype建立零数组。本例参数明确为float32，所以m/v也是float32；不能由此推论所有参数dtype下，MLX都会额外自动维护float32状态。

新的optimizer对象意味着新的历史状态。以后若要延续同一优化过程，不能只保留参数值而丢掉这些历史；本节只解释这个原因，保存与恢复实现仍留在Step20。上述状态值与参数值均是源码契约和手算预期，不是实际运行输出。

### Why the state consumes memory

设P是被优化的参数**标量个数**，d是每个标量占用的字节数。P既不是参数数组叶子的个数，也不是B×T监督位置数。假设参数、梯度与moments使用相同dtype，则：

\[
\text{Adam的m/v数组数据}\approx2Pd\ \text{bytes}.
\]

同样的P个参数位置，要各自保存一个m和一个v，因此仅这两份历史摘要就需要两份参数大小的数据。状态不会因为已经处理更多token就变成每个token一套，也不会随更新次数逐次保存全部旧梯度。

如果另外把当前参数和一份梯度也计入，四份数组的合计是：

\[
\text{参数+梯度+m+v}\approx4Pd\ \text{bytes}.
\]

这不是“Adam状态本身有四份”。本例P=3、float32每项4字节：

| 数据部分 | 数量 | 数组数据字节 |
|---|---|---|
| 当前参数 | 3个float32 | 12 |
| 一份梯度 | 3个float32 | 12 |
| 一阶矩m | 3个float32 | 12 |
| 二阶矩v | 3个float32 | 12 |
| 仅m+v | 6个float32 | **24** |
| 上述四份数组合计 | 12个float32 | **48** |

这个简单账本没有包含step/learning_rate等标量状态、Python容器、模型activations、临时数组、同时保留的新旧参数版本、分配器与缓存。尤其本节用 `parameters_after` 保留新值而仍保留原parameters，不能把48字节当作这个Notebook的总占用或峰值测量。这里说明的是额外状态随参数规模增长的原因，不做内存实验。

### Operational order, scope, and ledger

以后使用work副本时：安全复制75母版到已有 `notebooks/work/` 的时间戳新文件名 → 选择项目kernel → imports → 明确手工参数/梯度树的来源 → 阅读EMA、偏差修正与固定版本默认值 → 构造全新Adam并初始化state → 阅读手算 → 使用单次树接口示例并按路径理解状态。所有必要定义在新副本中；重复调用 `apply_gradients` 会继续增加step和历史，不是重新做第一轮手算。

17.4当轮只准备并发布本节，完成即停，没有进入17.5。没有运行Notebook、课程代码、测试、训练、实验、采样或用户work副本，没有读取或下载语料，没有代答练习。没有组装完整训练步骤，不扩展AdamW、学习率计划、执行/编译或checkpoint实现。材料准备、静态审阅与发布均不表示学员掌握或运行通过。

17.4验收后的累计实际修正：16.1=**1/2**、16.2=**1/2**（包括已记录的超时后完整回复补齐）、16.3至16.5及17.1至17.4各**0/2**；随后17.5验收保持**0/2**。每小节最多两次实际修正，跨轮与换对话不重置；后续按请求逐小节至大Step20全部完成后停止。

### References and continuity

- [17.3 nn.value_and_grad母版](../notebooks/74_step17_3_nn_value_and_grad.ipynb)：scalar mean loss、参数树及对应的返回梯度。
- [MLX v0.32.2 optimizers.py](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/optimizers/optimizers.py)：Adam真实默认值、两个公式分支、moment初始化和Optimizer的state/init/apply_gradients。
- [Official Adam API](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adam.html)：公开构造接口；固定版本行为仍以v0.32.2源码为准。
- [Course roadmap](course-roadmap.md) 与 [teaching handoff](teaching-handoff.md)：17.4范围、材料状态、累计修正计数及停止边界。


## Step 17.5 — Uncompiled training step

母版：[76_step17_5_uncompiled_training_step.ipynb](../notebooks/76_step17_5_uncompiled_training_step.ipynb)。17.5实际五文件提交、固定MLX v0.32.2接口和完整教学回复已由材料审核方独立确认核心静态PASS，累计实际修正**0/2**；下文保留本节已通过的教学与当时边界。本节把已经说明的forward、scalar loss、参数梯度和Adam接成一次 **uncompiled training step（未编译训练步骤）**，并明确何时要求这些数组结果完成求值。这里只准备一个自包含toy单步，没有训练循环或运行结果；完成后停在大Step17，不进入大Step18。

### Restore all dependencies in the new work copy

本节沿用Python3.12 / `.venv` kernel与固定 `mlx==0.32.2`。新副本包含所有必要导入、类、toy数据、loss函数、函数变换、Adam构造、单步函数和一次准备调用；无需旧kernel、旧Notebook执行、`%run`、语料缓存、sampler或完整MiniGPT定义。

先导入三个模块。`mx` 负责数组运算，`nn` 提供Module、交叉熵和针对Module的求导变换，`optim` 提供Adam：

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
```

恢复17.3的最小模型。`class ... (nn.Module)` 定义继承Module的类，`super().__init__()` 初始化父类，`self` 指当前实例。调用 `model(X)` 会进入 `__call__`。公开字典 `self.head` 中的 `bias` 是参数叶子，shape为 `(3,)`；`mx.broadcast_to` 把同一个向量用于每个batch/位置。

```python
class SharedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = {
            "bias": mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
        }

    def __call__(self, X):
        B, T = X.shape
        V = self.head["bias"].shape[0]
        return mx.broadcast_to(self.head["bias"], (B, T, V))


toy_chunks = mx.array([[0, 2, 1], [1, 0, 2]], dtype=mx.int32)
X = toy_chunks[:, :-1]
Y = toy_chunks[:, 1:]
```

这个toy只读取X的shape，不根据token内容或上下文预测；它保留接口以便手算，不是MiniGPT架构，也不是给MiniGPT推荐零初始化。X/Y都在Module之外。X=`[[0,2],[1,0]]`，Y=`[[2,1],[0,2]]`，B=2、T=2、V=3；Y已经shift，不再次右移。

### Forward and scalar loss: preserve the parameter-to-loss dependency

**Forward pass（前向计算）** 按模型规则由当前参数与输入得到raw logits。本例是在loss函数内部调用 `model(X)`，让后续交叉熵依赖当前模型参数。

**Scalar loss（标量损失）** 用17.2的方式得到：保持类别轴V，合并B/T为N=B×T个位置，逐位置交叉熵后再求等权mean。`return` 返回MLX scalar array，shape `()`，不在loss函数中转成Python数值。

```python
def loss_fn(model, X, Y):
    logits = model(X)
    B, T, V = logits.shape
    N = B * T
    flat_logits = mx.reshape(logits, (N, V))
    flat_targets = mx.reshape(Y, (N,))
    token_losses = nn.losses.cross_entropy(
        flat_logits,
        flat_targets,
        axis=-1,
        reduction="none",
    )
    return mx.mean(token_losses)
```

| 计算对象 | 本例shape | 含义 |
|---|---|---|
| 参数 `head.bias` | `(3,)` | 三个共享分数参数 |
| X与Y | `(2,2)` / `(2,2)` | 输入与已shift目标 |
| raw logits | `(2,2,3)` | 每个位置的三类别分数 |
| 展平logits / targets | `(4,3)` / `(4,)` | 四条对应的监督项 |
| token losses | `(4,)` | 四个位置的交叉熵 |
| scalar loss | `()` | 四项loss的等权平均 |

保留raw logits与整数targets，不先softmax再作为logits传入。四项平均的缩放已经进入loss和梯度，优化器处不再除以4。

### Reverse-mode autodiff: follow dependencies backward

**Reverse-mode automatic differentiation（反向模式自动微分）** 从scalar loss出发，沿前向计算依赖反向应用链式法则，得到loss对参数的导数。顺着一条计算路径时，局部导数相乘；同一参数通过多条路径影响loss时，各路径贡献相加。这里的“反向”不是倒着读tokens、求前向函数的逆函数，也不是逐个扰动参数做有限差分。

用r表示展平后的监督位置，v表示类别；本例r=0到3、v=0到2。令 `y_r` 是第r个target ID，`z_{r,v}` 是该位置第v类raw logit，`p_{r,v}` 是softmax概率。定义one-hot指示量 `q_{r,v}`：当v等于y_r时为1，否则为0。q只是推导记号，代码仍传整数targets。

反向链条从loss对自身的导数开始：

\[
\frac{\partial L}{\partial L}=1.
\]

因为 \(L=(1/N)\sum_r\ell_r\)，每个位置收到mean传来的权重：

\[
\frac{\partial L}{\partial\ell_r}=\frac1N.
\]

沿用17.3的交叉熵局部导数 \(\partial\ell_r/\partial z_{r,v}=p_{r,v}-q_{r,v}\)，链式法则给出：

\[
\frac{\partial L}{\partial z_{r,v}}
=\frac{p_{r,v}-q_{r,v}}N.
\]

本例每个位置都使用同一个参数分量：\(z_{r,v}=\theta_v\)。所以broadcast的各位置贡献汇总到同一个参数上：

\[
\frac{\partial L}{\partial\theta_v}
=\sum_{r=0}^{N-1}\frac{p_{r,v}-q_{r,v}}N.
\]

在零参数处，p的三个分量均为1/3；四个targets按顺序是 `[2,1,0,2]`，各类别比例为1/4、1/4、1/2。因此参数梯度仍是：

\[
\nabla_\theta L=[1/12,\ 1/12,\ -1/6],\qquad \text{shape }(3,).
\]

这说明逐logit导数可以有 `(B,T,V)` 的对应关系，但最终返回的是参数形状的梯度树。本节不手填这一梯度；准备代码将由 `nn.value_and_grad` 连接反向规则得到它。以上推导是理解计算关系，不是执行结果或数值检查。

### Create one consistent model, gradient function, and Adam instance

Python中 `loss_fn` 表示函数对象，`loss_fn(...)` 表示调用。`nn.value_and_grad(model, loss_fn)` 接收函数并返回一个新函数；构造时的model指定求导参数所属的Module。之后调用新函数时，`model,X,Y` 会转交给原来的 `loss_fn(model,X,Y)`，包装器不会自动插入model。

把这些对象在单步函数之外创建一次，并保持同一组实例：

```python
model = SharedLogitModel()
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(
    learning_rate=1e-3,
    betas=[0.9, 0.999],
    eps=1e-8,
    bias_correction=True,
)
optimizer.init(model.trainable_parameters())
```

`trainable_parameters()` 返回模型的可训练参数树。对这个全新optimizer，`init` 按该树准备零初始m/v，不增加step。配置延续17.4：本节明确使用 `bias_correction=True`；MLX v0.32.2的真实默认是False，learning_rate也没有默认数值。这些是教学设置，不是已调优配置。

不能只重建model却沿用绑定旧model的 `loss_and_grad_fn`。重新从首步开始时，要一起重建model、对应函数变换和optimizer。正常延续同一过程则要保留optimizer历史，不在每个单步内构造一个全新Adam。

### Optimizer update and the minimum evaluation contract

17.4的 `apply_gradients(grads, parameters)` 返回新参数树。本节使用针对Module的 **`optimizer.update(model, grads)`**：它应用优化器规则并把新参数写回model，同时推进优化器state。它的返回值是None，因此不写 `model = optimizer.update(...)`。

**Evaluation（求值）** 指真正得到数组数值结果。理解本单步只需要一个MLX事实：Python已经走过一行数组运算，不代表其所需数值计算都已经完成。**Explicit evaluation boundary（显式求值边界）** 用 `mx.eval(...)` 清楚指定本次调用结束前必须完成哪些结果及其依赖。

固定v0.32.2的 `mx.eval` 可接收多个数组，也可接收包含数组的dict/list/tuple树；完成指定数组所需的求值后返回None。这里选择三个根结果：

```python
mx.eval(loss_before_update, model.parameters(), optimizer.state)
```

其中 `model.parameters()` 在更新后读取当前模型参数树，`optimizer.state` 是更新后的m/v、step等状态；loss是本次用于求导的那一个scalar。只要求loss完成，不应当被描述为已经明确要求新的参数与优化器状态都完成。本节只解释这条必要边界，不展开Step18的图增长、异步、编译或内存测量。

### One ordinary Python training-step function

`def` 定义函数时不会执行函数体。下面显式传入model、X、Y、梯度函数和optimizer，让所需对象清楚可见。`loss_before_update, grads = ...` 是元组解包，按顺序接收scalar loss和参数梯度树。

**Uncompiled（未编译）** 在这里表示普通Python函数没有用 `mx.compile` 包装，不表示强制CPU执行，也不表示内部每一行都立即完成全部数值计算。

```python
def train_step(model, X, Y, loss_and_grad_fn, optimizer):
    loss_before_update, grads = loss_and_grad_fn(model, X, Y)
    optimizer.update(model, grads)
    mx.eval(loss_before_update, model.parameters(), optimizer.state)
    return loss_before_update
```

`loss_and_grad_fn` 调用包含loss函数内部的forward、交叉熵、mean以及对模型参数的反向模式求导；不用再在前面额外计算一次forward/loss。优化器取得返回的grads后更新参数；显式求值边界再要求本步关心的结果完成。

这里只返回loss；model与optimizer通过方法调用发生状态变化。返回值不必包含这些对象，调用方仍持有同一组实例。未添加手写 `.backward()`、`.grad` 或 `.zero_grad()` 工作流。

### Why the returned loss belongs to the parameters before the update

令\(\theta_{\text{before}}\)表示本步开始时的参数，\(\theta_{\text{after}}\)表示本步更新后的参数。函数返回的是：

\[
L_{\text{returned}}=L(\theta_{\text{before}};X,Y).
\]

虽然 `mx.eval` 写在update之后，已经建立的loss计算仍对应本次forward使用的旧参数；更新model中的参数引用，不会把这个loss自动改写为 \(L(\theta_{\text{after}};X,Y)\)。因此不能把返回loss标成“更新后的loss”或据此宣称本次更新降低了loss。本节不额外执行第二次forward来测量更新后的loss。

在首次调用、零参数和全新Adam的前提下，手算对应如下：

| 项目 | 数学预期与归属 |
|---|---|
| 返回的loss | 更新前loss \(\ln3\)，shape `()` |
| 本次参数梯度 | `[1/12,1/12,-1/6]`，bias叶子 `(3,)` |
| 更新后m | `[1/120,1/120,-1/60]`，`(3,)` |
| 更新后v | `[1/144000,1/144000,1/36000]`，`(3,)` |
| 更新后step | 1，scalar `()` |
| 更新后参数 | 约 `[-0.001,-0.001,+0.001]`，`(3,)` |

参数近似沿用17.4的首步带修正手算，展示近似暂时忽略极小epsilon；实际准备代码保留 `eps=1e-8`，并有浮点舍入。此表不是Notebook输出、运行验证或loss下降证据。

### Prepared single call and scalar logging

最后只准备一次 `train_step` 调用。返回之后，`array.item()` 可把仅含一个元素的MLX数组转换成Python标量，适合日志；本节loss本来就是shape `()`，因此满足要求。这个转换放在单步返回和显式求值之后，不放进需要求导的loss函数。

```python
loss_before_update = train_step(
    model, X, Y, loss_and_grad_fn, optimizer
)
print("loss before update:", loss_before_update.item())
```

这是未来work副本的调用示例，不是本轮执行记录。若以后重复调用，同一个model与optimizer会继续前进；不应再把结果当作新状态下的“第一次更新”。本节没有循环、采样、语料读取、断言、测试或训练实验。

### Operational order, scope, and ledger

以后使用work副本时：在仓库根目录更新 → 用时间戳新文件名与 `cp -n` 安全复制76母版到已有 `notebooks/work/` → 选择项目kernel → 本节imports/类/toy数据/loss定义 → 阅读反向模式与状态衔接 → 在单步外构造同组model/函数变换/Adam → 了解update与eval的真实接口 → 定义普通单步函数 → 阅读旧loss/新状态的手算归属 → 使用唯一准备调用及返回后日志。所有依赖在本节可恢复。

17.5当轮只编写并发布本节，完成即停在大Step17，没有进入大Step18。没有运行Notebook、课程代码、测试、训练、实验、采样或用户work副本，没有读取或下载语料，没有代答练习，也不声称学员掌握或运行通过。只铺垫本单步必须的evaluation boundary；不扩展 `mx.compile`、图增长实验、异步执行、性能测量、训练循环、checkpoint或生成。

17.5验收后的累计实际修正：16.1=**1/2**、16.2=**1/2**（包含已记录的超时后完整回复补齐）、16.3至16.5及17.1至17.5各**0/2**。每小节最多两次实际修正，跨轮与换对话不重置；后续按请求逐小节至大Step20全部完成后停止。

### References and continuity

- [17.3母版](../notebooks/74_step17_3_nn_value_and_grad.ipynb)与[17.4母版](../notebooks/75_step17_4_adam_optimizer.ipynb)：共享参数模型、scalar loss、参数梯度树和明确的Adam设置。
- [MLX v0.32.2 nn/utils.py](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/utils.py)：针对Module的函数变换与调用参数。
- [MLX v0.32.2 optimizers.py](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/optimizers/optimizers.py)：Adam默认行为、Optimizer.update和状态推进。
- [MLX v0.32.2 Python transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp)与[core transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/transforms.cpp)：eval的参数树接收与求值行为。
- [MLX v0.32.2 array interface](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/array.cpp)：单元素数组的item接口。
- [Course roadmap](course-roadmap.md)与[teaching handoff](teaching-handoff.md)：末节材料状态、累计修正与止于大Step17的边界。
