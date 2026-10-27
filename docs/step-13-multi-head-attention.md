# Step 13 — Multi-Head Attention

目标：理解 GPT-style Multi-Head Attention 背后的 exact tensor organization，而不是只记最终公式。

本 Step 保留现有 roadmap 的 13.1–13.5 结构。Step 13.1–13.3 材料已通过静态审核；当前只准备 **13.4 Transpose for batched attention**，不进入 13.5。

## Step 13.1 — Why multiple heads

### One head means one learned Q/K/V view

Single causal attention head 已经拥有一套 `W_Q/W_K/W_V`，能够形成一个 learned Query space、Key space 与 Value space。

一个 head 并非不能表达复杂关系，但所有 interaction 都要经过同一套 projection spaces，因此它的表示方式更受单一视角约束。

### Multiple learned subspaces

对 head `h`，概念上可以写：

```text
Q^(h) = X W_Q^(h)
K^(h) = X W_K^(h)
V^(h) = X W_V^(h)
```

不同 heads 拥有独立 learned projections，因此同一个 X 可以在多个 learned subspaces 中形成不同 compatibility geometry、attention pattern 与 Value payload view。

### No fixed human semantics per head

Multi-head architecture 不规定 `Head 1=syntax`、`Head 2=pronoun`、`Head 3=position`。真实 head behavior 取决于训练、layer、input 与其他 components；heads 可以出现 specialization，也可以重叠、冗余或难以映射到单一人类标签。

因此准确表述是：multiple heads 提供多个独立可学习 interaction subspaces，而不是多个预先命名的人类概念模块。

### Tiny two-head projection example

Notebook 用同一个 `X:(3,2)` 和两个手工 projection views：一个只保留第一个 input feature，另一个只保留第二个。两套 causal attention 得到不同 expected matrices，直观展示不同 subspaces 能形成不同 attention patterns。

这些 projection matrices 是人为构造的教学例子；真实模型中的 head-specific projections 应由训练学习。

### Scope boundary

Step 13.1 只解释为什么需要多个 heads。不进入 fused QKV projection、`C=H×D` channel split、`(B,T,H,D)` reshape/transposition、head concatenation 或 output projection。

Step 13.1 材料准备到此停止，不进入 Step 13.2。

## Step 13.2 — Fused QKV projection

### One larger projection instead of three calls

此前把 Q/K/V 写成三套独立 full-width projections：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

其中 `W_Q/W_K/W_V:(C,C)`。把三者沿 output-feature direction 横向拼接，可以定义：

```text
W_QKV = [W_Q | W_K | W_V] : (C,3C)
```

于是一次 `X W_QKV` 就产生 `(B,T,3C)` fused output，其最后一个 feature axis 依次包含 Q、K、V 三段。

### MLX shape convention

MLX 0.32.2 `nn.Linear(input_dims=C, output_dims=3*C)` 的 stored weight shape 是 `(3C,C)`；这与数学 row-vector convention 中的 `(C,3C)` 互为转置布局。

Forward：

```text
X            : (B,T,C)
qkv_proj(X)  : (B,T,3C)
```

### Split Q/K/V

MLX 0.32.2 支持：

```text
q, k, v = mx.split(qkv, indices_or_sections=3, axis=-1)
```

因为最后一个 axis 长度是 `3C`，可等分为三个 `C`，得到：

```text
Q : (B,T,C)
K : (B,T,C)
V : (B,T,C)
```

这里仍然只是 Q/K/V 三块，还没有把 C 再拆成 heads。

### Parameter-count equivalence

bias=False 时，三套 `Linear(C,C)` 总共有 `3C^2` weights；一个 `Linear(C,3C)` 同样有 `3C^2` weights。

bias=True 时两种写法也都增加总共 `3C` biases。因此 fused QKV 不是 parameter compression。

### Why fuse

Fused projection 把三个同源线性映射组织成一次更大的 projection，通常有利于减少调用/dispatch overhead 并使用更大的 matrix multiplication；但真实性能依赖 framework、compiler、hardware 与 shape，不应承诺固定倍数加速。

Step 13.2 材料准备到此停止，不进入 Step 13.3；`C=H×D` 与 channel-to-head reshape 留给下一既有小节。

## Step 13.3 — Split channels into heads

### `C = H × D`

Step 13.2 已得到 `Q/K/V:(B,T,C)`。标准等宽 Multi-Head organization 把 model width C 写成：

```text
C = H × D
```

其中 H 是 number of heads，D 是 head dimension。这里的 H 是 head count，不是前面 RNN 章节曾使用的 hidden-width 记号。

因此要求 `C % H == 0`，并定义 `D = C // H`。

### Exact index mapping

把 feature index `c` 写成 `c=hD+d`，于是：

```text
Q_heads[b,t,h,d] = Q[b,t,hD+d]
```

K/V 同理。这个 reshape 不做新的 projection、不聚合数值、不新增参数，只把一个 C-dimensional feature axis 逻辑上重新组织成 H×D。

### Shape transformation

```text
Q : (B,T,C) -> (B,T,H,D)
K : (B,T,C) -> (B,T,H,D)
V : (B,T,C) -> (B,T,H,D)
```

元素数量保持 `B*T*C = B*T*H*D`。

### MLX 0.32.2

MLX 0.32.2 的 `mx.reshape(a, shape)` 在保持元素总数的前提下改变 array shape。本节使用：

```text
q_heads = mx.reshape(q, (B,T,H,D))
k_heads = mx.reshape(k, (B,T,H,D))
v_heads = mx.reshape(v, (B,T,H,D))
```

本节只讨论 tensor semantics，不承诺 reshape 在底层实现中一定 zero-copy。

### Heads as tensor organization

GPT-style multiple heads 首先体现为 tensor axis organization，而不是必须构造 H 个独立 Python Module 并循环执行。不同 head 对应 Q/K/V output channels 中不同的 learned channel blocks。

Step 13.3 材料准备到此停止，不进入 Step 13.4；`(B,T,H,D)->(B,H,T,D)` transpose 与 batched attention 留给下一既有小节。

## Step 13.4 — Transpose for batched attention

### `(B,T,H,D) -> (B,H,T,D)`

Step 13.3 已得到 Q/K/V `(B,T,H,D)`。为了让每个 head 的最后两个 axes 变成 `(T,D)`，交换 T 与 H：

```text
(B,T,H,D) -> (B,H,T,D)
```

精确索引关系：

```text
Q_BHTD[b,h,t,d] = Q_BTHD[b,t,h,d]
```

MLX 0.32.2 使用 `mx.swapaxes(a,1,2)`。

### Batched matrix multiplication

对 Q/K 都得到 `(B,H,T,D)` 后，再把 K 的最后两个 axes 交换成 `(B,H,D,T)`。MLX `mx.matmul` 会用最后两个 axes 做矩阵乘法，并把更前面的 axes 作为 batch/broadcast dimensions：

```text
(B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)
```

因此 `scores[b,h,:,:]` 就是 sample b、head h 自己的 T×T compatibility matrix。

### Causal mask broadcasting

causal mask 仍可保持 `(T,T)`，通过 broadcasting 作用到 `(B,H,T,T)` scores；同一个 visibility pattern 服务所有 batch samples 和 heads，而具体 scores 仍是 head-specific。

### Logical head independence versus hardware mapping

Head-wise attention score computation 中，每个 head 只使用自己的 Q/K D-dimensional channels，不做 cross-head Q/K dot products。这是 logical head independence。

但 H 个 logical heads 不对应 H 个固定 GPU cores。MLX/compiler/kernel 会把整个 batched tensor operation 调度到硬件执行资源；具体 physical mapping 不是由 head index 决定的。

Step 13.4 材料准备到此停止，不进入 Step 13.5；head outputs 的 transpose-back、reshape/concatenation 与 output projection 留给下一既有小节。
