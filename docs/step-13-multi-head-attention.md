# Step 13 — Multi-Head Attention

目标：理解 GPT-style Multi-Head Attention 背后的 exact tensor organization，而不是只记最终公式。

本 Step 保留现有 roadmap 的 13.1–13.5 结构。Step 13.1 材料已通过静态审核；当前只准备 **13.2 Fused QKV projection**，不进入 13.3。

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
