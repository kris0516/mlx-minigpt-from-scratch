# Step 12 — Causal Self-Attention

目标：把 Step 11 的 general self-attention 变成适用于 autoregressive next-token language modeling 的机制。

本 Step 保留现有 roadmap 的 12.1–12.5 结构。Step 12.1–12.4 材料已通过静态审核；当前只准备 **12.5 Inspect an attention matrix**，不进入 13.1。

## Step 12.1 — Autoregressive constraint

### Shifted next-token prediction

若完整 token sequence 是 `x_0,...,x_T`，训练时常使用：

```text
input : x_0, x_1, ..., x_(T-1)
target: x_1, x_2, ..., x_T
```

因此 input position `t` 使用 prefix `x_0...x_t` 预测 `x_(t+1)`。当前 token `x_t` 属于合法上下文；future positions 不属于。

### Autoregressive factorization

Autoregressive 的核心是概率分解：

```text
p(x_0,...,x_T) = product_t p(x_t | x_<t)
```

它描述的是信息条件，而不是特定架构。Transformer 可以没有 recurrent state transition，同时仍然遵守 autoregressive factorization。

### Visibility rule inside self-attention

Step 11 的 score matrix rows=Query positions、columns=Key positions。若 Query position `i` 用 `x_0...x_i` 预测 `x_(i+1)`，则它合法访问的 Key/Value positions 必须满足：

```text
j <= i
```

`j<i` 是 past，`j=i` 是 current，二者允许；`j>i` 属于 future，不允许读取。

### Future leakage

例如：

```text
input : <BOS> I    love   coffee
target: I     love coffee <EOS>
```

在 input position 1，任务是用 `<BOS>, I` 预测 `love`。若 general self-attention 允许 Query 1 读取 future input position 2，它会直接看到 `love`，也就是自己的 target。这属于 future leakage / cheating。

这种泄漏会让 training condition 与真实 generation condition 不一致：训练时 loss 可能很低，但真正推理时 future token 尚未存在。

### Parallel training versus causal information

训练为了并行效率，可以一次把整段 shifted input 放入 `(B,T,C)` tensor；但每个 position 必须在信息上“假装”自己只知道 prefix。矩阵并行不等于未来信息可以合法使用。

Step 12.1 只建立 autoregressive visibility requirement。怎样把 `j<=i` 真正编码进 attention scores，属于 Step 12.2 causal mask。

Step 12.1 材料准备到此停止，不进入 Step 12.2。

## Step 12.2 — Causal mask

### Strict upper triangle corresponds to future positions

Step 12.1 已定义 row `i` 为 Query position、column `j` 为 Key position，并要求 `j<=i`。因此 future positions `j>i` 正好位于 `T×T` score matrix 的 strict upper triangle；主对角线 `j=i` 保留，因为 current token 是预测下一个 token 的合法上下文。

### Additive causal mask

定义：

```text
M_(i,j) = 0      if j <= i
M_(i,j) = -inf   if j > i
```

然后把它加到 scaled scores：

```text
S_masked = S_scaled + M
```

合法位置保持原 score，future positions 被强制变成 `-inf`。

### Why softmax produces zero future weight

对 masked position，数学上 `exp(-inf)=0`，所以 softmax numerator 为 0；只要该 Query row 至少有一个合法 finite score，该未来位置最终 attention weight 就是 0。

Causal mask 保留 diagonal，因此每一行至少有当前 position 可用，不会出现整行全部被屏蔽的情况。

### MLX 0.32.2 construction

MLX 0.32.2 支持 `mx.triu(x,k=1)` 与 `mx.where(condition,x,y)`。Notebook 使用：

```text
future_upper = mx.triu(mx.ones((T,T)), k=1)
causal_mask = mx.where(future_upper > 0, -inf, 0.0)
```

然后：

```text
masked_scores = scaled_scores + causal_mask
A = mx.softmax(masked_scores, axis=-1)
```

### Shape and semantics

`scaled_scores:(B,T,T)`、`causal_mask:(T,T)`，通过 broadcasting 得到 `(B,T,T)` masked scores，再沿 Key axis softmax。

Mask 表示“能不能看”的结构约束；attention weights 表示在合法 positions 中“实际看多少”。

Step 12.2 材料准备到此停止，不进入 Step 12.3；stable-softmax / numerical-stability 专题和完整 causal attention head 留给后续既有小节。

## Step 12.3 — Stable attention computation

### Why naive softmax can overflow

Softmax 数学定义是 `exp(z_i)/sum_j exp(z_j)`，但若 logits 很大，例如 `[1000,999,998]`，直接 exponential 可能在有限精度下 overflow 成 `inf`，继而出现 `inf/inf -> nan`。

### Shift invariance and subtract-max

对同一 row 减去任意常数 c 不改变 softmax：

```text
softmax(z) = softmax(z - c)
```

最有用的选择是 `c=max(z)`。这样最大 shifted logit 为 0，其余不大于 0，exponentials 最大只到 1。

例如 `[1000,999,998] -> [0,-1,-2]`，最终 softmax 仍约为 `[0.6652,0.2447,0.0900]`。

### Underflow versus overflow

subtract-max 后很负的 logits 可能使 exponential underflow 到 0；对理论上已极小的 probability，这通常比 positive-exponential overflow 导致 NaN 更可控。这里不把 underflow 说成在所有数值任务里都无害。

### Compatibility with causal -inf masking

如果 masked row 是 `[5,-inf,-inf]`，subtract max 后仍是 `[0,-inf,-inf]`，exponentials 为 `[1,0,0]`。标准 causal mask 保留 diagonal，因此每行至少有一个 finite legal score；避免整行全 `-inf` 后发生 `-inf - (-inf)` 的无效情况。

### What scaling helps with

`1/sqrt(d_k)` 与 subtract-max stable softmax 解决的是不同问题。Scaling 控制 dot-product logits 的典型尺度，减少随 `d_k` 增长而过度进入 saturated/peaked softmax regime 的倾向，并改善 optimization conditions；subtract-max 则直接解决 exponential overflow 风险。

因此不能把 scaling 简化成“只是为了防止梯度消失”。Gradient behavior 与 saturation 有关，但 scaling 更基础的作用是控制 score scale 与 softmax 工作区间。

### MLX 0.32.2

项目固定 `mlx==0.32.2`。该版本 `mx.softmax(a, axis=..., precise=False)` 官方定义为 numerically stable softmax。实际模型代码应直接使用 `mx.softmax`，而不是手写朴素 `exp/sum(exp)`。

`precise=True` 对低精度输入使用 float32 reduction accumulation；这是 API 精度选项，本节不扩展成 mixed-precision 专题。

Step 12.3 材料准备到此停止，不进入 Step 12.4；完整 causal attention head 实现留给后续既有小节。

## Step 12.4 — Implement a single causal attention head

### Minimal trainable structure

本节把已学组件封装成一个 MLX `nn.Module`。输入 `X:(B,T,C)`，单头 width 记作 `d_h`，教学实现令 `d_k=d_v=d_h`。

Trainable parts 只有三套 Q/K/V projections：

```text
q_proj : C -> d_h
k_proj : C -> d_h
v_proj : C -> d_h
```

Scaling factor `1/sqrt(d_h)` 与 causal visibility rule 都属于结构常量，不需要训练。

### Forward order and shapes

```text
X                  (B,T,C)
-> Q/K/V           (B,T,d_h)
-> Q @ K^T         (B,T,T)
-> * 1/sqrt(d_h)   (B,T,T)
-> + causal mask   (B,T,T)
-> softmax(-1)     (B,T,T)
-> weights @ V     (B,T,d_h)
```

Mask 必须在 softmax 前作用到 scaled scores；softmax 沿 Key axis `axis=-1`。

### Non-trainable mask handling in MLX 0.32.2

MLX `nn.Module` 会把直接存放在 module 上的 `mx.array` attributes 收入 `parameters()`；arrays 默认 trainable，除非显式 `freeze`。

因此本节主实现不把 mask 存成 `self.causal_mask`，而是根据当前 T 在 `__call__` 内局部构造 `(T,T)` causal mask。这样它不会进入 parameter tree，也自然适配不同 T。

如果未来为了性能缓存 mask 到 module state，则应显式 freeze 对应 key，使其保留在 state/parameters 中但不进入 `trainable_parameters()`。

### Scope boundary

本节得到的是 single causal self-attention head。没有 Multi-Head Attention、head concatenation、output projection、residual/normalization/feed-forward，也不做 Step 12.5 的专门 attention-matrix inspection。

Step 12.4 材料准备到此停止，不进入 Step 12.5。

## Step 12.5 — Inspect an attention matrix

### Read rows as Queries and columns as Keys

对单个 sample，attention matrix 是 `A:(T,T)`。第 i 行表示 Query position i 对所有 Key positions 的 attention distribution；第 j 列对应 Key position j。

Causal self-attention 需要满足两个结构不变量：

```text
sum_j A_(i,j) = 1
A_(i,j) = 0 for j > i
```

也就是每个 Query row 的合法权重和为 1，strict upper triangle 的 future weights 全部为 0。

### Hand-computed expected matrix

Notebook 使用全零 scaled scores 作为 deterministic teaching case。加入 causal mask 后，每一行所有合法 logits 相等，所以 softmax 在合法 positions 上均匀分配：

```text
[[1,   0,   0,   0],
 [1/2, 1/2, 0,   0],
 [1/3, 1/3, 1/3, 0],
 [1/4, 1/4, 1/4, 1/4]]
```

这个 uniform pattern 是人工等-logit案例的手算预期，不代表真实训练后的 attention 通常均匀。

### Structural checks

Notebook 准备两类检查：

```text
row_sums = sum(A, axis=-1)
future_weights = triu(A, k=1)
```

预期 row sums 约为 1，strict-upper-triangle future weights 的最大绝对值为 0。

### Diagnostic head versus runtime results

Step 12.4 的 head 正常只返回 output。Step 12.5 提供 diagnostic variant，额外返回 attention weights，方便执行时打印与可视化。

该 diagnostic example 特意使用全零输入 `x=0`，且 Q/K/V 三个 Linear 都设为 `bias=False`。因此无论 projection weights 如何随机初始化，都有 `Q=K=V=0`：合法位置的 pre-softmax scores 全为 0，causal softmax 会在每行合法 prefix 上均匀分配，future weights 为 0，且 `out=A@V=0`。这些具体数值可以在运行前数学推导；由于母版未执行，它们应称为 expected values，而不是已经观测到的 runtime results。

Notebook 还提供一个 heatmap cell：横轴 Key position、纵轴 Query position，用于执行时观察 causal lower-triangular structure；attention weight 不能自动被解释为确定的人类语义关系。

Step 12.5 材料准备到此停止，不进入 Step 13.1，也不讨论或实现 Multi-Head Attention。
