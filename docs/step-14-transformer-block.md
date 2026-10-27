# Step 14 — Transformer Block: residuals, normalization, and MLP

目标：逐步组装 GPT 中反复堆叠的 Transformer computation block。

本 Step 保留现有 roadmap 的 14.1–14.5 结构。Step 14.1–14.3 材料已通过静态审核；当前只准备 **14.4 Feed-Forward MLP inside the Block**，不进入 14.5。

## Step 14.1 — Residual connection

### Shape-preserving addition

Step 13 的 Multi-Head Attention 输入与输出均为 `(B,T,C)`，因此可以使用同 shape residual addition：

```text
y = x + F(x)
```

标准 Transformer residual 的核心是对应元素相加，不依赖 broadcasting 来补齐 shape。

### Update-as-correction

`x` 可视为当前 representation baseline，`F(x)` 可作为 learned correction / update，因此 `y=x+Δx`。这是一种有用直觉，不意味着训练一定强制 correction 很小。

### Gradient highway

对 `y=x+F(x)`：

```text
J_y = I + J_F
g_x = g_y + J_F^T g_y
```

其中 `g_y` 是不经过 F branch Jacobian 的 direct identity-path contribution。这解释了 residual 为什么为 gradient flow 提供更直接的路径。

### Accuracy boundary

Gradient highway 不是“梯度永远不会消失/爆炸”的保证。若 `F(x)=-x`，则 `y=0`、`J_y=I-I=0`，gradient 可以完全抵消。

同一个反例也说明 residual 不保证 mapping 可逆或信息零损失。Residual 的准确作用是提供 explicit identity route 和 correction formulation，而不是数学保证。

Residual addition 自身没有 learned parameters；trainable parameters 位于 F，例如 Attention sublayer。

Step 14.1 材料准备到此停止，不进入 Step 14.2；LayerNorm/RMSNorm、RMS/epsilon/gamma 与 normalization placement 留给后续既有小节。

## Step 14.2 — LayerNorm → RMSNorm historical path

### Per-token feature normalization

对 Transformer representation `x:(B,T,C)`，LayerNorm 与 RMSNorm 都对每个 `(b,t)` 的 `C`-dimensional feature vector 单独做 normalization；不跨 batch 或 sequence positions 混合统计量，shape 保持 `(B,T,C)`。

### LayerNorm

LayerNorm 对每个 token 计算 feature mean 与 population variance：

```text
mu = mean(x)
var = mean((x-mu)^2)
x_hat = (x-mu) / sqrt(var + eps)
y = gamma * x_hat + beta
```

因此 LayerNorm 同时执行 re-centering 与 re-scaling。Gamma / beta 是 learned per-feature parameters。

### RMSNorm

RMSNorm 使用：

```text
rms = sqrt(mean(x^2) + eps)
y = gamma * x / rms
```

它不减 feature mean，因此保留 re-scaling 而省去 re-centering。常见 RMSNorm 只有 learned gamma；MLX 0.32.2 的 `nn.RMSNorm` 没有 beta parameter。

### Epsilon and gamma

`eps` 是固定 numerical-stability constant，用于避免 denominator 接近/等于 0；它不是 learned parameter。`gamma` 是 learned per-feature scale，使模型可以在 normalization 后重新调节各 feature 的幅度。

### Historical path

LayerNorm 在 2016 年提出，并成为原始 Transformer 等序列模型中的经典 normalization 方法。RMSNorm 在 2019 年提出，可理解为对 LayerNorm normalization idea 的简化：去掉 mean-centering，保留基于 RMS 的 scale normalization。

这不意味着 RMSNorm 在所有任务上必然优于 LayerNorm，也不意味着两者数学等价。

### MLX 0.32.2

```text
nn.LayerNorm(dims=C, eps=1e-5, affine=True, bias=True)
nn.RMSNorm(dims=C, eps=1e-5)
```

LayerNorm 默认 gamma=1、beta=0；RMSNorm 默认 gamma=1。MLX 0.32.2 的 RMSNorm mean-square accumulation 使用 float32 precision。

Step 14.2 材料准备到此停止，不进入 Step 14.3；normalization 位于 residual branch 前还是后，以及 Pre-Norm / Post-Norm，留给下一既有小节。

## Step 14.3 — Pre-Norm vs Post-Norm

### Placement formulas

使用 generic sublayer F 与 normalization N：

```text
Post-Norm: y = N(x + F(x))
Pre-Norm : y = x + F(N(x))
```

两者 shape 都保持 `(B,T,C)`；差别是 normalization 在 computation graph 中的位置。

### Historical framing

原始 Transformer 的经典 Add & Norm 对应 Post-Norm pattern。随着模型变深，大量后续 Transformer / LLM architectures 使用 Pre-Norm 或相关 variants，一个重要动机是改善深层 optimization 与 gradient flow。

这不意味着 Pre-Norm 在任何任务、深度与训练设定下都无条件优于 Post-Norm。

### Gradient-path difference

Pre-Norm：

```text
J_pre = I + J_F J_N
g_x = g_y + J_N^T J_F^T g_y
```

因此存在一个不经过 normalization / sublayer Jacobian 的 direct identity gradient contribution `g_y`。

Post-Norm：

```text
J_post = J_N (I + J_F)
g_x = (I + J_F)^T J_N^T g_y
```

Residual identity contribution 仍存在于 `(I+J_F)`，但最终从 y 回到 x 的 gradient 仍要通过 normalization Jacobian。

### Block skeleton

```text
pre_norm_step : x + sublayer(norm(x))
post_norm_step: norm(x + sublayer(x))
```

本节只使用 generic sublayer F，不实现 Feed-Forward MLP 或完整 Transformer Block。

Step 14.3 材料准备到此停止，不进入 Step 14.4。

## Step 14.4 — Feed-Forward MLP inside the Block

### Position-wise transformation

对 `X:(B,T,C)`，Feed-Forward MLP 对每个 `X[b,t,:]` 独立应用同一套 learned function。参数在 batch 与 sequence positions 间共享，因此 MLP 本身不构造 T×T cross-token interaction。

Attention 的核心职责是 cross-position mixing；MLP 的核心职责是 within-position nonlinear feature transformation。由于输入 representation 已经可能包含 Attention 混入的上下文，MLP 仍可加工 contextual features。

### Classic expansion path

经典教学结构：

```text
(B,T,C)
-> Linear(C,4C)
-> activation
-> Linear(4C,C)
-> (B,T,C)
```

`4C` 是常见 expansion choice，不是数学硬约束。

### Why the activation matters

如果两层 Linear 中间没有 nonlinear activation，忽略 bias 时 `xW1W2 = xW_eff`，整体仍可合并成一个 linear map。因此中间 activation 是两层 MLP 获得 nonlinear expressivity 的关键。

本节使用 MLX 0.32.2 `nn.GELU()`；标准 GELU 为 `x * Phi(x)`。

### MLX module

```text
fc1: Linear(C,4C)
GELU
fc2: Linear(4C,C)
```

MLX `nn.Linear` 直接作用于最后一个 feature axis，因此整个 `(B,T,C)` tensor 可一次输入，无需 Python token loop。

### Parameter count

本节代码使用 `bias=True`。两层 MLP 参数总数：

```text
weights: 4C^2 + 4C^2 = 8C^2
biases : 4C + C = 5C
total  : 8C^2 + 5C
```

若两层都 `bias=False`，则为 `8C^2` weights。

### Link to residual / pre-norm

Step 14.3 的 generic `F` 现在可以具体取为 MLP，例如概念上的 `x_new = x + MLP(N(x))`。本节只说明 MLP 能作为 sublayer F，不把 Attention branch、MLP branch、两套 normalization 与 residual 全部组装。

Step 14.4 材料准备到此停止，不进入 Step 14.5；完整 Block implementation、shape invariants、Block parameter count 与 numerical sanity checks 留给下一既有小节。
