# Step 14 — Transformer Block: residuals, normalization, and MLP

目标：逐步组装 GPT 中反复堆叠的 Transformer computation block。

本 Step 保留现有 roadmap 的 14.1–14.5 结构。Step 14.1 材料已通过静态审核；当前只准备 **14.2 LayerNorm → RMSNorm historical path**，不进入 14.3。

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
