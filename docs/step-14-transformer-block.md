# Step 14 — Transformer Block: residuals, normalization, and MLP

目标：逐步组装 GPT 中反复堆叠的 Transformer computation block。

本 Step 保留现有 roadmap 的 14.1–14.5 结构。Step 13 材料已通过静态审核；当前只准备 **14.1 Residual connection**，不进入 14.2。

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
