# Step 12 — Causal Self-Attention

目标：把 Step 11 的 general self-attention 变成适用于 autoregressive next-token language modeling 的机制。

本 Step 保留现有 roadmap 的 12.1–12.5 结构。Step 11 材料已通过静态审核；当前只准备 **12.1 Autoregressive constraint**，不进入 12.2。

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
