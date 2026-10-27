# Step 15 — From the Original Transformer to GPT / Decoder-Only Transformer

目标：解释 2017 Transformer 的 encoder–decoder architecture，以及 GPT 后续究竟保留、删除和重组了什么。

本 Step 保留现有 roadmap 的 15.1–15.5 结构。Step 14 材料已通过静态审核；当前只准备 **15.1 Original Transformer architecture**，不进入 15.2。

## Step 15.1 — Original Transformer architecture

### Encoder–decoder structure

Original Transformer 仍然是 Seq2Seq encoder–decoder architecture。Source side 进入 encoder stack，产生 encoder memory；target-side shifted input 进入 decoder stack，并通过 cross-attention 读取 encoder memory。

### Encoder stack

Encoder layer 的核心是 source-side Multi-Head Self-Attention + position-wise FFN，并配合 residual / normalization。忽略 padding mask 时，source self-attention 不是 decoder-style causal attention；source positions 可以相互访问。

Encoder stack 输出保持 `(B,T_src,C)`，作为 encoder memory。

### Decoder stack

Original decoder layer 包含三类 sublayers：

```text
masked target self-attention
encoder–decoder cross-attention
position-wise FFN
```

Masked self-attention 遵守 `j<=i`，防止 future target leakage。

### Cross-attention roles and shapes

Cross-attention 中 Query 来自 decoder states，Key / Value 来自 encoder memory：

```text
Q : (B,T_tgt,d_k)
K : (B,T_src,d_k)
V : (B,T_src,d_v)
```

因此 score matrix 是 `(B,T_tgt,T_src)`；multi-head 版本为 `(B,H,T_tgt,T_src)`，不要求是方阵。

### Historical implementation contrast

2017 Original Transformer 的经典实现使用 encoder + decoder、Post-Norm / LayerNorm、ReLU FFN，并在 decoder 中加入 cross-attention。课程 Step 14.5 的 Pre-Norm + RMSNorm + GELU causal Block 是后续现代化教学选择，不能当作原始 2017 decoder layer 的逐项复刻。

### Autoregressive decoder

训练时 target 仍使用 shifted input，并依靠 masked self-attention 保证并行训练时不读取 future target tokens。Cross-attention 则可以读取完整 encoder memory（除 padding 等 mask）。

### Scope boundary

本节只恢复原始 Transformer 全局 architecture，不解释为什么 GPT 能移除 separate encoder，也不实现 decoder-only GPT。

Step 15.1 材料准备到此停止，不进入 Step 15.2。
