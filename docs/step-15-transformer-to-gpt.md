# Step 15 — From the Original Transformer to GPT / Decoder-Only Transformer

目标：解释 2017 Transformer 的 encoder–decoder architecture，以及 GPT 后续究竟保留、删除和重组了什么。

本 Step 保留现有 roadmap 的 15.1–15.5 结构。Step 15.1 材料已通过静态审核；当前只准备 **15.2 Why autoregressive GPT can be decoder-only**，不进入 15.3。

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

## Step 15.2 — Why autoregressive GPT can be decoder-only

### Autoregressive task factorization

GPT 的核心目标是同一条 token stream 上的 next-token modeling：

```text
p(x_0,...,x_T) = product_t p(x_t | x_<t)
```

预测下一个 token 所需要的条件信息就是同一 stream 中已经出现的 prefix，而不是一份独立 encoder memory。

### Shifted supervision is not encoder–decoder separation

训练时 `input=x_0...x_(T-1)` 与 `label=x_1...x_T` 是同一 underlying sequence 的 shifted views，只用于 supervision；它们不是 Original Transformer 中独立 source stream / target stream 的 architecture roles。

### Causal self-attention is sufficient for prefix conditioning

输入 position i 用于预测 `x_(i+1)`，允许访问 `j<=i`。因此 causal self-attention 恰好提供 `p(next | prefix)` 所需的信息集合。

### Remove encoder and cross-attention

Pure autoregressive LM 没有 separate source encoder memory，所以可以移除：

```text
encoder stack
encoder memory
encoder–decoder cross-attention
```

保留 decoder-side causal self-attention + FFN / residual / normalization lineage。

Decoder-only 并不意味着把 2017 decoder layer 原封不动保留，因为原版 decoder 还包含 cross-attention；GPT-style block 删除了这一 sublayer。

### Prompt-conditioned generation

Conditional context 可以被序列化为同一 stream 的 prefix，例如 `[instruction/source | response]`。生成 response 时，全部 prompt/source tokens 已位于 causal past，因此 response positions 可以通过 causal self-attention 读取它们。

这解释了 decoder-only 为什么仍可承担大量 prompt-conditioned generation，但不证明 encoder–decoder architecture 在所有 Seq2Seq 任务上都劣于 decoder-only。

### Training versus inference

训练可并行处理整条 shifted input tensor，但 causal mask 限制 future visibility；inference 时未来 token 尚不存在，因此仍需逐 token autoregressive generation。

### Scope boundary

本节只解释 decoder-only 的 sufficiency argument。不列出 MiniGPT 顶层模块，不实现 token/position embeddings、Block stacking、final norm 或 LM head。

Step 15.2 材料准备到此停止，不进入 Step 15.3。
