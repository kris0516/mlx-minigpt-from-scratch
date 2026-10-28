# Step 15 — From the Original Transformer to GPT / Decoder-Only Transformer

目标：解释 2017 Transformer 的 encoder–decoder architecture，以及 GPT 后续究竟保留、删除和重组了什么。

本 Step 保留现有 roadmap 的 15.1–15.5 结构。Step 15.1–15.3 材料已通过静态审核；当前只准备 **15.4 Implement `MiniGPT`**，不进入 15.5。

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

## Step 15.3 — MiniGPT top-level architecture

### Top-level shape chain

MiniGPT 的顶层数据流：

```text
token IDs (B,T)
-> token embedding (B,T,C)
+ learned position embedding (T,C)
-> initial hidden (B,T,C)
-> Block × N
-> final norm (B,T,C)
-> LM head C->V
-> logits (B,T,V)
```

### Token embedding

Token embedding table 为 `(V,C)`，将 categorical token IDs `(B,T)` lookup 成 continuous representations `(B,T,C)`。MLX 0.32.2 API：`nn.Embedding(num_embeddings=V,dims=C)`。

### Position embedding

本项目 MiniGPT 使用 learned absolute position table `(T_max,C)`。当前位置 IDs `0...T-1` lookup 为 `(T,C)`，broadcast 加到 token embeddings 上，因此要求 `T<=T_max`。

Causal mask 与 position embedding 不同：mask 规定 visibility，position embedding 提供 position-specific learned representation signal。本项目选择不是所有现代 LLM 的唯一 positional design。

### Block × N

每个 Transformer Block 保持 `(B,T,C)->(B,T,C)`。`Block × N` 通常表示 N 个结构相同但参数独立的 block instances，而不是默认共享一套 Block parameters。

### Final norm

当前课程使用 Pre-Norm Block，因此每个 Block 的最终 residual output 并不会自动在 block exit 再做一次 normalization。MiniGPT 顶层在 block stack 后使用 final RMSNorm，保持 `(B,T,C)`。

### LM head

LM head 将每个 position 的 C-dimensional hidden state 映射到 V-dimensional vocabulary logits：`(B,T,C)->(B,T,V)`。MLX 可使用 `nn.Linear(C,V,bias=False)`；stored weight shape 为 `(V,C)`。

Logits 是未归一化分数。本节不在顶层手动加 softmax，也不进入 next-token loss。

### Scope boundary

本节只建立顶层组件清单、职责与 shape flow。不实现 `MiniGPT(nn.Module)`、真实 Block list 或 top-level shape assertions，也不处理 causal-mask state-registration 等 Step 15.4 实现细节。

Step 15.3 材料准备到此停止，不进入 Step 15.4。

## Step 15.4 — Implement `MiniGPT`

### Clean MLX module structure

MiniGPT 由 token embedding、learned absolute position embedding、独立 Transformer Block list、final RMSNorm 与 LM head 组成。

```text
token_ids (B,T)
-> token + position representations (B,T,C)
-> Block × N
-> final norm
-> LM head
-> logits (B,T,V)
```

`self.blocks=[TransformerBlock(...) for _ in range(num_blocks)]` 创建独立子模块对象。MLX 0.32.2 的 `nn.Module` 会递归追踪 list / tuple / dict 中的 child modules 与 arrays。

### Shape and construction constraints

Constructor 检查 positive `vocab_size/dims/num_heads/num_blocks/max_seq_len` 以及 `dims % num_heads == 0`。

Forward 要求 token IDs 为 rank-2 `(B,T)`，并显式要求 `1<=T<=max_seq_len`。位置 IDs 在 forward 内通过 `mx.arange(T)` 创建；最终 logits contract 为 `(B,T,V)`。

### Structural constants versus trainable arrays

MLX Module 中直接保存的 `mx.array` 会被 `parameters()` 递归收集，若未冻结也会进入 `trainable_parameters()`。

因此当前 Attention 不把 causal mask 保存成 `self.causal_mask`，而是在每次 forward 中根据当前 T 局部创建。MiniGPT position IDs 同样 forward-local。`head_dim`、scale、max sequence length 等用 Python int/float metadata 表示。

如果未来缓存 structural array，可以把它作为 module state 后显式 `freeze(..., keys=..., recurse=False)`；被冻结 array 仍在 module state/parameters 中，但不会进入 trainable parameter set。

### Current implementation choices

本节沿用已有 Block：Pre-Norm + RMSNorm + causal MHA + GELU MLP。Learned absolute positions 是本项目教学选择；token embedding 与 LM-head weights 暂时独立，不做 weight tying。

### Scope boundary

本节只实现干净的 `MiniGPT(nn.Module)` 与必要 shape constraints。不计算完整 MiniGPT 参数总数，不做完整逐行 forward shape trace，也不进入 loss / training。

Step 15.4 材料准备到此停止，不进入 Step 15.5。
