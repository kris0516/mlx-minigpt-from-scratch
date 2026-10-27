# Step 10 — Seq2Seq and the Birth of Attention

目标：理解 recurrent encoder–decoder Seq2Seq 如何把一个 source sequence 转换成另一个 target sequence，并为后续 fixed-vector bottleneck 与 Attention 的历史动机建立基础。

本 Step 保留现有 roadmap 的 10.1–10.5 结构。Step 10.1–10.3 材料已通过静态审核；当前只准备 **10.4 Score → softmax → weighted sum**，不进入 10.5。

## Step 10.1 — Encoder-decoder Seq2Seq

### Source and target sequences

Seq2Seq 将任务写成 `source sequence -> target sequence`。本节使用 `T_src` 和 `T_tgt` 区分 source/target sequence length；二者不要求相等。source 与 target 也可以有不同 vocabulary，分别记作 `V_src` 与 `V_tgt`。

`<BOS>`（Beginning Of Sequence）与 `<EOS>`（End Of Sequence）是 vocabulary 中的 special tokens，用于告诉 decoder 从哪里开始以及何时结束生成。

### Encoder

source IDs `(B,T_src)` 经过 source embedding 得到 `(B,T_src,C_src)`，再由 recurrent encoder（教学示例使用 MLX GRU）产生 hidden-state sequence `(B,T_src,H)`。

经典 recurrent Seq2Seq 取 final encoder state：

```text
s = encoder_states[:, -1, :] -> (B,H)
```

作为 encoder 与 decoder 之间的 context/final-state bridge。本节只建立这个 bridge，不分析它的 fixed-vector bottleneck；后者属于 Step 10.2。

### Decoder

decoder 使用 encoder final state 作为 initial hidden state：

```text
h_-1^dec = s
```

target-side token embedding `(B,C_tgt)` 与 decoder recurrent state 一起产生新的 decoder hidden state，再经 output projection 映射到 target vocabulary logits `(B,V_tgt)`。

encoder 与 decoder 可以有独立参数，因为它们分别承担 source representation 与 target generation 两种角色。

### Teacher forcing and inference

训练时常使用 teacher forcing：把 shifted ground-truth target token 作为 decoder input。例如：

```text
decoder input : <BOS> A B C
target label  : A     B C <EOS>
```

推理时没有完整正确 target，因此从 `<BOS>` 开始，把模型上一步生成的 token 作为下一步输入，直到 `<EOS>` 或长度上限。

Teacher forcing 是训练策略，不是 encoder–decoder 架构本身的必要数学组成。

### Shape flow

Encoder：

```text
(B,T_src)
-> source embedding -> (B,T_src,C_src)
-> encoder GRU      -> (B,T_src,H)
-> final state s    -> (B,H)
```

Decoder training path：

```text
(B,T_tgt)
-> target embedding -> (B,T_tgt,C_tgt)
-> decoder GRU with initial hidden s -> (B,T_tgt,H)
-> output projection -> (B,T_tgt,V_tgt)
```

概念上不同样本可以有不同 source/target lengths；同一个 dense batch 仍需要 padding/masks/length bookkeeping 等工程处理。本节不展开 padding 实现。

Step 10.1 材料准备到此停止，不进入 Step 10.2。

## Step 10.2 — Fixed-vector bottleneck

### Interface bottleneck

经典 recurrent Seq2Seq 的 encoder 会产生完整 hidden-state trajectory `(B,T_src,H)`，但 decoder 只通过 final encoder state `s=(B,H)` 接收 source information。因此 bottleneck 发生在 encoder→decoder interface，而不是因为 encoder 本身只有一个 state。

```text
encoder_states: (B,T_src,H)
        ↓ take final state only
s:              (B,H)
        ↓
decoder
```

### Fixed width and practical representation pressure

当 `T_src` 增长时，context width H 不随 source length 增长。更长 source 中更多、分散的任务相关信息需要持续压缩到同一个 H-dimensional final state。

这应理解为 practical representation bottleneck，而不是用 `T_src > H` 做“数学上绝对无法编码历史”的维度计数证明。现实困难来自有限精度、有限容量、连续可训练更新、优化与可恢复性等共同约束。

### Long source-to-target recurrent path

记 source position 为 `j`，decoder position 为 `u`。早期 source information 先经过 `T_src-1-j` 个 encoder recurrent transitions 到 final state，再经过 `u+1` 个 decoder recurrent transitions 到 `h_u^dec`。

因此只计 recurrent transitions：

```text
path length = (T_src - 1 - j) + (u + 1)
            = T_src + u - j
```

这说明 Step 9 的 long-range credit-assignment 问题并未消失，而是跨过 encoder 与 decoder 两段 recurrent chain。

### Teacher forcing is a different issue

Teacher forcing 改变的是 training 时 decoder previous-token input 的来源；它不会改变 `source -> final encoder state -> decoder` 的接口。因此 teacher forcing 不能消除 fixed-vector bottleneck。

### Larger H and stronger gated RNNs

增大 H、使用更强 LSTM/GRU 或改善优化可以提高 bottleneck 的容量与可训练性，但不改变经典结构中 decoder 只通过 final encoder state 访问 source information 的事实。

Step 10.2 到这里提出关键历史问题：encoder 明明已经产生每个 source position 的 hidden states，为什么 decoder 只能依赖最后一个？本节不回答如何选择 source states，不进入 Step 10.3，也不介绍 Attention 的计算机制。

## Step 10.3 — Bahdanau-style attention intuition

### From one fixed context to all encoder states

经典 fixed-vector Seq2Seq 虽然产生 `(B,T_src,H)` 的 encoder state trajectory，但 decoder 主要只接收 final encoder state。Attention 的核心结构变化是：decoder 每个生成步骤都可以重新利用整条 encoder state trajectory，而不是永远只依赖同一个固定 source summary。

### Decoder state as the current information need

在 decoder step `u`，previous decoder state `h_(u-1)^dec` 表示当前 target-side context。一个 learned alignment / compatibility function 将它与每个 source-side encoder state `h_j^enc` 比较，得到 raw alignment score：

```text
e_(u,j) = a(h_(u-1)^dec, h_j^enc)
```

这是一种 content-based lookup：相关性由当前 decoder state 与 encoder representation 的内容共同决定，而不是硬编码 target/source 位置对应关系。本节不展开 alignment function 的具体网络形式。

### Attention weights as a distribution

每个 decoder step 对所有 source positions 得到一组 normalized attention weights `α_(u,j)`，满足：

```text
α_(u,j) >= 0
sum_j α_(u,j) = 1
```

因此 `α_u` 可视为 source positions 上的 attention distribution。它不要求 one-hot，可以同时关注多个 positions；也不应自动解释为人类语义上的真实概率或绝对可靠解释。

### Dynamic focus across decoder steps

不同 decoder steps 的 previous decoder state 不同，因此可以产生不同 attention distributions。对一个样本，把各 step 的 weights 堆叠后可形成 `(T_tgt,T_src)` 的矩阵；每一行表示一个 target/decoder step 对 source positions 的相对关注。

Step 10.3 Notebook 中的 heatmap weights 是人工给定的 intuition example，不是模型 score 计算结果。

### Attention context vector

对 step `u`，encoder states 为 `(B,T_src,H_enc)`，attention weights 为 `(B,T_src)`；Attention 利用它们形成 step-specific context vector `c_u=(B,H_enc)`。

与 fixed-vector Seq2Seq 的同一个 source summary 不同，Attention 可以产生 `c_0,c_1,c_2,...` 等随 decoder step 改变的 source summaries。

本节只建立这一概念结构。不推导 softmax，也不做 score→softmax→weighted-sum 的精确数值计算；这些属于 Step 10.4。

Step 10.3 材料准备到此停止，不进入 Step 10.4。

## Step 10.4 — Score → softmax → weighted sum

### Raw score to normalized attention weight

固定 decoder step `u` 时，alignment function 为各 source positions 产生 raw scores `e_(u,j)`。Raw score 可以是任意实数，不能直接当作满足非负且和为 1 的 attention distribution。

Softmax 定义为：

```text
alpha_(u,j) = exp(e_(u,j)) / sum_k exp(e_(u,k))
```

`exp(x)=e^x` 始终为正，并保持输入大小顺序；除以同一 decoder step 上所有 source positions 的指数和后，得到非负且总和为 1 的 normalized weights。

例如 scores `[2,1,0]`：指数约为 `[7.389,2.718,1]`，总和约 `11.107`，因此 weights 约 `[0.665,0.245,0.090]`。

### Weighted sum of encoder states

若 encoder states 为 `h_j^enc`，attention context 定义为：

```text
c_u = sum_j alpha_(u,j) * h_j^enc
```

例如 weights `[0.665,0.245,0.090]` 与 states `[1,0]`、`[0,2]`、`[3,1]` 组合后，context 约为 `[0.935,0.580]`。

因为 weights 非负且和为 1，这可以理解为 encoder states 的 weighted average。

### MLX softmax and broadcasting

MLX 0.32.2 支持 `mx.softmax(scores, axis=-1)`。对 one-decoder-step scores `(B,T_src)`，`axis=-1` 就是 source-position axis，因此每个 batch row 独立归一化，输出 shape 仍为 `(B,T_src)`。

为了把 `(B,T_src)` weights 逐 position 乘到 `(B,T_src,H_enc)` encoder states 上，可用 `weights[...,None]` 将其变为 `(B,T_src,1)`，再通过 broadcasting 得到 `(B,T_src,H_enc)`；最后 `mx.sum(..., axis=1)` reduce source-position axis，得到 context `(B,H_enc)`。

### Complete attention computation

固定 decoder step `u`：

```text
e_(u,j) = alignment score
alpha_(u,j) = softmax over j
c_u = sum_j alpha_(u,j) h_j^enc
```

shape flow：

```text
scores  (B,T_src)
-> softmax over source axis
weights (B,T_src)
-> scale encoder states (B,T_src,H_enc)
-> sum over source axis
context (B,H_enc)
```

本节只完成第一次精确 Attention computation。Bahdanau additive scoring network 的内部参数形式和后续 Transformer 历史桥接都不在本节展开。

Step 10.4 材料准备到此停止，不进入 Step 10.5。
