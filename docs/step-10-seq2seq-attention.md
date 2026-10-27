# Step 10 — Seq2Seq and the Birth of Attention

目标：理解 recurrent encoder–decoder Seq2Seq 如何把一个 source sequence 转换成另一个 target sequence，并为后续 fixed-vector bottleneck 与 Attention 的历史动机建立基础。

本 Step 保留现有 roadmap 的 10.1–10.5 结构。Step 10.1 材料已通过静态审核；当前只准备 **10.2 Fixed-vector bottleneck**，不进入 10.3。

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
