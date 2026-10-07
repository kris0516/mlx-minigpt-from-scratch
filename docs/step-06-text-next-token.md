# Step 06 — From Text to a Next-Token Learning Problem

目标：把原始文本转换成神经网络可以处理的离散序列，并最终形成 next-token prediction 训练样本。

本 Step 共 5 个小节：

1. **06.1 Vocabulary and Character Tokenizer**
2. **06.2 One-Hot vs Learned Embedding**
3. **06.3 Next-Token Shift**
4. **06.4 Context Window / Block Size**
5. **06.5 Sampling Batches**

## Step 06.1 — Vocabulary and Character Tokenizer

Character-level tokenizer 将每一个字符视为一个 token。

基本流程：

```text
text
-> vocabulary
-> stoi / itos
-> encode / decode
-> integer token IDs
```

### Vocabulary

```python
chars = sorted(set(text))
vocab_size = len(chars)
```

在 character-level tokenizer 中，空格、换行和标点同样是 token。

### Token IDs

Token ID 只是离散类别编号，不携带“数值大小”或几何距离意义。

真正的连续向量表示将在 learned embedding 阶段获得。

### stoi / itos

```text
stoi: character -> integer ID
itos: integer ID -> character
```

### Encode / decode

```python
encode(text) -> list[int]
decode(ids)  -> str
```

正确实现应该支持 round trip：

```text
decode(encode(text)) == text
```

### Scope

本项目先用 character tokenizer 保持教学透明。现代 LLM 更常使用 byte/subword/BPE 类 tokenizer；这些将在扩展部分讨论。


## Step 06.2 — One-Hot vs Learned Embedding

### Token IDs are labels, not numeric features

A token ID is an index/category label. Treating IDs as continuous values creates arbitrary order and distance relationships that depend only on tokenizer numbering.

### One-Hot

For vocabulary size `V`, one-hot vectors live in `V` dimensions:

```text
token 0 -> [1,0,0,...]
token 1 -> [0,1,0,...]
```

This removes fake numeric order, but all tokens remain symmetric and the representation is sparse.

### Learned embedding

Use a trainable table:

```text
E.shape = (V,C)
```

Each token ID selects one row.

Mathematically:

```text
one_hot(id) @ E == E[id]
```

but practical embedding lookup does not need to materialize one-hot vectors.

### Embedding width

`C` does not need to be greater than or equal to `V`. Vocabulary size and embedding width are independent design quantities.

### Geometry

Embedding vectors are trainable and can develop task-useful geometry. Human-interpretable clusters are not guaranteed; the vectors optimize the training objective.

### Shape bridge

```text
token IDs: (B,T)
embedding table: (V,C)
lookup output: (B,T,C)
```

Embedding parameter count is `V*C`.


## Step 06.3 — Next-Token Shift

### Autoregressive target construction

For a raw chunk of `T+1` tokens:

```text
chunk = [t0, t1, t2, ..., tT]
```

construct:

```text
X = chunk[:-1] = [t0, ..., t(T-1)]
Y = chunk[1:]  = [t1, ..., tT]
```

Both X and Y have length `T`. This is a slice, not a circular roll.

### One block provides T supervision signals

For `hello`:

```text
X = hell
Y = ello
```

a causal model learns:

```text
h    -> e
he   -> l
hel  -> l
hell -> o
```

Thus one length-`T` forward pass can supervise all `T` positions.

### Batch shapes

Starting from raw chunks:

```text
(B,T+1)
```

shift to:

```text
X: (B,T)
Y: (B,T)
```

Later:

```text
X IDs            -> embedding -> (B,T,C)
model logits                  -> (B,T,V)
targets Y                     -> (B,T)
```

Y remains integer class labels.

### No label leakage

Targets are used for loss comparison, not fed as the answer to the same position.

However, because `Y[t] == X[t+1]`, a model that can see future X positions could cheat. Therefore causal modeling also requires a causal visibility constraint:

```text
position t may only use positions <= t
```

Target shifting defines what to predict; the causal mask defines what information may be used to predict it.

### Teacher forcing

During training, ground-truth previous tokens are available in X. During autoregressive inference, future ground-truth tokens are unavailable, so generated tokens are appended back into the context step by step.
