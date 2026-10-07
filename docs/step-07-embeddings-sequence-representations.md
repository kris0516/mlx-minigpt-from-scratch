# Step 07 — Embeddings and Sequence Representations

目标：把离散 token IDs 变成 Transformer 真正处理的连续 hidden representation，并理解顺序信息如何加入。

本 Step 共 5 个小节：

1. **07.1 nn.Embedding as Learned Lookup Table**
2. **07.2 Embedding Geometry**
3. **07.3 Build the (B,T,C) Representation**
4. **07.4 Position Information**
5. **07.5 Token + Position Addition**

## Step 07.1 — nn.Embedding as Learned Lookup Table

### Module and parameters

```python
embedding = nn.Embedding(V, C)
```

creates a trainable module whose core parameter is:

```text
embedding.weight.shape = (V,C)
```

with `V*C` parameters.

### Lookup semantics

For integer token IDs:

```text
embedding(ids) == embedding.weight[ids]
```

conceptually. The API performs row lookup/gather rather than treating token IDs as continuous numeric features.

Repeated token IDs read the same embedding row.

### Why not explicit one-hot?

Mathematically:

```text
one_hot(id) @ E == E[id]
```

but explicit one-hot batches require shape `(B,T,V)`. Direct lookup avoids materializing this large representation.

### Shape rule

Embedding preserves the index axes and appends the embedding dimension:

```text
scalar -> (C,)
(T,)   -> (T,C)
(B,T)  -> (B,T,C)
```

### Learning

Token IDs are discrete input indices, not trainable continuous parameters. Gradients update `embedding.weight`, especially rows used in the forward computation. Repeated uses of one token share the same row and contribute to the same parameter.

For the teaching MiniGPT:

```python
self.wte = nn.Embedding(vocab_size, n_embd)
```

so:

```text
wte.weight.shape = (vocab_size, n_embd)
```
