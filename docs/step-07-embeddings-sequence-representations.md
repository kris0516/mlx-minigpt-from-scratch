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

## Step 07.2 — Embedding Geometry

### Norm and normalization

For a vector x, `||x||_2 = sqrt(sum_i x_i^2)`. Normalizing a nonzero vector with `x_hat = x / ||x||` changes its magnitude to 1 while preserving direction. Zero vectors have no defined direction.

### Euclidean distance

`d(a,b) = ||a-b||_2` measures absolute spatial distance and is sensitive to both direction and magnitude.

### Dot product

`a·b = ||a|| ||b|| cos(theta)`, so dot product depends on both vector norms and angle. It is not automatically cosine similarity.

### Cosine similarity

`cos(a,b) = (a·b) / (||a|| ||b||)` for nonzero vectors. It removes magnitude and compares direction. For normalized vectors, dot product equals cosine similarity.

### Pairwise geometry

If normalized embedding rows form `E_hat` with shape `(V,C)`, then `E_hat @ E_hat.T` has shape `(V,V)` and contains all pairwise cosine similarities.

Embedding geometry should not be over-interpreted: learned vectors optimize the training objective, and human-interpretable semantic clusters are not guaranteed.
