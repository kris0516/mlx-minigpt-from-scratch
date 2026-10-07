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

## Step 07.3 — Build the (B,T,C) Representation

### From IDs to hidden states

Token IDs with shape `(B,T)` pass through `nn.Embedding(V,C)` to produce `(B,T,C)`. Each scalar `X[b,t]` is replaced by one C-dimensional embedding row.

### Axis semantics

`B` selects a batch sample, `T` selects a token position, and `C` indexes hidden features. Thus `h[b,t,:]` is one token's complete hidden vector.

### C as model width

`C = n_embd` becomes the shared hidden width used across Transformer blocks. Internal sublayers may temporarily expand or project features, but residual-compatible block outputs normally return to `(B,T,C)`.

### Mixing roles

Linear/MLP layers primarily mix the final C axis independently at each `(b,t)` position. Attention later introduces cross-position mixing along T.

### Memory and position preview

A `(B,T,C)` tensor contains `B*T*C` values, so activation memory scales with all three dimensions. Token embedding lookup preserves token identity but does not itself encode absolute position; explicit position information is introduced next.

## Step 07.4 — Position Information

### Token identity vs position identity

Token embedding rows depend on token ID, not where the token appears. Reordering a sequence reorders the same token vectors; the vectors themselves do not gain an absolute-position identity.

### RNN order vs self-attention symmetry

RNN recurrence `h_t=f(x_t,h_{t-1})` is inherently order-sensitive because changing token order changes the computation path.

Unmasked self-attention without positional signals is permutation-equivariant: permuting input rows permutes output rows in the same way. This is not permutation invariance.

### Learned absolute positional embedding

The teaching MiniGPT uses `nn.Embedding(block_size, C)` for positions. For a current sequence length T, `mx.arange(T)` is looked up to produce `(T,C)` positional vectors.

### Causal-mask caveat and limitations

A causal mask itself depends on position indices and therefore breaks full arbitrary-permutation symmetry. It provides an order-dependent visibility structure, while positional representations explicitly encode position features. They serve different roles.

Learned absolute positions are simple but do not directly encode relative distance and do not naturally extrapolate beyond the learned `0...block_size-1` positions.

### Bridge to addition

Repeated copies of the same token share one token-embedding row but receive different position vectors. The next lesson combines `(B,T,C)` token embeddings with `(T,C)` position embeddings by broadcasting addition.

## Step 07.5 — Token + Position Addition

### Broadcasting

Token embeddings have shape `(B,T,C)` and learned positional embeddings have shape `(T,C)`. Broadcasting treats the latter as `(1,T,C)` and reuses the same position vectors across the batch.

For each location: `h[b,t,:] = token_emb[b,t,:] + pos_emb[t,:]`.

### Same token, different position

Repeated token IDs read the same token-embedding row, but different positions add different position vectors, producing different initial hidden representations.

### Addition vs concatenation

Addition preserves width C. Concatenating two C-dimensional vectors produces width 2C and usually requires a wider downstream network or a projection back to C. Concatenation is not mathematically wrong; addition is a compact architectural choice used by early GPT-style models.

### Information trade-off

The mapping `(e,p) -> e+p` is not uniquely invertible, so addition is not a lossless encoding of two separate C-dimensional vectors. Token embeddings, positional embeddings, and downstream layers are jointly trained so that the combined C-dimensional state is useful for the task.

A useful linear identity is `W(e+p)=We+Wp`, although later nonlinear layers and attention make the full computation more expressive.

### Step 7 complete

The representation path is now: token IDs `(B,T)` -> token embeddings `(B,T,C)`, position IDs `(T,)` -> position embeddings `(T,C)`, broadcasting addition -> initial hidden states `(B,T,C)`.
