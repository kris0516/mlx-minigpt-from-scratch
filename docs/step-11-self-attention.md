# Step 11 — Self-Attention from First Principles

目标：不直接背最终 Attention 公式，而是从信息检索角色出发，逐步推导 Query / Key / Value、compatibility score、scaling、softmax 与 Value retrieval。

本 Step 保留现有 roadmap 的 11.1–11.5 结构。Step 11.1–11.4 材料已通过静态审核；当前只准备 **11.5 Weighted Value retrieval**，不进入 12.1。

## Step 11.1 — Why separate Query, Key, and Value

### Self-Attention starts from one sequence representation set

Self-Attention 输入记作 `X=(B,T,C)`。对一个样本，sequence positions 的 base representations 是 `x_0,...,x_(T-1)`。`self` 表示发起 lookup 与作为 lookup candidates 的 base representations 来自同一 sequence set；它不意味着 Query、Key、Value 数值必须相同。

### Three functional roles

Query 表示当前 position 正在寻找什么信息；Key 表示一个候选 position 用哪些 features 参与相关性匹配；Value 表示该候选如果获得较高 attention weight，真正向查询位置贡献什么内容。

```text
Query = what am I looking for?
Key   = why should I match you?
Value = what content do I return if selected?
```

Key 的核心职责是 addressing / matching，Value 的核心职责是 payload / content delivery。

### Why one representation serving all three roles is restrictive

把同一个 base representation 直接同时用作 asking、matching 与 payload 在数学上可以工作，但会把三个不同目标耦合在同一 feature space：

- requester 想强调的 features 不一定等于 candidate 为了被找到应该暴露的 features；
- 用于 relevance matching 的 features 不一定等于真正应该被传递的 payload features；
- 同一个 token/position 在不同 interaction 中可能需要不同角色视角。

因此分开的 role-specific views 允许模型为 Query / Key / Value 学习不同的 task-specific lookup spaces。这里的结论是“更灵活”，不是“不分开就无法做 Attention”。

### Language intuition

在 `The animal did not cross the street because it was tired.` 这个直觉例子中，`it` 的 Query 可以强调“寻找 singular/animate antecedent”；`animal` 与 `street` 的 Keys 可暴露不同匹配属性；而被选中候选的 Value 可以携带比匹配标签更丰富的上下文语义。

这个例子只用于解释角色差异，不表示 Attention 自动等价于确定的语言学指代解释。

### Self-Attention vs Cross-Attention

Self-Attention 中 Query/Key/Value 的 base source 都来自同一 sequence；Cross-Attention 中 Query 可以来自 decoder side，而 Key/Value 来自 encoder side。Q/K/V 首先是功能角色，而不是三个固定来源。

下一节才学习怎样通过 learned linear projections 从同一个 `X` 产生具体的 Query / Key / Value tensors。本节不写 projection equations，也不计算 Query–Key compatibility matrix。

Step 11.1 材料准备到此停止，不进入 Step 11.2。

## Step 11.2 — Q/K/V Linear projections

### Learned role-specific views

Self-Attention 输入仍为 `X=(B,T,C)`。Q/K/V 的角色分离通过三套独立 trainable linear maps 落地；同一个 base representation 可以被重新组合到不同 feature spaces，而不是简单复制三份 X。

### Mathematical projections and shapes

本节使用 row-vector convention，并引入 `d_k` 与 `d_v`：

```text
W_Q : (C,d_k)
W_K : (C,d_k)
W_V : (C,d_v)

Q = X W_Q -> (B,T,d_k)
K = X W_K -> (B,T,d_k)
V = X W_V -> (B,T,d_v)
```

Q 与 K 使用兼容的 matching width；具体 compatibility computation 属于 Step 11.3。Value width `d_v` 可以不同于 `d_k`，二者都不要求等于输入 width C。

所有 sequence positions 共享同一套 projection parameters，因此 projection 改变 feature axis，但不删除 T axis。

### Tiny hand-computable example

Notebook 使用 `T=2,C=3,d_k=d_v=2` 的小矩阵：

```text
X = [[1,2,0],
     [0,1,1]]
```

配合三套不同教学矩阵，得到：

```text
Q = [[1,2],
     [1,0]]

K = [[3,1],
     [1,1]]

V = [[1,4],
     [1,3]]
```

这个例子只展示同一个 X 经不同 learned-map roles 可以得到不同 views，不计算任何 Query–Key score。

### MLX 0.32.2 storage convention

项目固定 `mlx==0.32.2`。该版本 `nn.Linear(input_dims, output_dims, bias=True)` 存储 `weight.shape=(output_dims,input_dims)`，forward 使用 `x @ weight.T + b`。

因此数学上若 `W_Q:(C,d_k)`，MLX `q_proj.weight` 的存储 shape 是 `(d_k,C)`；K/V 同理。Notebook 为了与纯矩阵公式一致使用 `bias=False`。

### Projection meaning

Q/K/V projections 可以理解为三套可学习 feature coordinate systems：Query space 表达“我需要什么”，Key space 表达“我能否匹配”，Value space 表达“被选中后我贡献什么”。它们是 role-specific views，不是原样复制 X。

Step 11.2 材料准备到此停止，不进入 Step 11.3。

## Step 11.3 — `QK^T` as compatibility score

### Dot product as one Query–Key score

给定 `q_i,k_j∈R^(d_k)`，单个 compatibility score 使用 dot product：

```text
s_(i,j) = q_i · k_j = sum_m q_(i,m) k_(j,m)
```

输入两个 `d_k` 维 vectors，输出一个 scalar raw score。这个 score 还不是 probability 或 attention weight。

### Dot product versus cosine similarity

Dot product 同时受向量方向与 magnitude 影响；cosine similarity 会额外除以两个向量的 L2 norms：

```text
cos(theta) = (q · k) / (||q||_2 ||k||_2)
```

例如 `q=[1,0]`、`k1=[1,0]`、`k2=[2,0]`：两个 cosine similarities 都是 1，但 dot products 分别为 1 和 2。只有 q、k 都单位归一化时，dot product 才与 cosine similarity 数值相同。

### Pairwise scores as `QK^T`

去掉 batch 后，`Q,K:(T,d_k)`。将 K 转置为 `(d_k,T)` 后：

```text
S = Q K^T -> (T,T)
```

其中 `S_(i,j)=q_i·k_j`。第 i 行是 Query position i 对所有 Key positions 的 scores；第 j 列是所有 Query positions 对 Key position j 的 scores。

Step 11.2 的教学 Q/K 例子得到：

```text
S = [[5,3],
     [3,1]]
```

这一矩阵碰巧对称只是数值巧合；一般独立 Q/K projections 不保证 `QK^T` 对称。

### Batch shape

真实 Self-Attention 中：

```text
Q : (B,T,d_k)
K : (B,T,d_k)
K_transposed : (B,d_k,T)

Q @ K_transposed -> (B,T,T)
```

每个 batch sample 独立得到一个 `T×T` raw compatibility matrix。

### Scope boundary

本节只建立 unscaled raw compatibility scores。Scaling、`1/sqrt(d_k)` 与 softmax 属于 Step 11.4，不在本节展开。

Step 11.3 材料准备到此停止，不进入 Step 11.4。

## Step 11.4 — Scaling and softmax

### Why dot-product scale grows with d_k

对单个 raw score `s=q·k=sum_m q_m k_m`，若用独立、零均值、单位方差的教学假设理解各分量，则每个乘积项的方差约为 1，`d_k` 项相加后：

```text
Var(s) ≈ d_k
Std(s) ≈ sqrt(d_k)
```

这解释了为什么 feature width 增大时 raw dot-product scores 的典型幅度容易一起增大。这个推导是 scaling 的经典统计动机，不是对训练后相关 Q/K components 的绝对定律。

### Scaling by sqrt(d_k)

定义 scaled score：

```text
scaled_scores = QK^T / sqrt(d_k)
```

在上述教学假设下，除以 `sqrt(d_k)` 会把 score variance 的典型量级拉回约 1。Scaling 不改变 score matrix shape，也不会直接把 scores 变成概率。

较大的 raw logits 容易让 softmax 变得非常 peaked；scaling 控制进入 softmax 前的典型数值幅度，使 optimization 通常更温和。

### Tiny softmax comparison

例如 raw scores `[4,2,0]`、`d_k=4`，scaling 后是 `[2,1,0]`。

```text
softmax([4,2,0]) ≈ [0.867,0.117,0.016]
softmax([2,1,0]) ≈ [0.665,0.245,0.090]
```

两者保持同样的相对排序，但 scaled version 没那么尖锐。

### Softmax axis and attention-weight shape

Step 11.3 已定义 `(B,T,T)` score matrix 的 rows=Query positions、columns=Key positions。因此每个 Query row 应沿 Key axis 归一化。

MLX 中：

```text
A = mx.softmax(scaled_scores, axis=-1)
```

输出 shape 仍为 `(B,T,T)`，并且 `A[b,i,:]` 是 sample b 中 Query position i 对所有 Key positions 的非负、和为 1 的 attention distribution。

### Complete weight formula

到本节为止，Attention weight computation 已成为：

```text
A = softmax(QK^T / sqrt(d_k))
```

本节只得到 A，不执行 Value retrieval。`A @ V` 与完整 single-head output 属于 Step 11.5。

Step 11.4 材料准备到此停止，不进入 Step 11.5。

## Step 11.5 — Weighted Value retrieval

### Attention weights retrieve Value content

Step 11.4 得到 `A:(B,T,T)`，其中 `A[b,i,j]` 表示 Query position i 对 Key position j 的 normalized attention weight。A 只决定“从哪里取多少”；真正被取回的 payload 位于 `V:(B,T,d_v)`。

单个 Query position 的输出是：

```text
o_i = sum_j A_(i,j) v_j
```

所有 Query positions 一次性计算就是：

```text
O = A @ V
```

shape 为 `(B,T,T) @ (B,T,d_v) -> (B,T,d_v)`。因此输出 width 跟随 `d_v`，而不是 `d_k`。

### Tiny full single-head calculation

Notebook 使用 `T=2,C=2,d_k=d_v=2` 的完整例子，从：

```text
X
-> Q/K/V projections
-> QK^T
-> divide by sqrt(d_k)
-> softmax
-> A
-> A @ V
-> O
```

一路算到底。

教学例子得到：

```text
A ≈ [[0.5000,0.5000],
     [0.8044,0.1956]]

O ≈ [[1.0000,2.0000],
     [1.6088,0.7824]]
```

第一行 Query 对两个 Keys 等权，因此输出是两个 Values 的平均；第二行 Query 更偏向 Key 0，因此输出更接近 `v_0`。

### Context-dependent representations

`O` 的第 i 行不再只是 position i 自己原来的 Value，而是根据该 Query 自己的 attention distribution 对整条 sequence 的 Values 做 weighted mixture 得到的 context-dependent representation。

### Full single-head formula

到此可以把 Step 11 的 general single-head self-attention 压缩为：

```text
O = softmax(QK^T / sqrt(d_k)) V
```

课程没有从最终公式开始背，而是先逐步建立 Q/K/V 角色、linear projections、compatibility score、scaling、softmax 与 Value retrieval 的因果关系。

### Scope boundary

当前仍是 general self-attention，没有 causal mask、autoregressive visibility constraint、Multi-Head Attention、output projection、residual connection 或 Transformer block。

Step 11.5 材料准备到此停止，不进入 Step 12.1。
