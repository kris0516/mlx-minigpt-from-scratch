# Step 11 — Self-Attention from First Principles

目标：不直接背最终 Attention 公式，而是从信息检索角色出发，逐步推导 Query / Key / Value、compatibility score、scaling、softmax 与 Value retrieval。

本 Step 保留现有 roadmap 的 11.1–11.5 结构。Step 11.1 材料已通过静态审核；当前只准备 **11.2 Q/K/V Linear projections**，不进入 11.3。

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
