# Step 11 — Self-Attention from First Principles

目标：不直接背最终 Attention 公式，而是从信息检索角色出发，逐步推导 Query / Key / Value、compatibility score、scaling、softmax 与 Value retrieval。

本 Step 保留现有 roadmap 的 11.1–11.5 结构。Step 10 材料已通过静态审核；当前只准备 **11.1 Why separate Query, Key, and Value**，不进入 11.2。

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
