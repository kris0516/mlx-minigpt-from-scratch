# Step 03 — Tensor Basics

## Step 03.1 — Scalar, Vector, Matrix, Tensor, Shape

这一小步只建立最基本的张量直觉。

### Scalar

单个数字。

```text
3.0
shape = ()
```

### Vector

一排数字。

```text
[1, 2, 3]
shape = (3,)
```

### Matrix

二维数字表。

```text
[[1, 2, 3],
 [4, 5, 6]]

shape = (2, 3)
```

### Tensor

深度学习中，tensor 可以泛指任意维度数组。

例如 3D tensor：

```text
shape = (2, 2, 3)
```

可以先理解成：

```text
2 个
  2 行
    3 列
```

本步暂时不学习 axis、reshape、transpose 或 broadcast。

## Step 03.2 — Axis

对：

```text
shape = (2, 2, 3)
```

可以逐层读取：

```text
axis 0: 2 个矩阵
axis 1: 每个矩阵 2 行
axis 2: 每行 3 个元素
```

Axis 从 0 开始编号。

索引示例：

```python
tensor[0]        # 固定 axis 0 的第 0 个位置
tensor[:, 0, :]  # axis 0 全部，axis 1 取第 0 行，axis 2 全部
tensor[:, :, 0]  # 前两轴全部，axis 2 取第 0 个元素
```

其中 `:` 表示该 axis 全部保留。

以后 Transformer 常见的：

```text
(B, T, C)
```

只是给三个 axis 起了语义名字：

```text
axis 0 = Batch
axis 1 = Time
axis 2 = Channel / Feature
```


## Step 03.3 — Reshape

`reshape` 改变 tensor 的逻辑形状，但要求元素总数保持一致。

例如：

```text
(2, 3) -> (3, 2)
```

因为：

```text
2 × 3 = 3 × 2 = 6
```

同样：

```text
(2, 3) -> (6,)
```

也是合法的。

最基本规则：

```text
旧 shape 各维长度的乘积 = 新 shape 各维长度的乘积
```

`reshape` 不等于 transpose；本阶段暂时不讲 transpose。


## Step 03.4 — Reshape vs Transpose

`reshape` 与 `transpose` 不是同一件事。

### reshape

```text
(2, 3) -> (3, 2)
```

元素按原顺序重新分组。

### transpose

```text
axis 0 <-> axis 1
```

会交换轴的顺序。

例如：

```text
[[1, 2, 3],
 [4, 5, 6]]
```

reshape 为 `(3, 2)`：

```text
[[1, 2],
 [3, 4],
 [5, 6]]
```

transpose 为 `(3, 2)`：

```text
[[1, 4],
 [2, 5],
 [3, 6]]
```

以后 Multi-Head Attention 会使用：

```text
(B, T, C)
-> reshape
(B, T, H, D)
-> transpose
(B, H, T, D)
```


## Step 03.5 — Broadcasting

Broadcasting 允许 shape 不完全相同的 tensor 做逐元素运算，只要它们的维度满足兼容规则。

例如：

```text
x.shape = (2, 3)
b.shape = (3,)
```

执行：

```python
x + b
```

逻辑上相当于把 `b` 沿缺少的最外层维度复用：

```text
[[1, 2, 3],      [[10, 20, 30],
 [4, 5, 6]]  +    [10, 20, 30]]
```

MiniGPT 中会用到：

```text
tok_emb: (B, T, C)
pos_emb:    (T, C)
```

相加时 `pos_emb` 会沿 Batch 维广播。


## Step 03.6 — Broadcasting Rules

Broadcasting 从最后一个维度开始，逐维向左比较。

每一对维度只要满足以下任一条件，就兼容：

1. 两个维度长度相同；
2. 其中一个维度长度为 1；
3. 一边缺少该维度。

示例：

```text
(2, 3) + (3,)   -> 可以
(2, 3) + (2, 1) -> 可以
(2, 3) + (2,)   -> 不可以
```

关键是从右往左比较，而不是从左往右。
