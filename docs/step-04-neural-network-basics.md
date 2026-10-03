# Step 04 — From Matrix Multiplication to Neural Networks

目标：从已经掌握的矩阵乘法出发，建立 MiniGPT 中最基础的神经网络组件。

本 Step 共 4 个小节：

1. **04.1 Linear** — 矩阵乘法如何成为可学习的线性层
2. **04.2 Parameters, Bias, Module** — 参数是什么，bias 为什么存在，MLX 如何组织网络
3. **04.3 Non-linearity** — 为什么纯 Linear 堆叠不够，ReLU/GELU 的意义
4. **04.4 MLP** — 组合出 MiniGPT 中的前馈网络

## Step 04.1 — Linear

### 1D vector 不等于 row matrix

下面三种 shape 必须区分：

```text
(2,)   -> 1D vector
(1, 2) -> 1×2 row matrix
(2, 1) -> 2×1 column matrix
```

它们可以包含相同的两个数字，但 rank 与轴结构不同。

### 为什么 `W.shape = (3,2)`

输入有 2 个特征，输出希望有 3 个特征。

每个输出都需要两个输入的加权组合：

```text
y0 = w00*x0 + w01*x1
y1 = w10*x0 + w11*x1
y2 = w20*x0 + w21*x1
```

因此需要 3 组权重，每组 2 个数：

```text
W.shape = (3,2)
          ↑  ↑
        输出 输入
```

### 两种等价写法

列向量写法：

```text
W      @ x_col -> y_col
(3,2)   (2,1)    (3,1)
```

features-last / row-vector 写法：

```text
x_row @ W.T -> y_row
(1,2)  (2,3)   (1,3)
```

两者只是数据摆放方向不同，计算出的三个特征值相同。

### 为什么 `(2,) @ (2,3)` 可以计算？

若左操作数是一维 vector：

```text
x.shape = (2,)
```

矩阵乘法 `@` 会使用 1D operand 的特殊规则。理解上可以把左边暂时补成：

```text
(2,) -> (1,2)
```

于是：

```text
(1,2) @ (2,3) -> (1,3)
```

完成乘法后，临时补进去的最前面那个 size-1 维度会被移除，所以实际返回：

```text
(3,)
```

重要：`x` 本身并没有永久变成 `(1,2)`。

显式写：

```python
x.reshape(1, 2) @ W.T
```

才会真正得到 shape `(1,3)`。

### MLX `nn.Linear`

MLX 的：

```python
nn.Linear(2, 3, bias=False)
```

内部权重 shape 是：

```text
(3,2)
= (output_dims, input_dims)
```

核心计算对应：

```python
x @ weight.T
```

真实 MiniGPT 中输入通常为：

```text
(B, T, C_in)
```

Linear 将最后一个维度变成：

```text
(B, T, C_out)
```

而 Batch 和 Time 维保持不变。


### `layer` 不是普通函数，而是可调用对象

执行：

```python
layer = nn.Linear(2, 3, bias=False)
```

会创建一个 `nn.Linear` 对象，并把它赋给变量 `layer`。

这个对象内部保存：

```text
layer
├── weight
├── bias（如果启用）
└── 前向计算逻辑
```

因此：

```python
layer.weight
```

可以访问这一层自己的权重。

而：

```python
out = layer(x)
```

并不是在调用一个普通 Python 函数，而是在调用这个对象定义的“可调用”行为。Python 中对象可以实现 `__call__`，于是可以像函数一样使用。

对 `nn.Linear` 而言，忽略 bias 时，概念上等价于：

```python
out = x @ layer.weight.T
```

所以要区分：

```python
layer = nn.Linear(...)
```

= 创建一层，并初始化/保存这层自己的参数。

而：

```python
layer(x)
```

= 用这层当前保存的参数对输入 `x` 做一次前向计算。
