# Step 04 — From Matrix Multiplication to Neural Networks

目标：从已经掌握的矩阵乘法出发，建立 MiniGPT 中最基础的神经网络组件。

本 Step 共 4 个小节：

1. **04.1 Linear** — 矩阵乘法如何成为可学习的线性层
2. **04.2 Parameters, Bias, Module** — 参数是什么，bias 为什么存在，MLX 如何组织网络
3. **04.3 Non-linearity** — 为什么纯 Linear 堆叠不够，ReLU/GELU 的意义
4. **04.4 MLP** — 组合出 MiniGPT 中的前馈网络

## Step 04.1 — Linear（canonical merged version）

本节整合此前三个 4.1 版本，正式保留以下全部主题：

### A. Linear = feature mixing

输入 `x = [x0, x1]`，若要得到 3 个输出：

```text
y0 = w00*x0 + w01*x1
y1 = w10*x0 + w11*x1
y2 = w20*x0 + w21*x1
```

因此：

```text
W.shape = (3,2)
        = (output_features, input_features)
```

每个输出都是输入特征的一次加权组合，因此 Linear 具有 feature mixing 的意义。

### B. 1D vector / row matrix / column matrix

```text
(2,)   -> 1D vector
(1,2)  -> 1×2 row matrix
(2,1)  -> 2×1 column matrix
```

三者可包含相同数字，但 axis 结构不同。

### C. 两种等价的矩阵写法

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

### D. 1D matmul 的特殊规则

```text
(2,) @ (2,3) -> (3,)
```

理解上，左侧 1D vector 在矩阵乘法语义中临时按 `(1,2)` 参与运算，计算结束后临时 size-1 维被移除。

原始变量本身不会永久变成 `(1,2)`。

### E. Batch 共用同一套参数

```text
X.shape = (B, C_in)
W.T     = (C_in, C_out)

X @ W.T -> (B, C_out)
```

每个样本使用同一个权重矩阵。这是 parameter sharing 的基本形式。

MiniGPT 中则常见：

```text
(B, T, C_in)
-> Linear
(B, T, C_out)
```

前面的 Batch / Time 维保留，Linear 主要变换最后一维。

### F. `nn.Linear` 与 `layer`

```python
import mlx.nn as nn
layer = nn.Linear(2, 3, bias=False)
```

`nn.Linear(...)` 创建一个 Linear 对象；`layer` 不是普通函数。

对象内部保存：

```text
layer
├── weight
├── bias（可选）
└── forward rule
```

Python 对象可以实现可调用行为，因此：

```python
layer(x)
```

表示调用这层的前向计算。

### G. 拆掉黑盒

当 `bias=False` 时：

```python
layer(X)
```

可直接与：

```python
X @ layer.weight.T
```

比较，两者应一致。

这证明 `nn.Linear` 的底层核心仍是前面手算的矩阵乘法；训练阶段才会让随机初始化的 `weight` 逐步变成有用参数。


## Step 04.2 — Parameters, Bias, and `nn.Module`

### Parameter

区分：

```text
input data      -> 每个样本变化
architecture    -> 由开发者设定
parameters      -> 训练过程中被更新
```

例如 `Linear(2,3,bias=False)` 的 weight shape 为 `(3,2)`，因此有 6 个权重参数。

### Bias

带 bias 的 Linear：

```text
y = xW^T + b
```

`Linear(2,3,bias=True)`：

```text
weight: (3,2) -> 6
bias:   (3,)  -> 3
total          -> 9
```

bias 让映射具有平移自由度。严格数学上 `Wx+b` 是 affine transformation，但神经网络库通常仍称其为 Linear layer。

### `nn.Module`

`nn.Module` 用于组织神经网络中的参数和子模块。

自定义模型：

```python
class TinyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 3)

    def __call__(self, x):
        return self.proj(x)
```

### Parameter tree

`model.parameters()` 会递归取得 Module 和子 Module 中的数组参数，并保留嵌套结构。

例如：

```text
TinyProjector
└── proj
    ├── weight
    └── bias
```

可用 `tree_flatten` 展平后统计参数总数。

MLX 中公开的 `mx.array` Module 成员会进入参数体系；默认可训练，除非 freeze。固定常量需要和模型参数区分开，这一点会在 causal mask 实现时再次使用。
