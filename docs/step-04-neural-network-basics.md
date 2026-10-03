# Step 04 — From Matrix Multiplication to Neural Networks

目标：从已经掌握的矩阵乘法出发，建立 MiniGPT 中最基础的神经网络组件。

本 Step 共 4 个小节：

1. **04.1 Linear** — 矩阵乘法如何成为可学习的线性层
2. **04.2 Parameters, Bias, Module** — 参数是什么，bias 为什么存在，MLX 如何组织网络
3. **04.3 Non-linearity** — 为什么纯 Linear 堆叠不够，ReLU/GELU 的意义
4. **04.4 MLP** — 组合出 MiniGPT 中的前馈网络

## Step 04.1 — Linear

Linear 的核心仍然是矩阵乘法。

对于输入最后一维长度为 2，希望得到 3 个输出特征时，可以使用：

```text
W.shape = (3, 2)
```

并计算：

```text
y = x @ W.T
```

每一个输出维度，都是输入各维度的一次加权组合。

例如：

```text
x = [2, -1]

W =
[[1, 0],
 [0, 1],
 [1, 1]]
```

则：

```text
y = [2, -1, 1]
```

MLX 的：

```python
nn.Linear(2, 3, bias=False)
```

内部持有形状为 `(3, 2)` 的 weight。其核心仍然对应：

```python
x @ weight.T
```

区别是，这个 weight 会成为模型参数，并在后续训练过程中通过梯度下降被调整。
