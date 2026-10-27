# Step 09 — BPTT, Vanishing Gradients, LSTM, and GRU

目标：理解 Vanilla RNN 的长期 credit assignment 为什么困难，以及 gating 为什么被引入。

本 Step 仍保持既有 5 个小节；本轮只修正 **09.1**，不进入 09.2：

1. **09.1 Backpropagation Through Time**
2. **09.2 Vanishing and Exploding Gradients**
3. **09.3 LSTM Cell State and Gates**
4. **09.4 GRU**
5. **09.5 Why LSTM/GRU Still Did Not Solve Everything**

## Step 09.1 — Backpropagation Through Time

### 统一符号与项目 API 基线

本节统一：

```text
B = batch size
T = sequence length
t = 0,1,...,T-1
C = token embedding / input feature width
V = tokenizer vocabulary size
H = RNN hidden-state width（可与 C 不同）
h_-1 = initial hidden state
h_(T-1) = final hidden state
```

`V` 是 tokenizer 整个 vocabulary 的不同 token 类型数，不是当前 T 个 token 中去重后的数量；T 会计入重复 token positions。

仓库 `pyproject.toml` 静态固定 `mlx==0.32.2`。MLX v0.32.2 `python/mlx/nn/layers/embedding.py` 的构造签名为：

```python
Embedding(num_embeddings: int, dims: int)
```

所以关键字调用必须写：

```python
import mlx.core as mx
import mlx.nn as nn

embedding = nn.Embedding(num_embeddings=V, dims=C)
```

`vocab_size` 可以作为项目变量名，但不是该版本 `nn.Embedding` 的关键字参数名。位置参数 `nn.Embedding(V, C)` 也有效。

### 偏导 ∂ 与 tanh 导数

多变量函数 `L(w_x,w_h,b)` 对单个变量的局部导数用 partial derivative：`∂L/∂w_h`。把多个偏导组织起来形成 gradient。

Vanilla scalar RNN：

```text
a_t = w_x*x_t + w_h*h_(t-1) + b
h_t = tanh(a_t)
```

`tanh'(a)=1-tanh^2(a)`。当 `a=0` 时导数为 1；当 `a=1` 时 `tanh(1)≈0.761594`、导数约 `0.419974`。当 `|a|` 很大时 tanh 接近 ±1、曲线变平、导数趋近 0（saturation）。本轮只建立局部导数直觉，不展开 9.2 的数值后果。

scalar recurrent local derivative：

```text
∂h_t/∂h_(t-1) = w_h * (1 - h_t^2)
```

### Jacobian 与列向量 gradient

向量对向量的局部导数用 Jacobian。若 `y=f(x)`，`J_ij=∂y_i/∂x_j`。例如：

```text
y1 = 2*x1 + x2
y2 = -x1 + 3*x2

J = [[ 2, 1],
     [-1, 3]]
```

forward 小变化按列向量写为 `dy = J dx`。若 backward upstream gradient 采用列向量 `g_y=∇_y L`，则：

```text
g_x = J^T g_y
```

例如 `g_y=[5,-2]^T`，得到 `g_x=[12,-1]^T`。转置与矩阵顺序不是装饰；shape 要满足 `(n,m)(m,1)=(n,1)`，且矩阵乘法不可交换。

RNN 中 `J_(t,t-1)=∂h_t/∂h_(t-1)` 的 shape 为 `(H,H)`。

### 三步 scalar BPTT（保留既有正确例子）

设 `T=3`、`t=0,1,2`、`h_-1=0`，参数：

```text
w_x = 0.7
w_h = 0.5
b   = 0.1
x   = [1.0, 0.5, -0.5]
target = 0.2
```

只在 final state `h_(T-1)=h_2` 定义 `L=0.5*(h_2-y)^2`。forward：

```text
a_0 = 0.800000   h_0 = 0.664037
a_1 = 0.782018   h_1 = 0.653864
a_2 = 0.076932   h_2 = 0.076780
L   = 0.00759153
```

定义 `delta_t=∂L/∂a_t`，backward 得：

```text
delta_2 ≈ -0.122493
dL/dh_1 ≈ -0.061247
delta_1 ≈ -0.035061
dL/dh_0 ≈ -0.017531
delta_0 ≈ -0.009801
```

`range(T)` 对 T=3 产生 `0,1,2`；`reversed(range(T))` 产生 `2,1,0`，符合 reverse-mode 从后向前的依赖顺序。

### Σ、shared parameter gradient 与 outer product

`Σ` 表示求和：`Σ_(t=0)^(T-1)c_t = c_0 + ... + c_(T-1)`。例如 `0.1 + (-0.2) + 0.3 = 0.2`。

scalar shared-parameter gradients：

```text
∂L/∂w_x = Σ_t delta_t*x_t
∂L/∂w_h = Σ_t delta_t*h_(t-1)
∂L/∂b   = Σ_t delta_t
```

三步例子的正确结果：

```text
dL/dw_x =  0.0339152868
dL/dw_h = -0.1033758647
dL/db   = -0.1673551432
```

向量情况下，采用列向量 `delta_t∈R^(H×1)`、`x_t∈R^(C×1)`：

```text
∂L/∂W_x = Σ_t delta_t x_t^T  -> (H,C)
∂L/∂W_h = Σ_t delta_t h_(t-1)^T -> (H,H)
```

`delta_t x_t^T` 是 outer product。例：`delta=[2,-1]^T`、`x=[3,4]^T`，outer product 为 `[[6,8],[-3,-4]]`。

### `mx.grad(loss_fn)(theta)`

`mx.grad(loss_fn)` 先返回一个新的 gradient function；之后 `(theta)` 才是用具体参数调用它。于是：

```python
grad_fn = mx.grad(loss_fn)
grads = grad_fn(theta)
```

与：

```python
grads = mx.grad(loss_fn)(theta)
```

含义相同。

本 scalar demo 中 `theta=mx.array([w_x,w_h,b])`，所以 `theta.shape==(3,)`；MLX 的 `(3,)` 是 1D array，不是显式 `(3,1)` 列矩阵。对应 gradient 也为 `(3,)`，三个分量依次对应 `∂L/∂w_x`、`∂L/∂w_h`、`∂L/∂b`。

### Π、temporal Jacobian product 与矩阵顺序

`Π` 表示乘积。例如 `Π_(k=1)^3 c_k = 0.5*0.8*0.25 = 0.1`。

因为本节统一 final state 为 `h_(T-1)`，scalar long-range sensitivity 写为：

```text
∂h_(T-1)/∂h_t = Π_(k=t+1)^(T-1) ∂h_k/∂h_(k-1)
```

向量 RNN 定义 `J_(k,k-1)=∂h_k/∂h_(k-1)∈R^(H×H)`。forward sensitivity 的矩阵顺序为：

```text
J_(T-1,t) = J_(T-1,T-2) ... J_(t+2,t+1) J_(t+1,t)
```

若 final column gradient 为 `g_(T-1)`，则：

```text
g_t = J_(T-1,t)^T g_(T-1)
    = J_(t+1,t)^T ... J_(T-1,T-2)^T g_(T-1)
```

每一步 shape 都是 `(H,H)(H,1)->(H,1)`。矩阵乘法不可交换，所以顺序不能任意更改。

### 本轮边界

本轮只修正并准备 Step 9.1 母版/文档，不执行 Notebook、不训练、不创建用户工作副本、不进入 Step 9.2，也不据此声明学员已掌握或完成练习。
