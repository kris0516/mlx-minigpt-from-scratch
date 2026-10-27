# Step 09 — BPTT, Vanishing Gradients, LSTM, and GRU

目标：理解 Vanilla RNN 的长期 credit assignment 为什么困难，以及 gating 为什么被引入。

本 Step 保留既有 9.1–9.5 课纲。Step 9.1、9.2、9.3 材料已通过静态审核；当前只准备 **09.4**，不进入 09.5：

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

### Step 9.1 材料状态

Step 9.1 母版已通过静态审核；这只表示材料准备状态，不代表学员已经掌握、完成练习或通过验收。

## Step 09.2 — Vanishing and Exploding Gradients

### Repeated scalar products

若一条长度 n 的教学链每一步都乘相同 scalar factor c，则 long-range sensitivity 为 `c^n`。`0.5^n` 快速衰减，`1.5^n` 快速增长；它们只建立 repeated-product 直觉，不是一般 vector RNN 的完整判据。

从 `h_t` 到 `h_(T-1)` 跨过的 recurrent transition 数是 `(T-1)-t`。

### Tanh saturation and state-dependent local factors

Scalar tanh RNN 的 local recurrent derivative 为 `w_h*(1-h_k^2)`。因此传播强度同时依赖 recurrent weight 与当前 hidden state。`a≈0` 时 tanh derivative 接近 1；`a=1` 时约 0.419974；`a=2` 时约 0.070651。进入 saturation 区会显著压低局部 sensitivity。

即使 `|w_h|>1`，若 tanh 已饱和，local factor 仍可能小于 1；不能只检查 scalar `w_h` 就判断长期 gradient 行为。

### Vector Jacobians and direction dependence

必须先区分 **future-path contribution** 与 **total hidden-state gradient**。

令 `J_(k,k-1)=∂h_k/∂h_(k-1)`。如果 loss 只在末端定义，或者这里只追踪未来 loss 沿 recurrent edge 回传的一条路径贡献，则：

```text
g_(k-1)^future = J_(k,k-1)^T g_k
```

其中 future term 表示 `h_(k-1)` 先影响 `h_k`，再影响之后的 losses。

如果 `L=Σ_t ℓ_t` 且每一步都有 local loss，则 total hidden-state gradient 应写成：

```text
g_(k-1) = ∇_(h_(k-1)) ℓ_(k-1) + J_(k,k-1)^T g_k
```

第一项是 **local term**：`h_(k-1)` 对当前时间位置 loss 的直接贡献；第二项是 **future term**：经 recurrent chain 从未来传回的贡献。后面的 vanishing/exploding Jacobian 连乘主要分析第二项的 long-range propagation，不能把这条路径贡献自动当成有 local losses 时的总梯度。

对 column-vector Vanilla RNN：`J_(k,k-1)=D_k W_h`，其中 `D_k=diag(1-h_k^2)` 是把每个 hidden coordinate 的 tanh derivative 放在对角线上的 `(H,H)` matrix。因此 future-path contribution 为 `g_(k-1)^future=W_h^T D_k g_k`；若 local loss 存在，再加上 local term。

同一个 Jacobian 可以放大某些 gradient directions、同时缩小另一些 directions。例如 `diag(1.5,0.5)` 对 `[1,0]^T` 产生 `1.5^n` 型增长，对 `[0,1]^T` 产生 `0.5^n` 型衰减。真实 RNN 的 `D_k` 随 state 变化，Jacobian 也随时间变化且矩阵顺序不可交换，因此不能把 scalar toy conclusion 直接套到所有矩阵情形。

在材料第一次使用 `l2_norm` 比较上述方向之前，先定义 **L2 norm（二范数 / Euclidean norm）**：

```text
||g||_2 = sqrt(Σ_i g_i^2)
```

它把一个 gradient vector 的大小变成非负 scalar。例如 `g=[3,4]^T` 时，`||g||_2=sqrt(3^2+4^2)=5`。

### Gradient norm and clipping

在单个向量的 L2 norm 已经定义之后，再扩展到 global gradient norm：对整个 gradient tree 的所有元素一起平方求和再开根号。

Global-norm clipping 在 gradient 太大时用同一比例缩小整个 gradient tree。例如 `[6,8]^T` 的 norm 为 10，若 `max_norm=5`，理想化 scale 为 0.5，得到 `[3,4]^T`。

项目固定 `mlx==0.32.2`。该版本提供 `mlx.optimizers.clip_grad_norm(grads, max_norm)`，返回 `(clipped_grads, total_norm)`；`total_norm` 是 clipping 前计算的 global norm。实现使用 `min(max_norm/(total_norm+1e-6),1)` 作为统一缩放因子。

### What clipping can and cannot solve

Gradient clipping 可以缓解 exploding gradient 导致的极端 update magnitude 与训练震荡风险，但不能恢复已经 vanishing 的 long-range signal，不能消除 tanh saturation，不能缩短 recurrent dependency path，也不能保证长期记忆或训练收敛。

Vanishing 描述的常常是远距离 **future-path contribution** 变弱，并不意味着 total hidden-state gradient 或整个 parameter gradient tensor 必然接近零。若 `L=Σ_tℓ_t`，hidden-state total gradient 还包含 local term `∇_(h_(k-1))ℓ_(k-1)`，parameter gradient 也可能包含 local-loss 和多个时间路径的求和贡献。

Step 9.2 只建立这些边界。材料准备到此停止，不进入 Step 9.3。

## Step 09.3 — LSTM Cell State and Gates

### Why a second recurrent state

LSTM 在 `h_t` 之外引入 cell state `c_t`。二者通常都是 `(B,H)`；`c_t` 提供更直接的内部 state-update 路径，`h_t` 是经过 output gate 过滤后的 hidden output。不要把它们机械等同为“长期记忆/短期记忆”。

### Sigmoid gates and element-wise control

`σ(z)=1/(1+e^-z)` 将实数压到 `(0,1)`，因此 gate 是 soft coefficient，不是 binary switch。`⊙` 表示 element-wise/Hadamard multiplication。

标准教学记号：

```text
i_t = sigmoid(W_xi x_t + W_hi h_(t-1) + b_i)
f_t = sigmoid(W_xf x_t + W_hf h_(t-1) + b_f)
g_t = tanh   (W_xg x_t + W_hg h_(t-1) + b_g)
o_t = sigmoid(W_xo x_t + W_ho h_(t-1) + b_o)

c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

`f/i/o` 是 gates；`g_t` 是 tanh candidate。

### Shapes and packed 4H projection

`x_t=(B,C)`，`h_(t-1)=c_(t-1)=(B,H)`。四组 gate/candidate 输出都是 `(B,H)`。工程实现常把四组 projections 打包成 `(B,4H)`，再 split 成 `i/f/g/o`；对应 packed weights 为 `(4H,C)` 与 `(4H,H)`。

### Tiny numeric update

若 `c_(t-1)=0.80`、`f_t=0.90`、`i_t=0.20`、`g_t=0.50`、`o_t=0.70`，则 retained old state 为 `0.72`，new write 为 `0.10`，所以 `c_t=0.82`，`h_t=0.7*tanh(0.82)≈0.4725`。

### Direct cell-state gradient path

只看 `c_(t-1) -> c_t` 的 direct edge，并在该局部 partial derivative 中把其他输入/gate values 视为给定，则 direct Jacobian contribution 为 `diag(f_t)`；相应 backward contribution 为 `f_t ⊙ g_(c_t)`。`f_t≈1` 可以给某些 coordinates 提供近似 identity 的保留路径，`f_t≈0` 则主动遗忘。

这只是 direct path contribution，不是整个 LSTM total Jacobian。Gates 依赖 `x_t/h_(t-1)`，完整 BPTT 还有其他路径；LSTM 改善长期 credit assignment，但不保证彻底消除 vanishing/exploding，也不是无限无损 memory。

### MLX 0.32.2 mapping

项目固定 `mlx==0.32.2`。该版本提供 `nn.LSTM(input_size, hidden_size, bias=True)`；batched input `(B,T,C)` 可配 initial `hidden/cell=(B,H)`，返回 `all_hidden, all_cell`，二者均为 `(B,T,H)`。

MLX v0.32.2 内部 packed parameters 为 `Wx=(4H,C)`、`Wh=(4H,H)`、`bias=(4H,)`（bias=True），并按 `i,f,g,o` 顺序 split。

Step 9.3 材料准备到此停止，不进入 Step 9.4。

## Step 09.4 — GRU

### Why GRU simplifies gated recurrence

GRU（Gated Recurrent Unit）只维护一个 recurrent hidden state `h_t`，不再维护独立的 LSTM cell state `c_t`。它用 reset gate `r_t`、update gate `z_t` 与 tanh candidate `n_t` 实现门控更新。

GRU 不是必然优于 LSTM 的替代品；它是另一种更紧凑的 gated recurrent architecture。

### Equations and gate semantics

本课程采用与 MLX 0.32.2 一致的约定：

```text
r_t = sigmoid(W_xr x_t + W_hr h_(t-1) + b_r)
z_t = sigmoid(W_xz x_t + W_hz h_(t-1) + b_z)
n_t = tanh(W_xn x_t + b_n + r_t ⊙ (W_hn h_(t-1) + b_hn))
h_t = (1-z_t) ⊙ n_t + z_t ⊙ h_(t-1)
```

`r_t` 主要控制上一 hidden projection 参与 candidate 构造的程度；`z_t` 控制最终 state 在旧 `h_(t-1)` 与新 candidate `n_t` 之间的逐元素加权插值。按这一约定，`z_t≈1` 更保留旧 state，`z_t≈0` 更采用 candidate。不同资料可能采用相反的 update-gate 表达，应以具体公式为准。

### Tiny numeric intuition

若 `h_(t-1)=0.8`、`n_t=0.2`，则 `z_t=0.9` 时 `h_t=0.74`，主要保留旧 state；`z_t=0.1` 时 `h_t=0.26`，主要采用 candidate。

Reset gate 不直接决定最终旧 state 保留多少，而是调节 candidate recurrent contribution。例如固定 input term 0.1、recurrent proposal 0.6 时，`r=0.2` 得 candidate preactivation 0.22，`r=0.9` 得 0.64。

### Shapes and direct retain path

`x_t=(B,C)`、`h_(t-1)=(B,H)`；`r_t/z_t/n_t` 都是 `(B,H)`。工程实现将三组 projection 打包为 `Wx=(3H,C)`、`Wh=(3H,H)`，因此 GRU 使用 `3H` packed width，而 LSTM 使用 `4H`。

只看 `h_(t-1) -> h_t` 中直接乘 `z_t` 的 retain edge，并在该局部 partial derivative 中把当前 gate/candidate 当作给定，则 direct Jacobian contribution 为 `diag(z_t)`，direct backward contribution 为 `z_t ⊙ g_(h_t)`。这只是 direct path contribution，不是完整 GRU total Jacobian；`r/z/n` 都依赖上一 hidden state。

### MLX 0.32.2 mapping and LSTM comparison

项目固定 `mlx==0.32.2`。该版本提供 `nn.GRU(input_size, hidden_size, bias=True)`；batched input `(B,T,C)` 可配 initial hidden `(B,H)`，返回 hidden-state sequence `(B,T,H)`。

MLX v0.32.2 packed parameters 为：`Wx=(3H,C)`、`Wh=(3H,H)`、`b=(3H,)` 与 candidate recurrent bias `bhn=(H,)`（bias=True）。

在相同 C/H、bias=True 的 MLX 0.32.2 实现中，GRU 参数量为 `3HC+3H^2+4H`，LSTM 为 `4HC+4H^2+4H`。例如 `C=3,H=4` 时分别为 100 与 128。参数更少不代表效果必然更好。

Step 9.4 材料准备到此停止，不进入 Step 9.5。
