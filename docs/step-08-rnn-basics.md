# Step 08 — RNN: The First Explicit Neural Sequence Memory

## 目标

在 MLP 之后，理解序列顺序、可变长度、历史记忆和时间参数共享为什么推动了 recurrent neural networks 的发展。

本 Step 保持五个小节：

1. **08.1 Why ordinary MLPs are awkward for variable-length sequences**
2. **08.2 Vanilla RNN equation**
3. **08.3 Time loop implementation**
4. **08.4 Small MLX RNN cell**
5. **08.5 What RNN solves and what it costs**

## Step 08.1 — Why ordinary MLPs are awkward for sequences

### 序列任务的三个额外需求

自然语言预测依赖顺序、历史上下文，以及不同长度输入。模型需要在相同计算规则下处理接续到来的 token。

### 不要混淆两种 MLP 用法

Position-wise MLP：`(B,T,C) -> (B,T,C)`，每个位置独立做 C-axis feature mixing；即使有 position embedding，它自身也没有跨位置的信息边。

Flatten + Dense：`(B,T,C) -> (B,T*C) -> (B,H)`，确实能够混合不同位置，但第一层的输入 width 与 T 绑定。单个无 bias Linear 的权重数为 `T*C*H`，固定 C,H 时随 T 增长。

MLP **并非绝对无法处理序列**：固定窗口、padding/masking、temporal MLP 等仍可行。这里只是在说明它们不是天然的 streaming/recurrent 历史建模。

### Recurrent hidden state 的核心动机

`h_t = f_θ(x_t, h_(t-1))`。同一套参数 θ 在所有时间步复用，维护固定大小的压缩历史摘要，不需要因为 T 改变而重新定义输入权重。

简单数值直觉：`h_t = x_t + 0.5*h_(t-1)`，顺序 `[1,0,2]` 的最终状态 2.25，反序 `[2,0,1]` 为 1.5；输入多一个值可以沿原规则继续处理。这不是完整可训练 RNN。

### Tensor shapes 与历史路线

输入 sequence `(B,T,C)`，当前 token `x_t=(B,C)`；维护 `h_t=(B,H)`，连续收集隐藏状态可得到 `(B,T,H)`。RNN 的同一序列时间步存在串行依赖。

Hidden state 不能保证无限精确记忆，长期依赖、BPTT 和 vanishing/exploding gradients 后面 Step 9 继续；完整 RNN 方程留在 8.2。

### 本讲绘图

包含参数量随 T 变化的对照折线图（不作模型性能等价比较）和 Position-wise MLP vs Recurrent-state 计算依赖图，顺手介绍 `ax.plot`、`ax.legend`、`ax.annotate`。

**下一节 8.2**：`h_t = tanh(W_x x_t + W_h h_(t-1) + b)`，逐项解释权重 shape 和手算。


## Step 08.2 — Vanilla RNN Equation

### Two learnable paths plus a nonlinearity

Vanilla RNN updates its historical state by `a_t = W_x x_t + W_h h_(t-1) + b`, then `h_t = tanh(a_t)`.

`W_x` maps the new C-dimensional input into H-dimensional hidden-state space; `W_h` propagates/transforms the previous H-dimensional historical state. The hidden state is not automatically the token-probability output.

### Shapes

Column-vector notation: `x_t=(C,1)`, `h_(t-1)=(H,1)`, `W_x=(H,C)`, `W_h=(H,H)`, `b=(H,1)`, `h_t=(H,1)`.

Apple MLX row/batch convention: `x_t=(B,C)`, `h_prev=(B,H)` and `h_t = mx.tanh(x_t @ W_x.T + h_prev @ W_h.T + b)` -> `(B,H)`. `H` need not equal C.

### Two-time-step worked example

For `W_x=I_2`, `W_h=0.5 I_2`, `b=0`, initial state `h_(-1)=0`, and `x0=[1,0]`, `x1=[0,1]`, hand calculation yields `h0≈[0.7616,0]` and `h1≈[0.3634,0.7616]`.

The first coordinate of h1 is nonzero even though the current x1 first coordinate is zero: it carries an effect of x0 through h0.

### Why tanh

`tanh` provides a bounded, sign-sensitive nonlinearity (output in `(-1,1)`), but its saturation may contribute to long-horizon gradient problems. Bounded outputs do not guarantee stable training or lossless long-term memory.

### Time-shared weights and next steps

One cell has `H*C + H*H + H` trainable scalar parameters, independent of the number of processed time steps T because parameters are shared, although runtime computation and BPTT costs do depend on T.

Step 8.3 will implement the full time loop and `(B,T,C)` to `(B,T,H)` state collection. Step 8.4 will wrap the complete RNN cell as an Apple MLX `nn.Module`; Step 9 covers BPTT and long-range gradient limitations.
