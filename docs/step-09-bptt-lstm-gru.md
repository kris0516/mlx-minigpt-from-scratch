# Step 09 — BPTT, Vanishing Gradients, LSTM, and GRU

目标：理解 Vanilla RNN 的长期 credit assignment 为什么困难，以及 gating 为什么被引入。

本 Step 共 5 个小节：

1. **09.1 Backpropagation Through Time**
2. **09.2 Vanishing and Exploding Gradients**
3. **09.3 LSTM Cell State and Gates**
4. **09.4 GRU**
5. **09.5 Why LSTM/GRU Still Did Not Solve Everything**

## Step 09.1 — Backpropagation Through Time

BPTT is ordinary reverse-mode backpropagation applied to an RNN graph unrolled across time. Multiple time nodes reuse the same model parameters.

For `L=sum_t l_t`, an intermediate state can receive both a local loss contribution and a future recurrent contribution. For scalar Vanilla RNN `a_t=w_x*x_t+w_h*h_(t-1)+b`, `h_t=tanh(a_t)`, the local recurrent derivative is `dh_t/dh_(t-1)=w_h*(1-h_t^2)`.

Using `w_x=0.7`, `w_h=0.5`, `b=0.1`, `x=[1,0.5,-0.5]`, final-loss BPTT gives deltas approximately `[-0.00980,-0.03506,-0.12249]`.

Shared parameters accumulate path contributions across all time uses: `dL/dw_x=sum_t delta_t*x_t`, `dL/dw_h=sum_t delta_t*h_(t-1)`, `dL/db=sum_t delta_t`. Apple MLX `mx.grad` verifies the hand calculation.

Long-range influence contains products such as `dh_T/dh_t = product_k dh_k/dh_(k-1)`. Step 9.2 studies the numerical consequences rather than assuming such products always shrink or always grow.
