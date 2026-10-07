# Step 05 — How a Neural Network Learns

目标：从“模型会 forward”推进到“模型能根据错误修改参数”。

本 Step 共 5 个小节：

1. **05.1 Prediction, Target, Loss**
2. **05.2 Minimal Calculus for Learning**
3. **05.3 Gradient and Automatic Differentiation**
4. **05.4 Gradient Descent Loop**
5. **05.5 What Learning Really Means**

## Step 05.1 — Prediction, Target, Loss

训练首先需要：

```text
prediction -> 模型输出
target     -> 正确答案
loss       -> prediction 与 target 差距的数值度量
```

教学阶段先使用 squared error：

```text
loss = (prediction - target)^2
```

多个样本时，先得到 per-sample loss，再做 mean reduction，得到 scalar loss。

最终训练常使用 scalar objective，因为下一步要研究每个 parameter 对整体目标的影响。

MiniGPT 最终会使用 Cross Entropy Loss；这里的 squared error 只是教学入口。


## Step 05.2 — Minimal Calculus for Learning

### Derivative as local sensitivity

For a scalar function such as:

```text
f(x)=x^2
```

the derivative at a point describes how quickly the output changes when the input changes slightly.

Finite difference approximation:

```text
(f(x+h)-f(x))/h
```

approaches the derivative as `h` becomes small.

### Minimal derivative rules

```text
d/dx(constant)=0
d/dx(x)=1
d/dx(x^2)=2x
d/dx(a*x+b)=a
```

### Training example

For:

```text
prediction = w*x
loss = (prediction-target)^2
```

define:

```text
e = wx-target
L = e^2
```

Then by the chain rule:

```text
dL/dw = (dL/de)(de/dw)
      = 2e*x
```

At `w=3, x=2, target=10`:

```text
e=-4
dL/dw=-16
```

The negative derivative means increasing `w` locally decreases the loss.

This scalar chain-rule view is the mathematical foundation for later backpropagation and automatic differentiation.
