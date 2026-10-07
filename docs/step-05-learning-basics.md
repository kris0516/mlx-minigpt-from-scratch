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


## Step 05.3 — Gradient and Automatic Differentiation

### From derivative to gradient

For multiple parameters:

```text
w = [w0, w1, ...]
```

the gradient is:

```text
[dL/dw0, dL/dw1, ...]
```

and follows the parameter structure.

Example:

```text
prediction = w0*x0 + w1*x1
loss = (prediction-target)^2
```

At:

```text
w=[3,1]
x=[2,-1]
target=10
```

the manual gradient is:

```text
[-20, 10]
```

### Reverse-mode intuition

The forward computation forms a graph from parameters to scalar loss.

Reverse-mode automatic differentiation traverses this computation in reverse, using local derivatives and the chain rule to propagate loss sensitivity back to earlier parameters.

It is not finite-difference perturbation.

### MLX

```python
grad_fn = mx.grad(loss_fn)
gradient = grad_fn(w)
```

`mx.grad` transforms a scalar-valued function into a gradient-producing function.

```python
loss_and_grad_fn = mx.value_and_grad(loss_fn)
loss, gradient = loss_and_grad_fn(w)
```

returns both the scalar function value and its gradient.

The module-aware `nn.value_and_grad` used later for full model parameter trees is intentionally deferred to the language-model training step.


## Step 05.4 — Gradient Descent Loop

Core update:

```text
parameter_new = parameter_old - learning_rate * gradient
```

The gradient points toward locally increasing loss, so gradient descent follows the negative gradient.

For the teaching model:

```text
prediction = w*x
x=2
target=10
loss=(w*x-target)^2
```

at `w=3`:

```text
loss=16
gradient=-16
```

with `lr=0.1`:

```text
w_new = 3 - 0.1*(-16) = 4.6
new_loss = 0.64
```

Repeated updates form the minimal training loop:

```text
parameters
-> forward
-> loss
-> gradient
-> update
-> repeat
```

### Learning rate

The learning rate controls step size, not direction.

Too small can be slow; too large can overshoot, oscillate, or diverge.

### Terminology

The current fixed tiny example is plain gradient descent. Full-batch GD, stochastic gradient descent, and mini-batch SGD differ in how much training data is used to estimate each gradient.

Adam still uses gradients; it changes how parameter updates are scaled using optimizer state.
