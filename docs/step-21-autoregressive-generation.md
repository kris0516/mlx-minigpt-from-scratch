# Step 21 — Course materials

当前此 Step 已发布至 21.2。仅静态材料，不是运行结果。

# Step 21.1 — Generation loop：把预测接回输入

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/92_step21_1_generation_loop.ipynb notebooks/work/92_step21_1_generation_loop_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版是**材料初稿 0/2**，不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

训练时模型同时看见许多已知位置的前缀，学习每个位置的下一个 token。生成时未来 token 不存在，只能先预测一个，再把它接回输入。这种用自己刚产生的结果作为下一步条件的方法叫 autoregressive generation（自回归生成）。这里先把“模型预测”和“挑选 token”拆开：callback 提供分数，loop 管理序列。

本节只支持一条非空 prompt、固定生成长度、不使用 EOS、不缓存 KV。这样我们可以看清四个动作：feed context → take last logits → choose → append。提供的 toy callback 是完整且可替换的；它不是已训练 MiniGPT，也不读取任何语料或权重。

## 2. 先备数学与 Python / MLX 语义

概率链式法则：p(x₁,…,xₙ)=∏ₜ p(xₜ | x₁,…,xₜ₋₁)。模型输出的 logits 是未归一化分数 z；softmax(z) 才是概率。选择 argmax 时无需先 softmax，因为指数和共同分母不改变最大值位置。

Python 中 `logits[:, -1, :]` 的 -1 表示时间轴最后一项，不是词表最后一项；`[:, None]` 增加一个长度为 1 的维度。`mx.concatenate(..., axis=1)` 沿时间轴连接，不会覆盖 prompt。`range(n)` 恰好迭代 n 次，n=0 时原样返回。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

纸笔例子：词表 [A,B,C,_]，prompt=[A]。人工转移表 A 行为 [0,3,1,-1]，最大分数位于 B，所以第一轮变为 [A,B]。下一轮取 B 行 [0,0,3,-1]，得到 [A,B,C]。第三轮才会使用 C 行。这个推导来自给定表格，不是程序实测。

若把全部位置各挑一个 token，会把已有前缀的位置也重新预测一遍；生成循环需要的是最后位置对“前缀之后那个 token”的预测。

## 4. Shape tracing 与接口契约

第 r 轮之前完整序列长度 L+r：IDs (1,L+r) → callback (1,L+r,V) → 最后一行 (1,V) → argmax (1,) → 增轴 (1,1) → append 后 (1,L+r+1)。本节暂不裁剪，明确要求每次调用模型时输入长度不超过 block_size。最后一个输出 token 可以让返回序列超过容量，因为它还没有被再次送入模型。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 6. 常见误区与未来衔接

生成不需要 target、loss、Adam 或梯度更新；`model.eval()` 只切换模式，不是 `mx.eval` 的替代。未来接入真实模型时，先按 Step 20 恢复同架构权重和同一 token 映射，再把模型本身作为 logits_fn；输入必须是合法 token ID，输出必须是原始 (B,T,V) logits。若有 dropout，保存原模式、切换 eval，最后恢复；本 toy 没有模式。

这里每步显式求值方便教学，但不声称这种 Python loop 是最快实现。不得把 A/B/C 的循环当作语言质量证据。

## 7. 练习（不附答案）

1. 手写 prompt=[B,C] 的前两轮输入、最后 logits 与输出。
2. 解释为什么不能把 logits[:, :, -1] 当成 next-token 分数。
3. 为 n=0、空 prompt、非法 ID、容量不足分别设计接口行为检查，不写训练循环。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/src/ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/ops.cpp)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

# Step 21.2 — Greedy vs sampling：选最大，还是按概率抽取

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/93_step21_2_greedy_vs_sampling.ipynb notebooks/work/93_step21_2_greedy_vs_sampling_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版是**材料初稿 0/2**，不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

相同 logits 可以对应不同解码策略。Greedy 每次取最大分数，容易复现但可能陷入重复；categorical sampling 按概率选 token，可以探索非最大候选，但不会自动提高事实性或质量。我们只研究选择器，不重新训练模型。改变解码规则不等于改变模型参数。

从给定分布抽取一次与从不同分布各抽一次是两个问题。为了隔离影响，本节固定同一行 logits，并使用显式 random key，不依赖 notebook 里其他随机操作的次数。

## 2. 先备数学与 Python / MLX 语义

Categorical（类别分布）给离散 token i 分配概率 pᵢ≥0，Σpᵢ=1。softmax: pᵢ=exp(zᵢ−m)/Σⱼexp(zⱼ−m)，m=max(z)。减最大值在精确算术下不改变比值，能避免直接计算过大的指数。

MLX v0.32.2 的 `mx.random.categorical` 接受 logits，不是 probabilities。若把 softmax 后的概率再当 logits，会再次指数化，得到不同分布。对于输入 (B,V)，默认 axis=-1 且不指定 shape/num_samples 时输出 (B,)。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

用 z=[ln2,ln3,ln5]，指数比例为 2:3:5，因此 p=[0.2,0.3,0.5]。Greedy 选择 ID 2；sampling 有 0.2 的概率选择 ID 0。想象 [0,1) 被切成 [0,.2)、[.2,.5)、[.5,1)，均匀抽一个数，再找它落在哪段。这解释概率语义，不是 MLX 内部算法实现说明。

连续两次都抽到 ID 2 不证明 sampling 变成了 greedy；有限样本不等于精确概率。

## 4. Shape tracing 与接口契约

固定 logits (1,3) → greedy ID (1,)；categorical 同样 (1,) → 为 append 增轴为 (1,1)。随机 key 用 `mx.random.key(seed)` 创建；`mx.random.split(key)` 返回两个子 key，把一个留下、一个用于此次抽样。相同 key 复用会复用随机性，不是独立抽样。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 6. 常见误区与未来衔接

本节输入约定是每行至少一个有限候选、无 NaN/+Inf；后续 top-k 允许其余候选为 -Inf。所有分数都为 -Inf 时没有合法概率分布，应在上游拒绝。

固定 seed 有助于相同实现中的复现，但不保证跨 MLX 版本或硬件逐位一致。Greedy 的稳定也依赖参数、输入、模式和数值路径；并列最大值应明确 tie policy，而不是把它叫作语言模型的“偏好”。此处不把并列情况下具体索引选择作为跨实现契约。

## 7. 练习（不附答案）

1. 对 z=[0,0,ln2] 计算概率并比较 greedy 与 sampling 的可能输出。
2. 解释为何重复八次结果不能验证概率恰好等于理论值。
3. 如果希望生成两条独立序列，应该如何管理各自 key？画出 key 流程，不给训练代码。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/src/random.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/random.cpp)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

