# Step 21 — Autoregressive generation

本章五节的教学材料已准备，材料计数：21.3 第一次实质修订 1/2，其余各节初始材料 0/2。只进行了文稿/Notebook JSON 的静态审阅；没有运行代码、实验、训练、验证、采样、checkpoint 或语料操作。材料范围完成不是学习完成、模型性能或运行通过的声明。所有实验都使用显式人工小数组或 toy callback，不依赖旧 kernel、真实语料或权重。

## 阅读顺序与独立工作副本

每节母版开头有独立的 `git pull` 与 `cp -n` 命令参考；不在母版中运行或填写练习。每本 notebook 自带必要 imports、helper 定义与人工输入。MLX 版本固定为 v0.32.2。

### 21.1 Generation loop：把预测接回输入

[独立母版](../notebooks/92_step21_1_generation_loop.ipynb)

训练时模型同时看见许多已知位置的前缀，学习每个位置的下一个 token。生成时未来 token 不存在，只能先预测一个，再把它接回输入。这种用自己刚产生的结果作为下一步条件的方法叫 autoregressive generation（自回归生成）。这里先把“模型预测”和“挑选 token”拆开：callback 提供分数，loop 管理序列。

本节只支持一条非空 prompt、固定生成长度、不使用 EOS、不缓存 KV。这样我们可以看清四个动作：feed context → take last logits → choose → append。提供的 toy callback 是完整且可替换的；它不是已训练 MiniGPT，也不读取任何语料或权重。

**关键数学/接口：**

概率链式法则：p(x₁,…,xₙ)=∏ₜ p(xₜ | x₁,…,xₜ₋₁)。模型输出的 logits 是未归一化分数 z；softmax(z) 才是概率。选择 argmax 时无需先 softmax，因为指数和共同分母不改变最大值位置。

Python 中 `logits[:, -1, :]` 的 -1 表示时间轴最后一项，不是词表最后一项；`[:, None]` 增加一个长度为 1 的维度。`mx.concatenate(..., axis=1)` 沿时间轴连接，不会覆盖 prompt。`range(n)` 恰好迭代 n 次，n=0 时原样返回。

**形状契约：**

第 r 轮之前完整序列长度 L+r：IDs (1,L+r) → callback (1,L+r,V) → 最后一行 (1,V) → argmax (1,) → 增轴 (1,1) → append 后 (1,L+r+1)。本节暂不裁剪，明确要求每次调用模型时输入长度不超过 block_size。最后一个输出 token 可以让返回序列超过容量，因为它还没有被再次送入模型。

**使用边界：**

生成不需要 target、loss、Adam 或梯度更新；`model.eval()` 只切换模式，不是 `mx.eval` 的替代。未来接入真实模型时，先按 Step 20 恢复同架构权重和同一 token 映射，再把模型本身作为 logits_fn；输入必须是合法 token ID，输出必须是原始 (B,T,V) logits。若有 dropout，保存原模式、切换 eval，最后恢复；本 toy 没有模式。

这里每步显式求值方便教学，但不声称这种 Python loop 是最快实现。不得把 A/B/C 的循环当作语言质量证据。

### 21.2 Greedy vs sampling：选最大，还是按概率抽取

[独立母版](../notebooks/93_step21_2_greedy_vs_sampling.ipynb)

相同 logits 可以对应不同解码策略。Greedy 每次取最大分数，容易复现但可能陷入重复；categorical sampling 按概率选 token，可以探索非最大候选，但不会自动提高事实性或质量。我们只研究选择器，不重新训练模型。改变解码规则不等于改变模型参数。

从给定分布抽取一次与从不同分布各抽一次是两个问题。为了隔离影响，本节固定同一行 logits，并使用显式 random key，不依赖 notebook 里其他随机操作的次数。

**关键数学/接口：**

Categorical（类别分布）给离散 token i 分配概率 pᵢ≥0，Σpᵢ=1。softmax: pᵢ=exp(zᵢ−m)/Σⱼexp(zⱼ−m)，m=max(z)。减最大值在精确算术下不改变比值，能避免直接计算过大的指数。

MLX v0.32.2 的 `mx.random.categorical` 接受 logits，不是 probabilities。若把 softmax 后的概率再当 logits，会再次指数化，得到不同分布。对于输入 (B,V)，默认 axis=-1 且不指定 shape/num_samples 时输出 (B,)。

**形状契约：**

固定 logits (1,3) → greedy ID (1,)；categorical 同样 (1,) → 为 append 增轴为 (1,1)。随机 key 用 `mx.random.key(seed)` 创建；`mx.random.split(key)` 返回两个子 key，把一个留下、一个用于此次抽样。相同 key 复用会复用随机性，不是独立抽样。

**使用边界：**

本节输入约定是每行至少一个有限候选、无 NaN/+Inf；后续 top-k 允许其余候选为 -Inf。所有分数都为 -Inf 时没有合法概率分布，应在上游拒绝。

固定 seed 有助于相同实现中的复现，但不保证跨 MLX 版本或硬件逐位一致。Greedy 的稳定也依赖参数、输入、模式和数值路径；并列最大值应明确 tie policy，而不是把它叫作语言模型的“偏好”。此处不把并列情况下具体索引选择作为跨实现契约。

### 21.3 Temperature：只改变相对概率的锐利程度

本节第一次实质修订（1/2）加入未执行的 matplotlib 分组柱状图源码：固定 toy logits，在 tau=0.5、1、2 下比较各 token 的理论概率。母版说明横纵轴、同色概率归一化与柱组偏移；没有运行、绘图结果或抽样频率测量。

[独立母版](../notebooks/94_step21_3_temperature.ipynb)

Temperature（温度）τ 调整选择阶段的 logit scale，不改变模型已经学到的权重。低温把概率集中在高分候选，高温让候选更接近均匀。温度不是创造力开关，也不能修复错误知识；它是对当前分布的可解释变换。

必须把“低温仍抽样”和“greedy”区分。温度非常小但为正时仍定义概率分布；τ=0 则除法未定义。课程把 greedy 作为单独分支，不用除以零近似。

**关键数学/接口：**

pᵢ(τ)=exp(zᵢ/τ)/Σⱼexp(zⱼ/τ)，τ>0。两候选的 odds ratio 为 pᵢ/pⱼ=exp((zᵢ−zⱼ)/τ)。因此降温放大分数差的作用。对所有 logits 加同一个常数不改变概率；对 logits 乘常数等价于调整温度。

当唯一最大值存在，τ→0⁺ 时概率集中到该最大值；并列最大时不能声称极限一定只选一个。τ→∞ 时有限 logits 趋于均匀。以上是数学极限，不是有限精度实现可以任意接近零的承诺。

**形状契约：**

(B,V) logits → 减行最大值 (B,1) broadcast → 除以标量 τ → (B,V) scaled logits → categorical 移除 V 轴，得到 (B,)。本节先显示概率，不进行训练或生成长序列。float32 减最大值是数值防护；仍需拒绝不有限温度和无法表示的缩放。

**使用边界：**

温度不会改变正温度下的分数排序，故通常不改变 greedy 的选择。高温可能提高低质量 token 的概率；低温仍可能重复。比较策略时应固定 prompt、参数、长度、其他过滤设置，并区分理论概率与单次输出。

未来真实模型接入点是 logits[:, -1, :] 之后、categorical 之前。不要在训练交叉熵中悄悄加入生成温度，也不要改 checkpoint。这里的参数拒绝策略是教学接口契约，不是 MLX 自带参数检查。

### 21.4 Top-k：限制候选集合后再采样

[独立母版](../notebooks/95_step21_4_top_k_sampling.ipynb)

完整词表里可能有大量很低分候选。Top-k 先保留最高分的 k 个候选，再在这个集合内抽样。它改变支持集（support），不是只让分布变尖。Temperature 改变候选之间的比例；top-k 直接使部分 token 不再可能出现。

本节明确采用“严格 k 个索引”的定义。另一种常见阈值法保留所有大于等于第 k 大分数的项，会因并列值留下超过 k 个候选。两种策略都能合理，但不能混写定义和实现。

**关键数学/接口：**

设 S 为保留索引集合，z'ᵢ=zᵢ（i∈S），否则为 −∞。因为 exp(−∞)=0，softmax(z') 的分母只包含 S：p'ᵢ=exp(zᵢ)/Σⱼ∈S exp(zⱼ)。categorical 接受这些 masked logits，内部按其定义采样，不需要先显式 softmax。

为清晰定义并列规则，本节用 Python 排序：先按分数降序，同分按 token ID 升序，再取前 k 个。这需要把小数组传到 host，适合四词玩具，不是高吞吐实现。此取舍比用不确定 tie 行为的排序假装严格可复现更透明。

**形状契约：**

单条 logits (V,) → host list 长 V → 排序得到 k 个原始 token ID → Boolean mask (V,) → masked logits (V,) → categorical 标量 ID。排序索引不是新的词表；输出必须仍指向原词表。接入批量 (B,V) 时需要逐行定义，本节不冒充已支持 batch。

**使用边界：**

正温度缩放不改变排序，所以在精确算术下 top-k 集合不受温度影响；数值误差、并列规则仍需说明。若组合两个助手，先对有限原始 logits 做 temperature scaling，再交给本节 top-k，不能把 -Inf 喂给 21.3 的有限输入检查。

k 必须至少 1，否则所有位置被屏蔽，分布不存在。把被排除分数写成 0 不是屏蔽，因为 exp(0)=1。Top-k 不是 top-p/nucleus sampling；后者根据累计概率确定可变大小集合，不在本节实现。

### 21.5 Context cropping：保留输出历史，限制模型窗口

[独立母版](../notebooks/96_step21_5_context_cropping.ipynb)

生成长度可以大于训练时的 block_size，但一次 forward 不能越过学到的位置表容量。最简单的处理是保留完整输出 history，每轮只把其最后 W 个 token 送入模型。裁剪的是模型上下文，不是用户最后看到的生成文本。

这不是长上下文能力升级。被裁掉的内容不再影响下一步预测；固定窗口里的 learned absolute positions 也会重新从零编号。原模型并不因此拥有超出训练长度的位置理解。

**关键数学/接口：**

定义完整长度 Lᵣ=L₀+r，实际输入长度 Tᵣ=min(Lᵣ,W)。Python `ids[:, -W:]` 从末尾取最多 W 项；若当前长度小于 W，不会补齐，不会产生负长度。最后分布为 p(xₙ₊₁ | xₘ,…,xₙ)，m=max(1,n−W+1)，是有限上下文条件分布。

无 KV cache 时，每轮都重新计算整个窗口的 embedding、投影、attention 和 MLP。Attention score 主项每轮约 O(W²d) 每头；N 个 decode 步约 O(NW²d)，忽略其他项与增长阶段。这是算术规模估算，不是运行速度测量。

**形状契约：**

history (1,Lᵣ) → context (1,min(Lᵣ,W)) → logits (1,min(Lᵣ,W),V) → last (1,V) → next (1,1) → history (1,Lᵣ+1)。返回完整 history 与每次 context 长度记录，用形状证据分离“输出长度”和“模型容量”。本例仅 greedy，无 EOS/batch/cache。

**使用边界：**

KV cache（键值缓存）为每层保存已经算过的 K、V，decode 只算新 token 的 Q/K/V，再让新 Q 关注缓存。缓存张量通常为 (B,Hkv,T,d)，每层 K/V 两份。它减少重复计算，同时增加驻留内存，不是免费优化。

不能给当前 learned-position 窗口简单拼一个 cache 就宣称等价：窗口滑动时旧 token 的绝对位置改变，深层 hidden states 的可见前缀也改变，旧 K/V 可能已经不对应“从头重算窗口”的定义。RoPE offset、窗口策略、cache 生命周期、prefill/decode mask 必须共同设计。本节只预告，不实现 KV cache、不修改 MiniGPT。

真实模型需要 W≤max_seq_len，并核对训练 block_size 与位置表容量。Toy 只依赖当前 token，不能用它证明真实 Transformer 裁剪前后输出等价。

## 静态检查与后续

每节都有动机、先备数学、纸笔例子、shape tracing、独立源码、未执行短实验及无答案练习。短实验中的 print 仅是未来观察入口，本次没有输出。允许的静态结构检查不等同于运行测试。

接入未来训练模型前，必须恢复同一 tokenizer 映射与架构，检查 vocab_size/max_seq_len，确保 callback 返回原始 (B,T,V) logits。当前例子仅 B=1、固定长度、无 EOS、无 padding、无 KV cache；批量结束条件与缓存应独立设计。Step 22 展开现代架构概念，不要求变更当前 MiniGPT。

## 固定版本官方来源

- https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/ops.cpp
- https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/random.cpp
