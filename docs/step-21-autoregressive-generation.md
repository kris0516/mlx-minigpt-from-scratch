# Step 21 — Course materials

当前此 Step 已发布至 21.4。仅静态材料，不是运行结果。

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

# Step 21.3 — Temperature：只改变相对概率的锐利程度

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/94_step21_3_temperature.ipynb notebooks/work/94_step21_3_temperature_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版已作**第一次实质修订 1/2**：增加温度分布分组柱状图教学。此计数不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

Temperature（温度）τ 调整选择阶段的 logit scale，不改变模型已经学到的权重。低温把概率集中在高分候选，高温让候选更接近均匀。温度不是创造力开关，也不能修复错误知识；它是对当前分布的可解释变换。

必须把“低温仍抽样”和“greedy”区分。温度非常小但为正时仍定义概率分布；τ=0 则除法未定义。课程把 greedy 作为单独分支，不用除以零近似。

## 2. 先备数学与 Python / MLX 语义

pᵢ(τ)=exp(zᵢ/τ)/Σⱼexp(zⱼ/τ)，τ>0。两候选的 odds ratio 为 pᵢ/pⱼ=exp((zᵢ−zⱼ)/τ)。因此降温放大分数差的作用。对所有 logits 加同一个常数不改变概率；对 logits 乘常数等价于调整温度。

当唯一最大值存在，τ→0⁺ 时概率集中到该最大值；并列最大时不能声称极限一定只选一个。τ→∞ 时有限 logits 趋于均匀。以上是数学极限，不是有限精度实现可以任意接近零的承诺。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

z=[0,ln2]。τ=1 时指数权重 1:2，概率 [1/3,2/3]；τ=0.5 时权重 1:4，概率 [1/5,4/5]；τ=2 时权重 1:√2，第二项概率 √2/(1+√2)。最大值位置没变，但 sampling 的选择频率变了。

注意先将 logits 除以 τ，再计算 softmax；“先 softmax 再把概率除以 τ”通常连概率和为 1 都不满足。

## 4. Shape tracing 与接口契约

(B,V) logits → 减行最大值 (B,1) broadcast → 除以标量 τ → (B,V) scaled logits → categorical 移除 V 轴，得到 (B,)。本节先显示概率，不进行训练或生成长序列。float32 减最大值是数值防护；仍需拒绝不有限温度和无法表示的缩放。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 5.1 用 grouped bar plot 比较温度分布（仅源码，未绘制）

同一份人工 logits `z=[0, ln(2)]` 在三种温度下给出三个 categorical distributions。横轴是原始 token ID（0、1），纵轴是该 token 的理论概率，范围 0 到 1；颜色表示 temperature。每一种颜色对应的两个柱高相加应为 1，而不是要求每个 token 位置的三种颜色相加为 1。

先按公式 `softmax(z / tau)` 计算各分布，再画图。这里复用上方 `temperature_logits` 的稳定化与参数检查，`mx.eval` 之后才转换为 Python list 交给 matplotlib。它们是对给定分数的理论分布作数值计算，不是 sampling 频率、不是真实训练模型的预测质量，也不是已取得的运行测量。当前并未执行绘图，因此没有图形输出或实测结论。

Grouped bars（分组柱）让同一个 token 的三种温度并排，避免相互遮挡。`width=0.24` 是每根柱子的宽度；第 i 组使用 `(i-1)*width` 偏移，使 i=0、1、2 分别位于中心左侧、中心和右侧。`centers=[0,1]` 保留 token 的刻度位置，偏移只改变展示坐标，不改变 token ID 或概率。

阅读时先比较 token 1 在降温和升温后的柱高，再检查同色两柱之和。图可以帮助观察趋势，但不能代替前面的 odds ratio 推导；改变横轴偏移或颜色不会改变概率模型。

## 6. 常见误区与未来衔接

温度不会改变正温度下的分数排序，故通常不改变 greedy 的选择。高温可能提高低质量 token 的概率；低温仍可能重复。比较策略时应固定 prompt、参数、长度、其他过滤设置，并区分理论概率与单次输出。

未来真实模型接入点是 logits[:, -1, :] 之后、categorical 之前。不要在训练交叉熵中悄悄加入生成温度，也不要改 checkpoint。这里的参数拒绝策略是教学接口契约，不是 MLX 自带参数检查。

## 7. 练习（不附答案）

1. 手算 z=[0,ln3] 在 τ=1 与 τ=0.5 的 odds ratio。
2. 提出一个并列最大 logits 例子，解释 τ→0⁺ 与 argmax tie policy 的区别。
3. 为什么 temperature=NaN、0、负数应拒绝？为接口写出预期错误清单，不给答案代码。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/src/random.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/random.cpp)
- [MLX v0.32.2 / python/src/ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/ops.cpp)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

# Step 21.4 — Top-k：限制候选集合后再采样

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/95_step21_4_top_k_sampling.ipynb notebooks/work/95_step21_4_top_k_sampling_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版是**材料初稿 0/2**，不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

完整词表里可能有大量很低分候选。Top-k 先保留最高分的 k 个候选，再在这个集合内抽样。它改变支持集（support），不是只让分布变尖。Temperature 改变候选之间的比例；top-k 直接使部分 token 不再可能出现。

本节明确采用“严格 k 个索引”的定义。另一种常见阈值法保留所有大于等于第 k 大分数的项，会因并列值留下超过 k 个候选。两种策略都能合理，但不能混写定义和实现。

## 2. 先备数学与 Python / MLX 语义

设 S 为保留索引集合，z'ᵢ=zᵢ（i∈S），否则为 −∞。因为 exp(−∞)=0，softmax(z') 的分母只包含 S：p'ᵢ=exp(zᵢ)/Σⱼ∈S exp(zⱼ)。categorical 接受这些 masked logits，内部按其定义采样，不需要先显式 softmax。

为清晰定义并列规则，本节用 Python 排序：先按分数降序，同分按 token ID 升序，再取前 k 个。这需要把小数组传到 host，适合四词玩具，不是高吞吐实现。此取舍比用不确定 tie 行为的排序假装严格可复现更透明。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

z=[ln6,ln3,ln1]，未过滤时概率 [.6,.3,.1]。k=2 保留 ID 0,1 后概率变为 [2/3,1/3,0]，不是 [.6,.3,0]，因为必须重新归一化。k=1 时只剩一个候选；k=V 时支持集不变。

并列例子 z=[4,2,2,0]，k=2：本节约定保留 ID 0 和 1；若用 z≥2 的阈值则会保留 0,1,2，恰好显示两种算法的差别。

## 4. Shape tracing 与接口契约

单条 logits (V,) → host list 长 V → 排序得到 k 个原始 token ID → Boolean mask (V,) → masked logits (V,) → categorical 标量 ID。排序索引不是新的词表；输出必须仍指向原词表。接入批量 (B,V) 时需要逐行定义，本节不冒充已支持 batch。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 6. 常见误区与未来衔接

正温度缩放不改变排序，所以在精确算术下 top-k 集合不受温度影响；数值误差、并列规则仍需说明。若组合两个助手，先对有限原始 logits 做 temperature scaling，再交给本节 top-k，不能把 -Inf 喂给 21.3 的有限输入检查。

k 必须至少 1，否则所有位置被屏蔽，分布不存在。把被排除分数写成 0 不是屏蔽，因为 exp(0)=1。Top-k 不是 top-p/nucleus sampling；后者根据累计概率确定可变大小集合，不在本节实现。

## 7. 练习（不附答案）

1. 对四个不同分数手算 k=2 的归一化概率。
2. 设计包含三个并列最大值的例子，对比严格 k 与阈值法。
3. 解释为何返回排序后位置而不是原 token ID 会导致错误解码。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/src/random.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/random.cpp)
- [MLX v0.32.2 / python/src/ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/ops.cpp)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

