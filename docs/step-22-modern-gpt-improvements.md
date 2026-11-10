# Step 22 — Course materials

当前此 Step 已发布至 22.4。仅静态材料，不是运行结果。

# Step 22.1 — Weight tying：输入与输出共享词向量

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/97_step22_1_weight_tying.ipynb notebooks/work/97_step22_1_weight_tying_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版是**材料初稿 0/2**，不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

当前 MiniGPT 的 token embedding 和 LM head 各自有一份词表相关矩阵。Weight tying（权重共享）让输入查表和输出投影使用同一份参数：读入一个词的向量，也作为识别该词的方向。它减少参数并引入 inductive bias（归纳偏置），但不保证每个数据集都更好。

这是新架构分支的概念课，不是要求修改已验收 MiniGPT。共享后的参数树与 optimizer state 路径会变化；不能直接把旧 untied checkpoint 严格加载到新模型并当成无损改造。

## 2. 先备数学与 Python / MLX 语义

令 E∈R^(V×C)。输入 token i 的 embedding 为 E[i]；hidden h∈R^C 的输出分数 z=hEᵀ。元素式 zᵢ=Σc h_c Eᵢc，是 hidden 与第 i 个词向量的点积。

独立 LM head 权重形状也是 (V,C)，但本来是另一个矩阵 W。Tying 是同一参数的双重使用，不是初始化时复制 E 到 W。复制后两者会独立更新。若保留 output bias b∈R^V，则 z=hEᵀ+b；共享节省 VC 个权重，不是 VC+V。本节采用无 output bias 便于看清关系。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

V=3，C=2，E=[[1,0],[0,1],[1,1]]，h=[2,3]。手算分数为 [2,3,5]，第三个向量同时对齐两个维度。输入 ID 2 查出的就是 [1,1]。这里 embedding 的值是人为指定，不代表训练后的语义。

参数算例 V=100、C=16：两个独立矩阵共 3200 个标量；共享后为 1600。若原 LM head 带 bias，则 bias 的 100 个参数要另外保留或明确删除。

## 4. Shape tracing 与接口契约

IDs (B,T) → E 查表 (B,T,C)。hidden (B,T,C) @ E.T (C,V) → logits (B,T,V)。转置只交换 E 的两个轴，不能把 batch 或时间轴转置。MLX v0.32.2 `nn.Embedding.as_linear(h)` 正是 h @ self.weight.T，没有隐式 bias。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 6. 常见误区与未来衔接

在训练中同一 E 同时收到输入查表路径与输出分类路径的梯度贡献，由自动微分汇总；不要手工对共享参数更新两次。这个例子只计算前向，不建立 loss 或 optimizer。

Weight tying 不是减少词表，也不是量化。Embedding 和 head 维度不一致时需要额外映射，不可直接转置相乘。只共享权重而保留 bias，与连 bias 一起删除是两个不同设计。旧公式 P=2VC+TmaxC+N(12C²+7C)+C 属于原来的具体架构；不得不说明改动就套到新模型上。

## 7. 练习（不附答案）

1. V=7、C=4，分别计算保留与移除 output bias 的节省量。
2. 画出 E 在输入、输出两条路径中的使用位置。
3. 说明“把两个数组赋成相同初值”与“调用同一个 embedding.as_linear”在训练中的区别。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/mlx/nn/layers/embedding.py](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/embedding.py)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

# Step 22.2 — Learned positions → RoPE：用旋转表达相对位移

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/98_step22_2_rotary_position_embeddings.ipynb notebooks/work/98_step22_2_rotary_position_embeddings_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版是**材料初稿 0/2**，不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

Learned absolute positions 给位置 p 一条可训练向量，再加到 token embedding 上，直观但受位置表容量约束。RoPE（rotary position embedding，旋转位置编码）把位置信息放进每个 attention head 的 Q/K：对相邻维度对施加位置相关的二维旋转。

它不是给 embedding 再加一张表，也通常不旋转 V。本节手写相邻配对的数学参考，便于看清公式，不替换原模型；未来若用 fused RoPE 必须核对 pairing convention。

## 2. 先备数学与 Python / MLX 语义

二维旋转 R(θ)=[[cosθ,−sinθ],[sinθ,cosθ]]，作用于 (a,b) 得 (a cosθ−b sinθ, a sinθ+b cosθ)。旋转保持长度与同角度旋转后的点积。

对 head dimension d 为偶数，维度对 j 使用频率 ωⱼ=base^(−2j/d)，位置 p 的角度为 pωⱼ。Qₚ'=R(pω)Qₚ，Kq'=R(qω)Kq，因此 Qₚ'ᵀKq'=QₚᵀR((q−p)ω)Kq。点积依赖相对位移 q−p。内容向量本身当然仍不同，不能说整个 attention 只取决于距离。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

只看一个维度对，Q=[1,0]，K=[1,0]。若 Q 角度为 0，K 角度为 π/2，旋转后的点积为 0；若两者都额外旋转 π/3，点积仍为 0。这来自 cos(π/2)，不是运行结果。零位置角度为 0，旋转应保持原向量。

默认 base=10000 控制不同维度对的频率尺度。能计算任意位置的角度不代表模型已经学会任意长序列；训练长度、频率外推、数值精度与评测仍重要。

## 4. Shape tracing 与接口契约

Q/K (B,H,T,d) → reshape (B,H,T,d/2,2) → even/odd (B,H,T,d/2)。angles (T,d/2) 广播到 batch/head；stack 后恢复 (B,H,T,d)。position offset 是该 chunk 第一个 token 的绝对位置；增量 decode 时不能每轮都重置为 0。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 6. 常见误区与未来衔接

本函数用 float32 返回以突出数学，不承诺保持混合精度 dtype，也不支持部分维度 RoPE。生产实现还需明确 rotary dimensions、base、scale、offset、pairing、dtype。

MLX fast.rope 存在 traditional 配对参数；本节不调用它，该底层接口的 traditional 是必填 keyword，不能省略；避免将相邻配对与分半配对混为一谈。更换 position 机制意味着重新设计/训练或有依据的转换，不能将 learned position checkpoint 删除一张表后宣称等价。RoPE 与 KV cache 配合仍需保持位置编号和 mask 对齐。

## 7. 练习（不附答案）

1. 手算 (2,1) 旋转 π/2 后的坐标与平方范数。
2. 用 R 的转置与乘法解释相对位置公式。
3. 若已有 8 个 token 缓存，下一个单 token chunk 的 offset 应如何定义？说明依据。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/src/fast.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/fast.cpp)
- [MLX v0.32.2 / python/src/ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/ops.cpp)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

# Step 22.3 — GELU → SwiGLU：把内容分支与门控分支分开

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/99_step22_3_swiglu_gated_mlp.ipynb notebooks/work/99_step22_3_swiglu_gated_mlp_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版是**材料初稿 0/2**，不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

原始 MLP 常写成 Linear(C,4C) → GELU → Linear(4C,C)。Gated MLP 使用两个投影分支，一个生成内容，一个生成门控信号，然后逐元素相乘。SwiGLU 是用 SiLU/Swish 激活门控分支的 GLU 变体，许多现代语言模型采用它。

“门”不是硬开关，也不是注意力权重。本节不声称换一行激活就能无代价变好；结构和参数预算都变了，需要独立训练和比较。

## 2. 先备数学与 Python / MLX 语义

σ(x)=1/(1+exp(−x))，SiLU(x)=xσ(x)。SwiGLU(h)=[SiLU(hWg) ⊙ (hWu)]Wd，其中 ⊙ 为同形状逐元素乘法，Wg,Wu∈R^(C×F)，Wd∈R^(F×C)。MLX Linear 存储 weight 为 (out,in)，调用已处理转置。

无 bias 的标准 MLP 若隐藏宽度为 4C，参数为 8C²；无 bias gated MLP 有三份矩阵，参数为 3CF。要接近原预算，取 F≈8C/3，而不是仍用 4C。实际模型可能按硬件友好的倍数取整，必须再报告真实参数数目。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

先只看 gate=[0,2]、up=[3,−1]。SiLU(0)=0，SiLU(2)=2/(1+e⁻²)，乘积为 [0,−2/(1+e⁻²)]。门控结果可以为负，也不限制在 [0,1]，不能把 SiLU gate 当成概率。

C=6 时，标准 F=24 的两矩阵共有 288 权重；gated F=16 的三矩阵也有 288。这是符号计算，不是实际模型树计数或内存测量。

## 4. Shape tracing 与接口契约

h (B,T,C) → gate/up 各 (B,T,F) → SiLU(gate) (B,T,F) → elementwise product (B,T,F) → down (B,T,C)。MLP 在每个 token 的 feature 轴处理，不混合时间位置；跨 token 交互仍来自 attention。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 6. 常见误区与未来衔接

这个随机初始化模块只用于形状观察，不是迁移后的 MiniGPT，也不能说明质量提升。即使两个方案参数数量相同，激活驻留、乘法、kernel fusion、初始化分布都可能不同；不得据参数量宣称吞吐等价。

GLU 常用 sigmoid gate，SwiGLU 用 SiLU gate；GELU 是另一种激活，两者不能只凭名字替换。改变隐藏维度或增加分支后，旧权重和 Adam moments 的 shape/path 都不匹配，应保留原 baseline，另建实验配置。

## 7. 练习（不附答案）

1. C=12 时求理论同预算 F，并分别写出三个 MLX weight shape。
2. 解释逐元素乘法与矩阵乘法在这里各自出现的位置。
3. 考虑保留 bias 的方案，重写两个 MLP 的参数公式，不把无 bias 公式直接套用。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/mlx/nn/layers/activations.py](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/activations.py)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

# Step 22.4 — MHA → MQA / GQA：减少 KV 头的数量

## 先建立独立工作副本（以下只是命令参考，本次未执行）

在仓库根目录先同步母版：

```bash
git pull
```

再创建不覆盖已有文件的时间戳副本（仓库应已有 notebooks/work 目录）：

```bash
cp -n notebooks/100_step22_4_mqa_gqa_attention.ipynb notebooks/work/100_step22_4_mqa_gqa_attention_$(date +%Y%m%d_%H%M%S)_work.ipynb
```

只在工作副本中填写练习或未来运行。`cp -n` 在同名目标存在时不覆盖；同一秒再次操作也不会覆盖已有内容。当前母版是**材料初稿 0/2**，不是学习完成证明。

**执行边界：**所有代码单元 execution_count 为 null、outputs 为空。本次不执行 notebook、MLX 实验、测试、训练、验证、采样、语料读取/下载或 checkpoint 操作。下面的短实验是将来可观察的安排，不是已观察结果。MLX API 基线固定 v0.32.2。

## 1. 为什么需要这一节

Multi-head attention（MHA）为每个 query head 配一个独立 K/V head。Multi-query attention（MQA）让所有 query heads 共享一组 K/V；grouped-query attention（GQA）在中间取折衷：多组 query heads 各共享一个 K/V head。

动机尤其来自 autoregressive decoding 的 KV cache。每步只生成一个 token，却需要读取历史 K/V；减少 KV heads 能减少逻辑缓存容量和相关带宽需求。它改变的是架构与参数，不是纯粹换一个执行算法。

## 2. 先备数学与 Python / MLX 语义

令 Hq 为 query heads、Hkv 为 KV heads，要求 Hq % Hkv=0。每组 g=Hq/Hkv 个 query heads 使用同一个 KV head。MHA: Hkv=Hq；MQA: Hkv=1；GQA: 1<Hkv<Hq。

每层 K/V cache 的元素数为 2BTHkv d，N 层、每元素 s bytes 时逻辑容量为 2NBTHkv d s。这个公式没有包含模型权重、临时 attention scores、allocator 保留、碎片或系统 RAM。GQA 减少 KV 数量不等于消除 query heads 的打分计算。

## 3. 纸笔走一遍（讲解例子，不是练习答案）

B=1，N=2，T=64，d=16，Hq=4，float16 每元素 2 bytes。MHA Hkv=4：逻辑 cache 为 32768 bytes；GQA Hkv=2：16384 bytes；MQA Hkv=1：8192 bytes。只是在给定假设下的手算，不是硬件测量。

GQA Hq=4,Hkv=2 的映射为 query heads [0,1]→KV head 0，[2,3]→KV head 1。不能交错成 [0,2] 与 [1,3] 又声称是同一 head 排列。

## 4. Shape tracing 与接口契约

Q (B,Hq,Tq,d)，K/V (B,Hkv,Tk,d)。参考实现先 repeat K/V 到 (B,Hq,Tk,d)，score=(Q @ Kᵀ)/√d → (B,Hq,Tq,Tk)，softmax 后乘 V → (B,Hq,Tq,d)。显式 repeat 只用来教学，会放大临时存储；fast SDPA 应直接接收未平铺 K/V。

## 5. 短实验计划

先预测 shape 与趋势，再在未来工作副本中运行下一个单元。这里没有运行记录。若未来结果和推导不同，先检查输入、axis、dtype、mask、随机 key，再讨论数值误差；不要把一次随机输出当成分布证明。

## 6. 常见误区与未来衔接

MLX v0.32.2 fast SDPA 支持 MHA/MQA/GQA，要求 scale 用 keyword；不要为 fast 路径预先 repeat K/V。本节相等 T 的 causal mask 不涉及增量 decode 的偏移；非方阵需要单独设计对齐。

不同 head 数意味着 Q/K/V projection 输出宽度不同。原来的三个 C→C 投影不能只修改 reshape 就得到 GQA。把原 MHA checkpoint 的 K/V 平均成组也不是无损变换，不能宣称已兼容旧训练结果。参考/fast 数值差应使用容差理解，本次没有执行或记录差值。

## 7. 练习（不附答案）

1. Hq=8、Hkv=2 时画出 query→KV 映射，并列出 Q/K/V shapes。
2. 给定 N、B、T、d、dtype，写出 cache 容量随 Hkv 的比例关系。
3. 解释为什么 reference 中的 repeat 不应出现在最终缓存存储里。

请先用纸笔作答，再决定是否在工作副本中添加自己的代码。母版不填入参考答案。

## 8. 官方 API 阅读与完成边界

- [MLX v0.32.2 / python/src/fast.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/fast.cpp)

接口说明以固定版本源码为准。本次只有文本与源代码静态检查，没有运行通过声明。能够复述概念与完成练习后再进入下一节；材料覆盖范围不代表读者已经掌握。

