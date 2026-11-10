# Step 22 — Modern GPT improvements

本章五节的教学材料已准备，初始材料计数各为 0/2。只进行了文稿/Notebook JSON 的静态审阅；没有运行代码、实验、训练、验证、采样、checkpoint 或语料操作。材料范围完成不是学习完成、模型性能或运行通过的声明。所有实验都使用显式人工小数组或 toy callback，不依赖旧 kernel、真实语料或权重。

## 阅读顺序与独立工作副本

每节母版开头有独立的 `git pull` 与 `cp -n` 命令参考；不在母版中运行或填写练习。每本 notebook 自带必要 imports、helper 定义与人工输入。MLX 版本固定为 v0.32.2。

### 22.1 Weight tying：输入与输出共享词向量

[独立母版](../notebooks/97_step22_1_weight_tying.ipynb)

当前 MiniGPT 的 token embedding 和 LM head 各自有一份词表相关矩阵。Weight tying（权重共享）让输入查表和输出投影使用同一份参数：读入一个词的向量，也作为识别该词的方向。它减少参数并引入 inductive bias（归纳偏置），但不保证每个数据集都更好。

这是新架构分支的概念课，不是要求修改已验收 MiniGPT。共享后的参数树与 optimizer state 路径会变化；不能直接把旧 untied checkpoint 严格加载到新模型并当成无损改造。

**关键数学/接口：**

令 E∈R^(V×C)。输入 token i 的 embedding 为 E[i]；hidden h∈R^C 的输出分数 z=hEᵀ。元素式 zᵢ=Σc h_c Eᵢc，是 hidden 与第 i 个词向量的点积。

独立 LM head 权重形状也是 (V,C)，但本来是另一个矩阵 W。Tying 是同一参数的双重使用，不是初始化时复制 E 到 W。复制后两者会独立更新。若保留 output bias b∈R^V，则 z=hEᵀ+b；共享节省 VC 个权重，不是 VC+V。本节采用无 output bias 便于看清关系。

**形状契约：**

IDs (B,T) → E 查表 (B,T,C)。hidden (B,T,C) @ E.T (C,V) → logits (B,T,V)。转置只交换 E 的两个轴，不能把 batch 或时间轴转置。MLX v0.32.2 `nn.Embedding.as_linear(h)` 正是 h @ self.weight.T，没有隐式 bias。

**使用边界：**

在训练中同一 E 同时收到输入查表路径与输出分类路径的梯度贡献，由自动微分汇总；不要手工对共享参数更新两次。这个例子只计算前向，不建立 loss 或 optimizer。

Weight tying 不是减少词表，也不是量化。Embedding 和 head 维度不一致时需要额外映射，不可直接转置相乘。只共享权重而保留 bias，与连 bias 一起删除是两个不同设计。旧公式 P=2VC+TmaxC+N(12C²+7C)+C 属于原来的具体架构；不得不说明改动就套到新模型上。

### 22.2 Learned positions → RoPE：用旋转表达相对位移

[独立母版](../notebooks/98_step22_2_rotary_position_embeddings.ipynb)

Learned absolute positions 给位置 p 一条可训练向量，再加到 token embedding 上，直观但受位置表容量约束。RoPE（rotary position embedding，旋转位置编码）把位置信息放进每个 attention head 的 Q/K：对相邻维度对施加位置相关的二维旋转。

它不是给 embedding 再加一张表，也通常不旋转 V。本节手写相邻配对的数学参考，便于看清公式，不替换原模型；未来若用 fused RoPE 必须核对 pairing convention。

**关键数学/接口：**

二维旋转 R(θ)=[[cosθ,−sinθ],[sinθ,cosθ]]，作用于 (a,b) 得 (a cosθ−b sinθ, a sinθ+b cosθ)。旋转保持长度与同角度旋转后的点积。

对 head dimension d 为偶数，维度对 j 使用频率 ωⱼ=base^(−2j/d)，位置 p 的角度为 pωⱼ。Qₚ'=R(pω)Qₚ，Kq'=R(qω)Kq，因此 Qₚ'ᵀKq'=QₚᵀR((q−p)ω)Kq。点积依赖相对位移 q−p。内容向量本身当然仍不同，不能说整个 attention 只取决于距离。

**形状契约：**

Q/K (B,H,T,d) → reshape (B,H,T,d/2,2) → even/odd (B,H,T,d/2)。angles (T,d/2) 广播到 batch/head；stack 后恢复 (B,H,T,d)。position offset 是该 chunk 第一个 token 的绝对位置；增量 decode 时不能每轮都重置为 0。

**使用边界：**

本函数用 float32 返回以突出数学，不承诺保持混合精度 dtype，也不支持部分维度 RoPE。生产实现还需明确 rotary dimensions、base、scale、offset、pairing、dtype。

MLX fast.rope 存在 traditional 配对参数；本节不调用它，该底层接口的 traditional 是必填 keyword，不能省略；避免将相邻配对与分半配对混为一谈。更换 position 机制意味着重新设计/训练或有依据的转换，不能将 learned position checkpoint 删除一张表后宣称等价。RoPE 与 KV cache 配合仍需保持位置编号和 mask 对齐。

### 22.3 GELU → SwiGLU：把内容分支与门控分支分开

[独立母版](../notebooks/99_step22_3_swiglu_gated_mlp.ipynb)

原始 MLP 常写成 Linear(C,4C) → GELU → Linear(4C,C)。Gated MLP 使用两个投影分支，一个生成内容，一个生成门控信号，然后逐元素相乘。SwiGLU 是用 SiLU/Swish 激活门控分支的 GLU 变体，许多现代语言模型采用它。

“门”不是硬开关，也不是注意力权重。本节不声称换一行激活就能无代价变好；结构和参数预算都变了，需要独立训练和比较。

**关键数学/接口：**

σ(x)=1/(1+exp(−x))，SiLU(x)=xσ(x)。SwiGLU(h)=[SiLU(hWg) ⊙ (hWu)]Wd，其中 ⊙ 为同形状逐元素乘法，Wg,Wu∈R^(C×F)，Wd∈R^(F×C)。MLX Linear 存储 weight 为 (out,in)，调用已处理转置。

无 bias 的标准 MLP 若隐藏宽度为 4C，参数为 8C²；无 bias gated MLP 有三份矩阵，参数为 3CF。要接近原预算，取 F≈8C/3，而不是仍用 4C。实际模型可能按硬件友好的倍数取整，必须再报告真实参数数目。

**形状契约：**

h (B,T,C) → gate/up 各 (B,T,F) → SiLU(gate) (B,T,F) → elementwise product (B,T,F) → down (B,T,C)。MLP 在每个 token 的 feature 轴处理，不混合时间位置；跨 token 交互仍来自 attention。

**使用边界：**

这个随机初始化模块只用于形状观察，不是迁移后的 MiniGPT，也不能说明质量提升。即使两个方案参数数量相同，激活驻留、乘法、kernel fusion、初始化分布都可能不同；不得据参数量宣称吞吐等价。

GLU 常用 sigmoid gate，SwiGLU 用 SiLU gate；GELU 是另一种激活，两者不能只凭名字替换。改变隐藏维度或增加分支后，旧权重和 Adam moments 的 shape/path 都不匹配，应保留原 baseline，另建实验配置。

### 22.4 MHA → MQA / GQA：减少 KV 头的数量

[独立母版](../notebooks/100_step22_4_mqa_gqa_attention.ipynb)

Multi-head attention（MHA）为每个 query head 配一个独立 K/V head。Multi-query attention（MQA）让所有 query heads 共享一组 K/V；grouped-query attention（GQA）在中间取折衷：多组 query heads 各共享一个 K/V head。

动机尤其来自 autoregressive decoding 的 KV cache。每步只生成一个 token，却需要读取历史 K/V；减少 KV heads 能减少逻辑缓存容量和相关带宽需求。它改变的是架构与参数，不是纯粹换一个执行算法。

**关键数学/接口：**

令 Hq 为 query heads、Hkv 为 KV heads，要求 Hq % Hkv=0。每组 g=Hq/Hkv 个 query heads 使用同一个 KV head。MHA: Hkv=Hq；MQA: Hkv=1；GQA: 1<Hkv<Hq。

每层 K/V cache 的元素数为 2BTHkv d，N 层、每元素 s bytes 时逻辑容量为 2NBTHkv d s。这个公式没有包含模型权重、临时 attention scores、allocator 保留、碎片或系统 RAM。GQA 减少 KV 数量不等于消除 query heads 的打分计算。

**形状契约：**

Q (B,Hq,Tq,d)，K/V (B,Hkv,Tk,d)。参考实现先 repeat K/V 到 (B,Hq,Tk,d)，score=(Q @ Kᵀ)/√d → (B,Hq,Tq,Tk)，softmax 后乘 V → (B,Hq,Tq,d)。显式 repeat 只用来教学，会放大临时存储；fast SDPA 应直接接收未平铺 K/V。

**使用边界：**

MLX v0.32.2 fast SDPA 支持 MHA/MQA/GQA，要求 scale 用 keyword；不要为 fast 路径预先 repeat K/V。本节相等 T 的 causal mask 不涉及增量 decode 的偏移；非方阵需要单独设计对齐。

不同 head 数意味着 Q/K/V projection 输出宽度不同。原来的三个 C→C 投影不能只修改 reshape 就得到 GQA。把原 MHA checkpoint 的 K/V 平均成组也不是无损变换，不能宣称已兼容旧训练结果。参考/fast 数值差应使用容差理解，本次没有执行或记录差值。

### 22.5 Flash-style attention：数学相同，执行路径不同

[独立母版](../notebooks/101_step22_5_flash_style_attention.ipynb)

普通 attention 容易显式构建 T×T score/probability 矩阵；长序列时，中间矩阵读写会成为巨大负担。Flash-style attention 用 tiling（分块）与在线归一化避免把完整矩阵反复写回较慢内存。它不把 dense attention 近似成稀疏注意力，而是改变同一数学运算的执行算法。

Fused kernel 把多步运算放在一个协调执行路径中，减少中间结果和调度开销。fast API 名称不是实际硬件已使用某一 FlashAttention 论文 kernel 的证明；具体选择受设备、shape、dtype、mask 和实现 heuristics 影响。

**关键数学/接口：**

对单行 scores sⱼ，stable softmax 输出为 o=Σexp(sⱼ−m)vⱼ / Σexp(sⱼ−m)，m=maxⱼ sⱼ。在线处理一个新块 b：m'=max(m,max s_b)，α=exp(m−m')；l'=αl+Σexp(s_b−m')；u'=αu+Σexp(s_b−m')v_b。最后 o=u/l。

这里 u 是未除分母的加权和，l 是归一化分母，m 是运行最大值。当新块最大值更大时，必须把旧 u/l 同时乘 α；只更新最大值不缩放旧累计会出错。本节只为无 mask、非空、有限分数推导单行 reference，避免全屏蔽行的 −Inf−(−Inf)。

**形状契约：**

Q/K/V (B,H,T,d) → 普通 score (B,H,T,T) → P 同 shape → output (B,H,T,d)。分块方案为每行维护 m/l 标量与 u 长 d 向量，并暂存有限 tile。Dense attention 的算术量仍为 O(T²d)，但不需要把完整 score/probability 驻留在外部内存。

本节先用单行 scores (T,) 和 values (T,dv) 观察在线归一化，再展示 MLX fast SDPA 与显式 causal reference 的接口；不写自定义 Metal kernel、不测峰值或速度。

**使用边界：**

此 Python online reference 不是 FlashAttention kernel：MLX lazy graph 可能持有各块节点，Python loop 也有调度成本。不能用源码循环或未执行的样例声称节省实际峰值内存。真正性能评估需要控制设备、dtype、shape、预热、同步边界与 allocator 统计范围。

v0.32.2 的 `mask="causal"` 是 lower-right alignment（最后 query 对齐最后 key），方阵时与普通下三角一致。非方阵 prefill/decode 必须再次核对；array mask 为 broadcast-compatible Boolean 或 additive mask。`force_fused=False` 是默认，不强迫任何后端；强迫 fused 可能不支持或更慢，本课不设置。

课程范围决定：weight tying/RoPE/SwiGLU/GQA 是可选架构扩展；fast SDPA 是可选执行替换；KV cache 是可选推理状态设计。先保留已验收的最小模型作为基线，Step 23 的整理不应悄悄把这些概念示例混入核心实现。

## 静态检查与后续

每节都有动机、先备数学、纸笔例子、shape tracing、独立源码、未执行短实验及无答案练习。短实验中的 print 仅是未来观察入口，本次没有输出。允许的静态结构检查不等同于运行测试。

Step 22 的内容是可选扩展与执行机制比较，不修改当前 MiniGPT，也不承诺旧 checkpoint 兼容。Step 23 的源文件整理应保留教学 baseline；不要在重构同时默认切换 RoPE、SwiGLU、GQA 等架构。参数量、逻辑 cache 容量、实际 allocator 峰值、系统 RAM 与速度是不同指标。

## 固定版本官方来源

- https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/activations.py
- https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/embedding.py
- https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/fast.cpp
- https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/ops.cpp
