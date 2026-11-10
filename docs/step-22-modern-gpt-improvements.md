# Step 22 — Course materials

当前此 Step 已发布至 22.1。仅静态材料，不是运行结果。

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

