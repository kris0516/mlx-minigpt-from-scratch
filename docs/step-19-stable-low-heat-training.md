# Step 19 — Stable Low-Heat Training

沿用既有roadmap，本轮只准备并发布 **19.1 Hyperparameter baseline**，初始实际修正 **0/2**；完成即停，不进入19.2。当前母版：[82_step19_1_hyperparameter_baseline.ipynb](../notebooks/82_step19_1_hyperparameter_baseline.ipynb)。

18.5真实五文件提交 `57dc27ee121430b11802b7c6f54fa2617ccd710e`、固定MLX v0.32.2接口及完整回复已由材料审核方独立核心静态验收PASS，实际修正0/2；大Step18全部材料已通过。这是材料验收，不是学员掌握或运行反馈。19.1只登记当前初始材料，不继承此前外部验收结论。

本大Step的stable／low-heat是课程目标，本节仅准备配置候选，没有已测得的稳定性、低热、内存、性能或收敛结果。已通过课节保持原样，后续按请求逐小节至大Step20全部材料完成后停止，不进入21。

## Step 19.1 — Hyperparameter baseline

### 0. 新工作副本怎样自包含？

继续使用项目Python3.12 / `.venv`；项目依赖仍固定 `mlx==0.32.2`。本节的可执行单元只准备普通Python字典、整数／浮点配置与返回字典的函数，使用Python内置能力，**不需要额外import或安装依赖**。

新副本按顺序提供：

1. 六项候选字典 `baseline_candidate`。
2. 从候选计算的head width、监督位置数与context capacity。
3. `make_model_kwargs`、`make_sampler_kwargs`、`make_optimizer_kwargs` 三个完整定义。
4. 仅整理sampler／Adam参数字典的使用单元。

这里没有调用 `MiniGPT`、`get_batch` 或Adam构造函数，所以不需要借用旧kernel中的模型类、tokens、RNG或optimizer。本节用真实接口表说明它们怎样接配置，不把未定义对象放进可执行调用。

真实 `vocab_size` 只作为模型配置函数的**显式实参**；本节不在顶层引用未恢复的 `itos`，也不填一个假的词表大小。以后真正连接语料与模型时，V必须来自与token IDs一致的字符表。这不影响当前六项配置工作的自包含性。

所有以下单元仍然只编写为材料；本轮不执行，包括这些普通Python配置单元。

### 1. Parameter、Hyperparameter与Baseline的区别

**Parameter（模型参数）** 是训练会更新的数组数值，例如embedding矩阵、attention投影权重、MLP权重。**Hyperparameter（超参数）** 是我们预先指定的结构或更新设置，例如向量有多宽、叠多少层、一次看多长窗口、学习率是多少。

**Baseline（基线）** 是后续比较使用的明确参照。19.1先给出 **baseline configuration candidate（基线配置候选）**，只有在相应运行与记录发生后，才可能拥有实测基线结果。

| 内容层次 | 本节能写什么 |
|---|---|
| 配置候选 | 明确指定六项数值，并解释各自作用 |
| 接口约束／静态推导 | 判断整数尺寸、整除、shape与字段映射是否相容 |
| 真正测得的结果 | 本轮没有loss曲线、吞吐量、温度、内存或收敛数据 |

前面的toy不能被改名为已调优基线：

- 15.4／15.5的 `V=32,C=16,H=4,N=2,T_max=16` 与输入 `(1,5)` 用于MiniGPT shape和参数手算。
- 17.3至18.5的SharedLogitModel只有三个共享logits，读取X.shape而不建模上下文；其中 `B=2,T=2,V=3` 和学习率 `1e-3` 用于接口与状态教学。

本节继承这些材料的接口、数学和状态关系，不继承“已找到有效训练设置”的结论。已通过出处：[65母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/65_step15_4_implement_minigpt.ipynb)、[66母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/66_step15_5_full_forward_shape_trace.ipynb)与[81母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/81_step18_5_memory_instrumentation.ipynb)。

### 2. 先约定符号与真实名字

以下符号都描述已有模型／数据接口，不增加新的模型组件：

| 符号 | 英文术语／中文含义 | 真实接口位置 |
|---|---|---|
| \(C\) | model width，模型宽度 | `MiniGPT` 的 `dims` |
| \(N\) | layers，独立Transformer Block数量 | `num_blocks` |
| \(H\) | heads，attention头数 | `num_heads` |
| \(D=C/H\) | head dimension，每头宽度 | 由C和H推导 |
| \(T\) | block size，当前输入窗口长度 | sampler的 `block_size` |
| \(B\) | batch size，一批窗口的行数 | sampler的 `batch_size` |
| \(\eta\) | learning rate，学习率 | Adam的 `learning_rate` |
| \(T_{\max}\) | 模型支持的最大位置数 | `max_seq_len` |
| \(V\) | vocabulary size，词表大小 | `vocab_size`，由实际字符表决定 |

输入X与shifted targets Y的shape都是 `(B,T)`；模型隐藏表示为 `(B,T,C)`；输出raw logits为 `(B,T,V)`。C是每个位置的特征维度，V是类别数量，二者职责不同，**没有C必须大于等于V的要求**。

本节表格中的V保留为数学符号，不是一个尚未定义却被代码引用的Python变量，也不是传给MLX数组构造函数的符号shape。

### 3. Model Width与Layers：横向宽度和纵向深度

#### Model width：每个token用多长的向量表示？

宽度C决定embedding和residual stream的最后一轴。把C从16改为64，表示每个位置由64个特征分量描述；它不把词表改成64个字符，也不把输入长度改成64。

在已有实现中，合并QKV投影将 `(B,T,C)` 映射为 `(B,T,3C)`，MLP中间宽度固定为4C，再投影回C。固定MLX v0.32.2的Linear权重按 `(output_dims,input_dims)` 存储，例如QKV权重为 `(3C,C)`。因此一些参数项随 \(C^2\) 增长，这是结构上的数学关系，不是实测速度／内存比例。

#### Layers：顺序通过多少个独立Block？

N是Transformer Block数量，每一层有自己的参数。已有代码使用逐次构造的Block列表，不把同一个Block引用重复N次。各层都接收并返回 `(B,T,C)`，所以叠两层不会给最终输出新增一个“层数轴”。

本节保持已通过的Pre-Norm、RMSNorm、attention与 `C→4C→C` MLP结构，也保持token embedding与LM head不共享权重。只配置宽度和层数，不改模型架构选项。

依据：[既有MiniGPT类](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/65_step15_4_implement_minigpt.ipynb)与[MLX v0.32.2 Linear](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/linear.py)。

### 4. Heads：把固定宽度分成几份？

当前实现把每个Q/K/V的C维拆成H个头，每头宽度为：

$$
D=\frac{C}{H}.
$$

需要H和C为正整数，且H整除C。本节候选会取 \(C=64,H=4\)，所以 \(D=16\)。这是对既有shape contract的手算。

代码中的 `dims % num_heads` 使用取余运算：余数为0表示可以整除；在已满足整除时，`dims // num_heads` 得到整数head width。本节不靠整数除法偷偷截掉不能均分的维度。

分头后每个Q/K/V的shape为 `(B,H,T,D)`，attention scores为 `(B,H,T,T)`。H不对应物理GPU核心数。

在本课程现有full-width QKV设计中，固定C时，QKV与输出投影的参数总数不随H变化；变化的是分头组织和D。另一方面，scores的逻辑元素数含H，因此不能进一步说“头数完全不影响资源”。实际分配与性能仍受实现和执行方式影响，本轮不测量。

### 5. Block Size：一次输入窗口有多长？

16.3的固定长度sampler让 `T = block_size`。它描述每行输入的token数，**不表示Transformer Block数量**；后者是N／num_blocks。

每行还要有最后一个next-token target，因此从同一个split中取得T+1个连续token，再形成：

$$
X[b,t]=\text{data}[s_b+t],\qquad
Y[b,t]=\text{data}[s_b+t+1],
\quad 0\le t<T.
$$

b是batch行下标，t是行内位置，\(s_b\) 是这一行在所选split中的起点。Y已经shift，不在loss处再次shift。

若取T=64，每行需要一个65-token chunk；所用split至少满足：

$$
N_{\text{split}}\ge T+1=65.
$$

这是split长度下限，不是词表大小。当前sampler允许重复起点和重叠窗口，所以不能把这个下限错误改成 \(B(T+1)\)。train和validation各自使用自己的split，窗口不跨分界；本节不读取任何split。

模型的learned position table容量由 \(T_{\max}\) 决定，已有forward约束是：

$$
1\le T\le T_{\max}.
$$

本次候选采用 `max_seq_len = block_size = 64`，让采样窗口与模型位置容量对齐。这是本节显式选择，模型接口本身允许T小于max_seq_len。

当前显式attention scores包含两个T轴。只看逻辑元素数，在B、H不变时，T翻倍会使 \(BHT^2\) 变为四倍；它不证明实际峰值内存、时间或温度也会按四倍变化。接口来源：[69 sampler母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/69_step16_3_batch_sampler.ipynb)。

### 6. Batch Size：一次更新汇总多少个监督位置？

B是一批窗口的行数。固定T时，增大B增加这次前向／loss中的窗口行数，不改变模型权重本身的shape。

沿17.1／17.2，每个 `logits[b,t,:]` 对应一个 `Y[b,t]`，总计B×T个 **token-level supervision positions（token级监督位置）**。当B=4、T=64时：

$$
BT=4\times64=256.
$$

它们用于一次mean loss：

$$
\mathcal L(\theta;X,Y)
=\frac{1}{BT}\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\ell_{b,t}.
$$

\(\ell_{b,t}\) 是该位置的交叉熵，\(\theta\) 表示模型参数。双重求和把所有行、所有位置相加，再除以位置数。一次这样的loss及其梯度供给一次optimizer update，不是进行256次更新。

这个mean已经包含 \(1/(BT)\) 的缩放，不在Adam处再自动把梯度或学习率除以256。更换B/T也不自动授权套用某种学习率缩放规律。

由于窗口可能重叠，256个监督位置不保证来自256个互不重复的语料token；它也不是256个不同类别。实际监督对应关系继续由原sampler决定。

### 7. Learning Rate：缩放当前Adam更新方向

\(\eta\) 是learning rate。本节候选 `3e-4` 使用Python科学计数法，等于 \(3\times10^{-4}=0.0003\)，是普通浮点配置，不是MLX默认学习率。

沿课程显式使用 `bias_correction=True` 的Adam，可以把一次更新写成：

$$
\theta_k=\theta_{k-1}-\eta u_k,\qquad
u_k=\frac{\widehat m_k}{\sqrt{\widehat v_k}+\epsilon}.
$$

k表示第k次更新；\(\widehat m_k,\widehat v_k\) 是17.4已讲过的偏差修正矩，epsilon是分母里的小常数。向量分式按对应分量计算。u取决于当前梯度和Adam历史，因此同一学习率不意味着每步参数移动同样距离，更不意味着loss下降固定数值。

只看一个分量的算术小例子：若人为给定 \(\theta=0.5,u=0.1,\eta=0.0003\)，则：

$$
\theta_{\text{new}}
=0.5-0.0003\times0.1
=0.49997.
$$

这里的u是解释比例关系的给定数，不是本项目实际算出的梯度或更新方向；结果也是纸算，不是训练记录。

固定MLX v0.32.2中，Adam的learning_rate是必填参数；betas默认[0.9,0.999]，eps默认1e-8，**bias_correction默认False**。本节继续保留课程显式True，只在六项候选中选择learning_rate，不引入scheduler或新优化器。

源码依据：[v0.32.2 Adam签名与apply_single](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/optimizers/optimizers.py)。

### 8. 本轮选定一组候选，不安排搜索

下面是一组人为指定、待后续验证的教学起点。理由在于让完整MiniGPT的宽度、深度、分头、窗口和监督规模形成明确的参照，不代表这些数值优于其他组合。

| 教学配置键 | 候选值 | 本轮的设计理由 |
|---|---:|---|
| `model_width` | 64 | 明确采用64维token表示，便于计算分头与投影shape |
| `layers` | 2 | 保留多个独立Block，仍容易逐层对应已有模型 |
| `heads` | 4 | 与64整除，每头16维 |
| `block_size` | 64 | 明确窗口长度，并在本次同时设位置容量64 |
| `batch_size` | 4 | 一次使用四个窗口，得到256个监督位置 |
| `learning_rate` | 3e-4 | 明确一个固定Adam更新尺度，尚无调优证据 |

这不是机器的安全容量上限，也没有“低热”“一定不Swap”“必然收敛”的结论。原toy学习率1e-3没有被当成默认或已调优起点。

先解释Python写法：`{key: value}` 创建dict（字典）；字符串key给每项起名，冒号后是数值。前五项使用整数，最后一项使用浮点数。`baseline_candidate["heads"]` 以后按键读取头数。字典名和这六个key是本节的配置格式，稍后会映射到已有接口的真实参数名。

```python
# BASELINE CONFIGURATION CANDIDATE — prepared only, NOT executed.
baseline_candidate = {
    "model_width": 64,
    "layers": 2,
    "heads": 4,
    "block_size": 64,
    "batch_size": 4,
    "learning_rate": 3e-4,
}

# One explicit candidate; no model, optimizer, data, search, or measurement.
```

#### 从六项配置派生三项信息

head_dim由C/H决定，监督位置数由B×T决定；它们不是又新增两个自由调节量。max_seq_len则采用本节已经说明的“等于block_size”约定。

下面只读取普通字典数值，使用整数除法 `//` 与乘法 `*`，返回另一个普通字典。所选64与4在数学上整除；这不是一段已经运行通过的验证代码。

如果以后修改C或H，仍需遵守整除约束，不能把 `//` 的截断结果当成修复不相容配置的方法。若修改原候选，已经算出的派生字典不会自动随之更新，应重新由新候选整理。

```python
# Derived Python configuration only; no arrays are constructed.
derived_candidate = {
    "head_dim": (
        baseline_candidate["model_width"] // baseline_candidate["heads"]
    ),
    "supervision_positions_per_batch": (
        baseline_candidate["batch_size"] * baseline_candidate["block_size"]
    ),
    "max_seq_len": baseline_candidate["block_size"],
}
```

### 9. 把候选完整代入Shape

代入 \(C=64,N=2,H=4,T=64,B=4\)，手算 \(D=16\)、\(BT=256\)。以下只是从已有定义推导的expected shapes，**没有构造数组或做forward**：

| 位置 | 静态推导shape |
|---|---|
| X、shifted Y | 各 `(4,64)` |
| token／residual features | `(4,64,64)` |
| 合并QKV投影 | `(4,64,192)`，因为3C=192 |
| 分头后的Q、K、V | 各 `(4,4,64,16)` |
| attention scores／weights | `(4,4,64,64)` |
| 每个Block输出 | `(4,64,64)`；顺序经过两个独立Block |
| MLP中间层 | `(4,64,256)`，因为4C=256 |
| raw logits | `(4,64,V)` |
| flattened logits／targets | `(256,V)`／`(256,)` |
| token losses／mean loss | `(256,)`／`()` |

最后两个位置沿用既有cross-entropy与mean定义，没有增加新的loss。V来自实际字符表，本轮不假定是32、3或其他数值；不同字符表即使长度相同，token ID的映射也必须与模型使用的词表一致。

#### 参数量：保留已有公式，不把它变成实测内存

沿15.5已经通过的当前架构公式，令P为可训练标量参数个数：

$$
P=2VC+T_{\max}C+N(12C^2+7C)+C.
$$

两项VC分别来自未共享的token embedding与LM head；位置表占 \(T_{\max}C\)；每个独立Block占 \(12C^2+7C\)；最后C是final RMSNorm。公式保留原来的attention无bias、MLP两层有bias、每层两套RMSNorm等设置，没有改变架构。

本候选 \(C=64,N=2,T_{\max}=64\)，所以：

$$
P=128V+64\times64+2(12\times64^2+7\times64)+64
=103360+128V.
$$

因为没有恢复实际词表V，本轮不填一个最终总数。固定C时H不出现在这个参数公式里，B也不改变参数个数；这不表示它们对一次计算的中间数组没有影响。

P是从结构算出的参数数量，不是active／peak／cache memory。18.5已经说明真实分配还取决于状态、临时值、共享存储、缓存和执行边界，本节不重做测量。

公式来源：[66 shape与参数手算母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/66_step15_5_full_forward_shape_trace.ipynb)。固定版本层结构依据：[Linear](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/linear.py)、[Embedding](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/embedding.py)、[RMSNorm](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/normalization.py)。

### 10. 配置怎样送到已有接口？

以下是接口签名说明，不是执行调用：

```text
MiniGPT(vocab_size, dims, num_heads, num_blocks, max_seq_len)
get_batch(data, batch_size, block_size, rng)
Adam(learning_rate, betas=[0.9, 0.999], eps=1e-8, bias_correction=False)
```

MiniGPT是仓库已有的课程类，不是MLX库自动提供的同名模型。它没有名为layers、heads、block_size、batch_size或learning_rate的构造参数，不能把本节整个六项字典原样当成模型构造参数。

| 候选字段／来源 | 应送到哪里 |
|---|---|
| model_width | 模型 `dims` |
| layers | 模型 `num_blocks` |
| heads | 模型 `num_heads` |
| block_size | sampler的同名参数；本次也作为模型 `max_seq_len` |
| batch_size | sampler的同名参数 |
| learning_rate | Adam的同名参数 |
| 真实字符表长度 | 模型 `vocab_size`，作为另外一个显式实参 |

已有接口需要守住的约束：

- 模型各尺寸／数量为正，并满足C能被H整除；forward输入rank为2，且 \(1\le T\le T_{\max}\)。
- 原模型签名用int标注尺寸，但Python类型标注不会自动替你做严格整数校验。本节候选使用实际正Python整数。
- sampler使用正整数B/T，所选一维split有至少T+1个token；token ID与同一词表的合法索引一致。
- 本候选learning rate是正浮点数0.0003；Adam接口要求显式提供learning_rate，不存在本节可省略后自动得到3e-4的默认值。

下面三个函数只整理字段，不运行模型构造、数据采样或优化器。**kwargs** 是keyword arguments（关键字实参）的常用缩写，这里表示以后可用于对应接口的参数字典。

`def` 定义函数，`config` 与 `vocab_size` 是调用时明确提供的参数；`return` 返回新字典。函数里的 `config["..."]` 读取调用者传入的配置，不偷偷读取旧kernel变量。

模型配置函数的vocab_size按合同应为来自实际词表的正整数。若以后由已有字符表itos取得V，V就是该同一映射的长度；本节不调用任何读取语料或恢复itos的操作，也不把未绑定V伪装成可用模型。

```python
def make_model_kwargs(config, vocab_size):
    # vocab_size must come from the actual vocabulary used for token IDs.
    # This function returns configuration only; it does not create MiniGPT.
    return {
        "vocab_size": vocab_size,
        "dims": config["model_width"],
        "num_heads": config["heads"],
        "num_blocks": config["layers"],
        "max_seq_len": config["block_size"],
    }


def make_sampler_kwargs(config):
    # data and rng are separate runtime inputs, not hidden kernel globals.
    return {
        "batch_size": config["batch_size"],
        "block_size": config["block_size"],
    }


def make_optimizer_kwargs(config):
    # Preserve the explicit course Adam settings; do not construct Adam.
    return {
        "learning_rate": config["learning_rate"],
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "bias_correction": True,  # Course choice; MLX v0.32.2 default is False.
    }
```

#### 能在本节整理的配置，与尚未绑定的词表

sampler和Adam所需的配置字典已能完全由本节候选整理。下面只调用刚才的普通Python函数，得到字典；没有调用sampler、Adam或任何MLX数值运算。

`sampler_kwargs` 不含token stream或RNG对象；`adam_kwargs` 不含模型参数或m/v历史。配置字典与运行状态的职责不同。

`make_model_kwargs` 已完整定义，但本节不调用它，因为本轮没有取得真实词表V。不写 `vocab_size=None` 的伪完整模型配置，也不借用toy V。六项候选本身仍然完整；词表来源是一项明确的数据接口依赖，不是从旧kernel隐式取得的前置。

这些mapping函数按已有合同整理字段，不是完整配置验证器，也没有执行测试或声称约束检查已经运行通过。

```python
# Configuration preparation only — NOT executed during material preparation.
sampler_kwargs = make_sampler_kwargs(baseline_candidate)
adam_kwargs = make_optimizer_kwargs(baseline_candidate)

# make_model_kwargs is ready for an explicit, real vocabulary size later.
# No model/optimizer construction, corpus access, sampler call, or training.
```

### 11. 与Adam状态、Compile和内存边界怎样衔接？

本节字典记录的是“以后怎样配置对象”。它不是训练参数树，也不是18.4中同时捕获的 `[model.state, optimizer.state]`。

如果以后修改原字典，之前已经整理出的kwargs不会自动重新计算，已经创建的模型、Adam或compiled callable也不会仅因字典变化自动改好：

- C、N等结构变化需要据新配置建立匹配的模型及其依赖对象；不能让新模型继续配旧参数形状的Adam状态。
- H改变会改变head layout与相关计算关系，即使某些投影参数shape不变，也不能把旧编译关系当作自动更新。
- B/T改变会改变输入规格，涉及18.3讲过的shape相关缓存条件；候选配置本身不代表该规格的编译图已经准备好。
- 若本次同时按 `max_seq_len=block_size` 改变位置表容量，那也涉及模型结构。只改变当前输入T且仍小于原容量，则是另一种情况。
- 改变配置中的学习率不等于已经修改某个现存Adam对象里的学习率状态。

真正的单步仍要区分旧参数loss与新model／optimizer状态；真正的内存读数仍需明确的求值与测量边界。本节只承接这个关系，不加入compile、eval、内存getter、预热或更新调用。

配置字典也不能单独恢复训练进度：它不包含学到的权重、Adam历史或sampler状态。本节不实现checkpoint，不提前展开后续小节。

### 工作副本顺序、材料状态与停止点

以后使用本节新工作副本时：

1. 在仓库根目录先git pull，再以时间戳新文件名与 `cp -n` 将82母版复制到已有 `notebooks/work/`，保留母版和笔记分离。
2. 选择项目Python3.12 / `.venv` kernel。从候选字典开始，按顺序取得派生配置与三个mapping函数；本节代码全部使用Python内置能力，无需执行旧Notebook或恢复旧kernel。
3. 本节能整理六项候选、派生信息、sampler／Adam kwargs。模型kwargs函数保留明确的真实vocab_size实参；当前没有实例化或读取语料的步骤。
4. 把本节当作后续材料的配置参照，停在19.1，不自行追加训练、搜索、测量或验证结果。

| 项目 | 本轮状态 |
|---|---|
| 六项候选 | 已写入材料，待后续验证 |
| D、BT、shape、参数公式 | 按既有定义作静态／纸面推导 |
| 真实词表V、train/validation数据 | 本轮未读取、未恢复 |
| 模型／Adam运行对象、编译图 | 本轮未构造或调用 |
| 温度、Swap、速度、内存、loss曲线、收敛 | 无测量、无运行结论 |
| 学员掌握／运行验证 | 不作此类声明 |

累计实际修正：16.1=**1/2**、16.2=**1/2**（含已记录的超时后完整回复补齐），16.3至16.5、17.1至17.5、18.1至18.5各**0/2**，19.1初始生成=**0/2**。每节最多两次实际修正，跨轮和换对话不重置。

本轮仅编写发布母版／文档，没有执行Notebook、配置代码、代码测试、训练、实验、测量、采样或用户工作副本，没有读取／下载语料，也没有代答练习。

**完成即停在19.1，不进入19.2。** 后续仍须按请求逐小节准备，至大Step20全部材料完成后停止，不进入21。

### 固定版本依据与课程链接

- [既有roadmap](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/docs/course-roadmap.md)：19.1六项范围与后续小节划分。
- [65 MiniGPT实现](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/65_step15_4_implement_minigpt.ipynb)、[66 shape／参数公式](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/66_step15_5_full_forward_shape_trace.ipynb)：真实构造签名、独立Block、context contract与当前参数公式。
- [69 Batch sampler](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/69_step16_3_batch_sampler.ipynb)：B/T、T+1、shifted targets与split边界；[73 Flatten B×T](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/73_step17_2_flatten_bt_supervision.ipynb)：多个token-level targets与mean loss。
- [80 Mutable training state](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/80_step18_4_mutable_training_state.ipynb)、[81 Memory instrumentation](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/57dc27ee121430b11802b7c6f54fa2617ccd710e/notebooks/81_step18_5_memory_instrumentation.ipynb)：匹配对象组、状态版本、显式求值与测量口径。
- [MLX v0.32.2 Adam](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/optimizers/optimizers.py)：learning_rate必填、betas／eps默认值、bias_correction默认False与课程显式True。
- [MLX v0.32.2 Linear](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/linear.py)、[Embedding](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/embedding.py)、[RMSNorm](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/normalization.py)：当前结构的接口、权重布局与参数项。
- [当前roadmap](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/course-roadmap.md)与[handoff](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/teaching-handoff.md)：材料状态、累计修正与停止边界。

接口核对以 **MLX v0.32.2真实源码** 为准。在线文档可能随发布更新；本节没有将别的实现默认值套到MLX，也没有将源码核对写成运行验证。


---

## Step 19.2 — Train / Validation Loop

Current master: [83_step19_2_train_validation_loop.ipynb](../notebooks/83_step19_2_train_validation_loop.ipynb). Initial material preparation count: **0/2**. Step 19.1's six-file commit `3d2e95606daf1b5d789599c3d29fd0696240be26`, true MiniGPT/sampler interfaces, fixed MLX v0.32.2 contract, and full reply independently passed core static acceptance with corrections 0/2. This is material acceptance, not learner mastery, local corpus availability, or runtime feedback.

This lesson covers only periodic validation loss and relevant train/eval behavior. It carries forward the 19.1 candidate (width 64, 2 blocks, 4 heads, block size 64, batch size 4, learning rate 3e-4), the actual MiniGPT, the split-local sampler, scalar cross-entropy loss, Adam, and the explicit MLX evaluation/compiled-state boundary. It provides a self-contained work-copy setup and complete model definitions; corpus acquisition is a separately defined optional entry point and defaults to disabled. The local bootstrap reports a missing cache instead of silently substituting toy data. The material was published without running Notebook cells, reading/downloading the corpus, sampling, training, validating, testing, or measuring.

Validation batches are sampled only from the held-out validation split; training batches are sampled only from the train split. Validation directly calls the scalar loss and does not receive an optimizer or use `nn.value_and_grad`. It does not update model parameters or Adam state. A dedicated Python RNG is recreated from the validation seed for each validation estimate, so the selected validation windows are fixed across checks without consuming the persistent training RNG. The estimate is a sampled-window metric, not a claim that the full validation stream was traversed.

The validation helper temporarily calls `model.eval()` and restores the prior uniform mode in `finally`. In MLX v0.32.2, `eval()` is equivalent to `train(False)`; mode switching does not disable gradients or freeze parameters. The current MiniGPT has no Dropout, BatchNorm, or `self.training`-dependent forward branch, so train/eval mode does not change its current forward formula. `mx.eval` is a separate materialization boundary. The compiled training graph returns the pre-update loss while updating model/optimizer state; periodic validation then measures the current post-update parameter version. Logs distinguish these values.

The notebook's small four-update/two-step-validation schedule is a control-flow teaching example only, not a tuned run or the Step 19.5 baseline. No loss value, convergence, memory, speed, temperature, low-heat, or Swap claim is made. All cells are unexecuted and output-free. The current turn stops at 19.2 before 19.3.


### Step 19.2 dependency-completeness correction (first correction, 1/2)

The master at `notebooks/83_step19_2_train_validation_loop.ipynb` now defines `loss_fn` before its first use. It consumes raw logits with shape `(B,T,V)` and already-shifted targets `(B,T)`, flattens them in matching order to `(B*T,V)` and `(B*T,)`, applies cross entropy, and returns the mean scalar MLX array. It does not shift a second time, pre-apply softmax, or convert to a Python scalar inside the differentiable loss.

The notebook also now invokes a local bootstrap before model construction: locate the repository root (or set the explicit root/path), keep optional acquisition disabled by default, load the selected local cache, derive `vocab_size` from the same character mapping/token stream, perform the contiguous 90/10 split, and check that each split contains at least `block_size + 1` tokens. Missing cache raises a clear error; no implicit download or toy-data fallback is used. This correction changes dependencies only, not the accepted training/validation loop. No Notebook, data read/download, test, training, validation, sampling, experiment, or measurement was run.


---

## Step 19.3 — Training Stability

Current master: [84_step19_3_training_stability.ipynb](../notebooks/84_step19_3_training_stability.ipynb). Initial material preparation count: **0/2**. Step 19.2's actual commit `8a83021c12dd8f5561b562d4156238cefd507a84` and its full initial/correction replies independently passed core static acceptance with cumulative correction count 1/2. This is material acceptance, not learner mastery or runtime feedback.

The lesson covers initialization, global gradient norms, optional global gradient clipping, and NaN/Inf/divergence detection. It includes self-contained imports, the 19.1 candidate, optional corpus acquisition disabled by default, local cache loading, contiguous 90/10 split, split-local sampler, complete MiniGPT classes, scalar loss, validation helper, Adam and a stability-aware step definition. Missing cache stops clearly; no implicit download or toy-data fallback is used.

The global L2 norm combines squared values from all gradient leaves. Optional clipping scales the whole gradient tree before the Adam update; it does not guarantee loss reduction or diagnose bad data, NaN parameters, or an unsuitable learning rate. Non-finite checks mark an anomaly but do not prove a root cause. The candidate clipping threshold is not tuned. All code cells are unexecuted and output-free; no training, validation, sampling, testing, or measurement was performed. This is material preparation only. Stop at 19.3 before 19.4.


### Step 19.3 first minimal correction (1/2)

The accepted stability logic is unchanged. The gradient-tree helpers now use the actual MLX v0.32.2 namespace: the master imports `tree_flatten` and `tree_map` from `mlx.utils`, calls `tree_flatten(grad_tree)` in `global_grad_norm`, and calls `tree_map(...)` in optional global clipping. No notebook cells were executed; this is a source/interface correction, not runtime verification or learner mastery.


---

## Step 19.4 — Apple Silicon Resource Control

Current master: [85_step19_4_apple_silicon_resource_control.ipynb](../notebooks/85_step19_4_apple_silicon_resource_control.ipynb). Initial material count: **0/2**. Step 19.3 commit `d193f85640bf9f36e79fcaf035612f76ba133895` and its first minimal correction independently passed core static acceptance, cumulative correction 1/2. This is material acceptance only, not learner mastery or runtime feedback.

The lesson covers unified memory awareness, avoiding Swap as a risk-management goal, conservative batch/model sizing, the 18.5 MLX allocator counters versus system memory/Swap, explicit evaluation boundaries, and sustained heat versus short bursts. It includes self-contained imports, configuration, local corpus loader/split/sampler, complete MiniGPT definitions and loss. The cache path is explicit, optional acquisition defaults to disabled, and missing cache fails clearly. The local bootstrap and all resource-reading snippets are code for future practice only; no code, corpus access, training, measurement, system resource setting or permission change was performed.

Shape-based byte counts are logical payload estimates, not measured peak/process/system memory. MLX active/cache/peak counters are not Swap Used. Conservative sizing can reduce logical workload (for example, attention-score elements scale with B·H·T² for fixed H), but does not guarantee zero Swap, a temperature ceiling, a fixed speed, or lower peak memory. Short-burst behavior does not establish sustained thermal behavior. The lesson does not run training and does not enter Step 19.5.


---

## Step 19.5 — Baseline Training Run

Current master: [86_step19_5_baseline_training_run.ipynb](../notebooks/86_step19_5_baseline_training_run.ipynb). Initial material count: **0/2**. Step 19.4's commit `c2fccb8895b0e9076beb6a599833bbac7aeb71b3` was independently accepted as material, corrections 0/2; Step 19.2 and 19.3 remain 1/2 each. This is not learner mastery or runtime feedback.

The lesson prepares a future baseline protocol for wall-clock speed, training/validation loss curves, MLX allocator counters, and a JSON record. It reuses the 19.1 candidate and the accepted full MiniGPT, split-local sampler, scalar loss, stability helpers, Adam, and compiled stateful step. Corpus acquisition is disabled by default; missing local cache fails clearly. The proposed 20-update schedule, validation frequency, and clipping threshold are protocol candidates, not tuned values.

The protocol separates one warm-up update from timed samples; every timed training step includes explicit evaluation of the returned pre-update loss and updated state. Validation loss is measured after the corresponding completed update using the current parameters. Step time and whole-loop wall time have different scopes; per-update timing starts after batch sampling and includes the compiled step plus explicit evaluation, excluding validation and warm-up. Whole-loop wall time includes validation. Memory counters are MLX allocator bytes, not system RAM or Swap. The JSON record includes configuration, data SHA-256, split metadata, seeds, environment, timings, loss curve and allocator counters; unobserved temperature, Swap and system pressure remain null. All code cells are unexecuted and output-free. No training, validation, data access, timing, memory measurement, or system change was performed. Stop at 19.5 before Step 20.


### Step 19.5 first merge correction (1/2)

The master now restores the full dependency set before first use: `baseline_candidate`, `loop_plan` (20 measured updates / validation every 5 / 4 validation batches), `seeds`, `clip_max_norm`, config-mapping helpers, global norm, finite checks, and clipping helpers. The malformed comma-prefixed bootstrap was replaced with valid Python source; local cache remains explicit, acquisition defaults off, split remains contiguous 90/10, and both splits are checked for T+1 tokens.

Host-side `mx.eval` and finite-value checks are not inside a compiled trace. The baseline uses a deliberately uncompiled `stable_train_step`: loss/gradient checks and optional clipping happen before Adam update, then updated state is evaluated. The unused legacy loop entry point was removed. The protocol preserves one warm-up update plus 20 timed updates; validation points are labeled after completed updates 6, 11, 16, and 21. X/Y are evaluated before the per-step timer, so per-step time excludes Python sampling and includes the host checks plus final state evaluation. The record uses `mx.__version__`, includes `clip_max_norm`, and saves JSON with exclusive-create mode `x`. No code or data was executed/read; no runtime or learner-mastery claim is made.
