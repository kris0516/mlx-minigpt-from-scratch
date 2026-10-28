# Step 16 — Tiny Shakespeare Data Pipeline

目标：把 Step 15 已经完成的 MiniGPT 从 toy token tensors 接到一份真实、可重复获取的小型 character-level language-model corpus。

本 Step 保留现有 roadmap 的 16.1–16.5 结构。Step 16.1、16.2 的材料与核心教学均已通过，各实际修正 **1/2**；16.3 已独立核验核心 PASS，实际修正 **0/2**。材料审核方已独立核验16.4的完整回复、母版和本 Step 文档，核心静态 PASS，实际修正仍为 **0/2**；这是材料验收，不是学员学习或运行反馈。当前只准备并发布 **16.5 Toy sampling vs industrial input pipelines** 初版，实际修正 **0/2**。

大 Step 16 的五个小节材料现已准备，16.5仍是当前初版，不写成全部小节均已独立验收。本轮完成后停在大 Step 16，不进入 Step 17。

16.2 的一次实际修正为用户确认的超时后完整回复补齐，已补记，不因从 mlx-chap1 转到 mlx-chap2 而重置。材料状态不代表用户运行或掌握情况。后续仍按请求逐小节推进，至大 Step 20 全部完成后停止。

## Step 16.1 — Acquire and inspect Tiny Shakespeare

### Source and local cache

课程使用 Karpathy `char-rnn` 中的 `data/tinyshakespeare/input.txt`，代码里的远端来源为：

```text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

本地缓存路径：

```text
data/tinyshakespeare/input.txt
```

仓库 `.gitignore` 已经包含 `/data/`，所以后续用户实际下载的本地 corpus cache 不需要提交进 Git。

Acquisition helper 使用 cache-first 逻辑：文件存在则直接复用；不存在时才通过 Python standard-library `urllib.request.urlopen` 下载。Definition cell 与真正调用 acquisition 的 usage cell 分离，母版不执行下载。

### UTF-8 text inspection

缓存文件通过 `Path.read_text(encoding='utf-8')` 读取成 Python `str`。需要区分：

```text
len(text)                 character/code-point count
len(text.encode('utf-8')) UTF-8 byte count
```

preview 使用 `repr(text[:N])`，方便观察 newline 等 whitespace。

### Character vocabulary and encoding

复用 Step 6.1 的 character tokenizer：

```text
characters = sorted(set(text))
stoi = character -> integer ID
itos = integer ID -> character
encode / decode
```

newline、space、punctuation 与大小写字母都作为普通 character tokens。

当前 minimal tokenizer 不添加 `<UNK>`；由于 vocabulary 从完整 corpus 建立，corpus 本身可完整编码。外部 unseen character 会暴露 vocabulary mismatch。

### Corpus scale

character-level tokenizer 中一个 character 对应一个 token，因此：

```text
N_token = N_char
```

完整 corpus 编码后是一条 1-D integer token stream：

```text
token_stream.shape == (N,)
```

标准 Karpathy / nanoGPT Tiny Shakespeare 字符版的 reference expectation 为约/标准报告的 `1,115,394 characters` 与 `65` 个 unique characters。Notebook 不把这些数字写成硬 assertion；local cache inspection 是实际 operational authority。

### N versus model context T

Corpus length N 描述整份语料规模；MiniGPT context length T / T_max 描述一次 forward 的 window。通常 `N >> T`。本节只建立整条 token stream，不切 train/validation，也不采样 `(B,T)` windows。

### Operational order in the work copy

```text
1. run acquisition-helper definition
2. run acquisition usage -> cache hit / first download
3. read UTF-8 text + preview
4. inspect corpus size / vocabulary
5. define stoi / itos / encode / decode
6. run short round-trip sanity check
7. encode the full corpus to an MLX integer array (N,)
```

### Scope boundary

Step 16.1 只覆盖 acquisition/cache、encoding 与 corpus-scale inspection。最终静态审核通过，实际修正计数为 1/2（一次交付补齐）。

不进入 Step 16.2：不创建 train/validation split、不讨论 held-out validation 的实现、不创建 batch sampler、block_size、shifted batches、loss、optimizer 或 training loop。

本轮材料准备时不实际下载 Tiny Shakespeare，也不执行任何 Notebook cell、检查、测试或训练。

## Step 16.2 — Train/validation split

### Dependency / reuse

16.2 的输入仍是 Step 16.1 的 character-level `token_stream:(N,)`。为避免依赖另一个 Notebook 的 kernel state，16.2 母版自带一个 **local-cache bootstrap helper**：只读取 `data/tinyshakespeare/input.txt`，重建 `characters/stoi/itos/token_stream`，不联网。

如果 cache 不存在，helper 只抛出明确错误并提示先完成 16.1 acquisition；不会偷偷下载语料。

### Why held-out validation exists

Training split 提供允许更新模型参数的训练样本；validation split 用于在未参与参数更新的 held-out text 上评估模型。

核心规则：

```text
validation tokens are not optimizer training examples
```

本节不实现 loss / gradient / optimizer，只先建立数据隔离。

### Contiguous 90/10 split

对完整 1-D stream：

```text
token_stream:(N,)
```

使用：

```text
N_train = floor(0.9 * N)
N_val   = N - N_train
```

并切成：

```text
train_tokens = token_stream[:N_train]   -> (N_train,)
val_tokens   = token_stream[N_train:]   -> (N_val,)
```

原始 corpus index ranges 不重叠，因此 held-out validation token positions 不会同时属于 training split。

### Sequence integrity

当前单文件 character corpus 使用 contiguous split，不把单个 tokens 随机撒到 train/validation，以免破坏连续上下文结构。

后续 Step 16.3 的 training windows 必须完全落在 `train_tokens` 内，validation windows 必须完全落在 `val_tokens` 内，不能跨越 split boundary。本节只说明约束，不实现 window / block_size / sampler。

### Tokenizer boundary

沿用 Step 16.1 从完整 corpus 建立的 fixed character vocabulary。16.2 的 held-out 数据隔离针对 sequence tokens / optimization examples；本节不重新设计 tokenizer。

### Shape boundary

Split 后仍然只是两条 1-D streams：

```text
train_tokens:(N_train,)
val_tokens:(N_val,)
```

仍没有 `(B,T)` batch shape。B 与固定 context/window length 留给 Step 16.3。

### Scope boundary

Step 16.2 只覆盖 held-out validation 的目的、train/validation 数据隔离与 contiguous split。材料与核心教学已通过，实际修正计数为 **1/2**（包含超时后的完整回复补齐）；不重置，也不返工已通过内容。

不进入 Step 16.3：不定义 `block_size`、batch sampler、随机 start index、shifted `(x,y)` windows、loss、optimizer 或 training loop。

本轮材料准备不读取本地语料、不执行 split、不运行 Notebook、assertion、测试或训练。

## Step 16.3 — Batch sampler

母版：[69_step16_3_batch_sampler.ipynb](../notebooks/69_step16_3_batch_sampler.ipynb)。本节沿用6.5的采样接口，把16.2的真实 train/validation streams 接到模型需要的 `(B,T)` 输入与标签形状。

### New work copy and dependency definitions

本节母版包含 `Path`、`random`、`mlx.core` 导入，以及 `find_project_root`、`load_cached_character_corpus`、`split_train_validation` 的 definition-only cell。后两者复用16.2的字符词表、`mx.int32` stream 与连续90/10切分，不需要共享旧kernel，也不通过 `%run` 执行旧Notebook。

路径 usage 从 kernel CWD 向上定位本仓库，默认选仓库根目录的 `data/tinyshakespeare/input.txt`，显示绝对路径。若16.1的既有缓存位于其他目录，在本节把 `CACHE_PATH` 改为该文件的绝对路径，再运行bootstrap。缺缓存时只明确报错；16.3不下载、不复制或移动语料。

Bootstrap usage 才实际读取本地缓存并恢复 `itos`、`token_stream`、`train_tokens`、`val_tokens`、`split_index`。这些使用代码仅为未来work副本准备，本轮不执行。

### Batch size, block_size, and the extra target token

`batch_size=B` 指batch行数；本节固定长度sampler中 `block_size=T` 指每行输入长度。它与Transformer Block数量无关。以后送入MiniGPT时需要 `T<=max_seq_len`。

每行需读取连续T+1个token，因为最后一个输入位置也需要真实的next-token target：

```python
chunk = data[s : s + T + 1]
x = chunk[:-1]  # first T tokens
y = chunk[1:]   # last T tokens
```

即 `X[b,t]=data[s_b+t]`、`Y[b,t]=data[s_b+t+1]`，t=0…T−1。Shift不把序列开头接回末尾。

手算例：data=`[0,1,2,3,4,5,6,7]`，T=3，手工起点为1和4：

| 行 | X | Y |
|---|---|---|
| 0 | `[1,2,3]` | `[2,3,4]` |
| 1 | `[4,5,6]` | `[5,6,7]` |

这些是用于解释的指定数值，不是执行得到的随机输出。

### Legal start indices and the split boundary

最后一个target下标为s+T。要求 `s+T<N_split`，所以合法起点为 `0…N_split−T−1`，共N_split−T个。`rng.randrange(0,N_split-block_size)` 的右端不包含在内，恰好对应这个范围。

N_split=T+1时只有起点0；N_split<=T时必须报错。先选split再调用，整个T+1 chunk都在该split内：

```python
get_batch(train_tokens, batch_size, block_size, rng)
get_batch(val_tokens, batch_size, block_size, rng)
```

只依据完整stream上的起点归类不够，窗口最后的target也必须受边界限制。返回的starts是split-local indices；validation的原始下标为split_index+start。Batch行可以重叠或重复，每行内部仍按连续文本顺序排列。

### Sampler and output contract

接口保持 `get_batch(data, batch_size, block_size, rng)`。输入为单个已选split的一维整数stream；实现检查一维、正整数B/T和T+1最小长度。

对B个起点分别切片，将每行`(T,)`放入xs/ys，再用 `mx.stack(...,axis=0)` 增加Batch轴：

| 返回对象 | 类型/shape |
|---|---|
| X | MLX整数array，`(B,T)` |
| Y | MLX整数array，`(B,T)` |
| starts | Python `list[int]`，长度B |

Bootstrap的stream为`mx.int32`，切片与stack保持此dtype。Y仍是正确类别ID，不是one-hot或embedding。未来模型logits的`(B,T,V)`中，每个`logits[b,t,:]`与一个`Y[b,t]`配对；本节不实现loss或forward。

Sampler定义监督配对，causal mask定义每个位置允许使用的输入前缀。X/Y重叠并不允许模型看未来；沿用Step12的因果约束。

### Prepared usage and operational order

准备B=2、T=8的小例子，使用未显式指定seed的`random.Random()`提供起点。分别创建train/val的X/Y，准备打印shape、dtype、局部起点及各一行decoded文本。所有具体随机起点和字符串都由未来实际采样决定，材料不伪造运行输出。

之后真正使用work副本时，顺序为：依赖定义 → 路径确认 → 本地bootstrap/split → sampler定义 → 小batch usage → 按需观察一行文本。命令工作流保留两个独立终端块，先`git pull`，再以新文件名和`cp -n`安全复制本节母版，避免覆盖笔记。

### Scope and correction ledger

16.3 母版及完整回复核心内容已由用户独立核验 PASS，实际修正仍为 **0/2**；16.1、16.2累计实际修正均保留 **1/2**。不返工已通过课节。每小节最多两次实际修正，新对话不重置。

16.3 的原课节范围停在 batch sampler；seed、determinism与RNG-state resume 留给16.4，工业数据管线留给16.5，loss/optimizer/training另有后续课节。材料准备不下载或读取语料，不执行Notebook、assertion、测试、训练、实验或用户work副本，也不代答练习。

API参考：[Python 3.12 randrange](https://docs.python.org/3.12/library/random.html#random.randrange)、[MLX stack](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stack.html)。项目继续固定`mlx==0.32.2`。

## Step 16.4 — Reproducibility

母版：[70_step16_4_reproducibility.ipynb](../notebooks/70_step16_4_reproducibility.ipynb)。本节承接16.3的 `random.Random` 和 `get_batch`，解释随机采样怎样从头复现、怎样从保存的位置继续，以及完整恢复训练还缺哪些状态。

### New work copy and dependency recovery

新母版自带16.3的 `Path`、`random`、`mlx.core` 导入，以及 `find_project_root`、`load_cached_character_corpus`、`split_train_validation`、`get_batch` 定义。依赖定义不读取语料或采样；不依赖旧Notebook的kernel，也不通过 `%run` 执行旧课节。

路径 cell 默认指向仓库根目录的 `data/tinyshakespeare/input.txt`。若16.1既有缓存位于其他目录，将 `CACHE_PATH` 改为该文件的绝对路径。随后单独的bootstrap usage才从既有缓存恢复相同的字符映射、`mx.int32` stream、连续90/10 split；缺缓存只报错，不下载或移动文件。这些都是以后使用work副本的操作步骤，本轮材料准备没有执行。

### Seed, RNG state, and continuing draws

**Seed（随机种子）**是初始化随机数生成器的输入；**RNG state（随机数生成器状态）**记录它当前可继续产生后续输出的位置。伪随机数由算法产生，state随调用变化，不是每次凭空重新开始。

用 `S_0` 表示初始状态，seed决定：

```text
S_0 = Init(seed)
```

用 `S_k` 表示第k个batch之前的状态，`G` 表示本sampler取完整batch起点的过程，`N_split-T` 是合法起点数量：

```text
(starts_k, S_{k+1}) = G(S_k; B, N_split-T)
```

这里是整个batch的状态转移，不表示每次 `randrange` 恰好消耗一个底层随机数。保持一个rng对象持续使用，状态就继续前进。若每个batch前都重新执行 `random.Random(SEED)`，会重新从同一序列开头取样。

手工示意取N=8、T=3、B=2，合法起点0至4。假设连续两次batch的起点分别为 `[1,4]` 和 `[0,3]`：保存第一次之后的state，恢复后应接着取得第二组；重新使用原seed则回到第一组所处的起点。这只是便于解释的假设序列，不是某个seed的实测输出。

### Deterministic sampling expectations

**Deterministic sampling（确定性采样）**是在条件相同的情况下重复同样的采样结果；不是每次调用都应返回相同batch。

| 要重复的对象 | 必须保持的条件 |
|---|---|
| 从头的起点序列 | 相同seed、sampler实现、调用参数及顺序、兼容的Python运行环境 |
| 从某个边界继续的起点序列 | 恢复该边界的RNG state，后续条件与调用顺序相同 |
| 相同的X/Y内容 | 起点相同，且token数据、字符映射和split相同 |
| 完整训练轨迹 | 还涉及模型、优化器、其他随机源和执行环境；本节不保证或验证 |

同一个数据文件名或相同长度不足以说明token内容一致。改变B、T、split长度，或插入额外的随机调用，都可能改变后续起点；即使起点偶然相同，也不能据此认定全部状态相同。

输出接口仍是 `X/Y:(B,T)` 的整数数组以及长度B的 `starts` 列表。每行的shift关系仍为 `X[b,t]=data[starts[b]+t]`、`Y[b,t]=data[starts[b]+t+1]`；seed/state不改变shape或split边界。

### Separate random sources and prepared usage

母版先解释API再准备以下fresh-run setup：

```python
TRAIN_SEED = 2026
VAL_SEED = 2027
MLX_SEED = 2028

train_rng = random.Random(TRAIN_SEED)
val_rng = random.Random(VAL_SEED)
mx.random.seed(MLX_SEED)
```

两个Python对象分别负责train与validation起点；validation只消耗 `val_rng`，不会改变 `train_rng` 的位置。不同seed不保证每次结果不同，也不保证窗口互不重叠。

模块级 `random.seed(...)` 管理模块自己的默认生成器，不会替这些显式创建的Random对象设seed。`mx.random.seed(...)` 管理MLX随机源，不会给Python sampler设seed。它应在将来需要受其影响的MLX随机初始化之前调用；本节不创建模型或MLX随机数组。当前sampler的MLX切片与stack本身不产生随机起点。

使用代码继续取B=2、T=8，train/validation各准备一次batch调用，输出仍为 `(2,8)`。另用 `train_batches_sampled`、`val_batches_sampled` 记录各自已经采过多少个batch；这些计数不是optimizer训练步数、epoch或底层随机数调用次数。本轮不执行setup或sampling usage，不给出假造随机输出。

### In-memory continuation: getstate and setstate

`rng.getstate()` 返回当前状态；`rng.setstate(saved_state)` 将兼容状态装回生成器。两者自身不取新的随机样本。状态按完整对象保存，不拆解或猜测其内部tuple结构。

母版准备一个Python dictionary作为**内存快照（in-memory snapshot）**，保存：

- train_rng与val_rng的两份 `getstate()`；
- batch_size与block_size；
- train/validation已采batch计数。

恢复时创建两个替代Random对象，以 `setstate(...)` 覆盖其初始状态，再恢复B/T和计数。保留当前kernel中的同一份数据、split、词表与sampler定义，下一次取样就从保存的边界继续。恢复后不要重跑fresh-run setup，否则又会回到seed决定的起点。

快照保存“下一次从哪里开始”，不会自动重放刚采出的上一batch。它也无法追回16.3中没有保存seed/state的旧随机序列。内存字典不包含语料、MLX RNG、模型或优化器，kernel重启后便不存在；本节没有把它写成可跨进程恢复的完整checkpoint。

### What later resume must preserve

这里只明确将来恢复继续所需的条件，不提前实现checkpoint文件或训练恢复。

| 状态或条件 | 为什么需要 |
|---|---|
| 确切数据内容、字符ID映射、split规则/位置 | 同一起点必须仍指向相同token与同一数据集 |
| sampler代码、B/T等配置 | 保持相同合法范围、调用行为与batch形状 |
| 所有实际参与的Python RNG states | 从保存的位置继续各自的随机序列 |
| 采样进度及未来训练步数 | 区分已完成与下一步；两种计数不能混用 |
| Python/MLX版本、设备、dtype及相关执行条件 | seed本身不能保证跨环境逐位相同 |
| 将来的模型权重 | 保留模型已学到的参数 |
| 将来的optimizer状态 | 保留更新规则累计的历史；其内部细节留给Step17 |
| 将来实际使用的MLX随机状态 | 继续MLX侧随机计算，不能只重设初始seed |

MLX状态的实际保存/恢复API、模型与optimizer落盘、加载顺序和恢复验证留给后续Step18/20。项目仍固定MLX 0.32.2；不把 `mx.random.state` 当作普通可写list来提供赋值恢复代码，也不承诺仅设置几个seed就能得到跨硬件完全一致的训练结果。

### Operational order, scope, and ledger

以后使用新work副本时：依赖定义 → 既有缓存路径确认 → 本地bootstrap → 阅读seed/state与API → fresh-run setup → 按需采样 → 保存内存快照 → 恢复该快照后继续。恢复分支不重复fresh-run setup；每次重复sampling cell都会继续消耗对应RNG，顺序属于复现条件。

16.4 原课节范围止于 reproducibility，工业输入管线留给16.5。材料审核方已独立核验本节完整回复、母版和 Step 16 文档，核心静态 PASS；这不代表学员掌握或运行验证。原材料准备没有下载或读取语料，没有执行Notebook、测试、采样、复现实验、训练或用户work副本，没有代答练习。

截至16.4的累计实际修正：16.1=**1/2**，16.2=**1/2**，16.3=**0/2**，16.4=**0/2**。已通过课节不返工，计数不因换轮或换对话重置。

参考：[Python 3.12 random：seed、getstate、setstate与复现说明](https://docs.python.org/3.12/library/random.html)、[MLX random.seed](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.seed.html)、[MLX 0.32.2 random binding source](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/random.cpp)。

## Step 16.5 — Toy sampling vs industrial input pipelines

母版：[71_step16_5_toy_vs_industrial_input_pipelines.ipynb](../notebooks/71_step16_5_toy_vs_industrial_input_pipelines.ipynb)。本节依照既定 roadmap，将 random windows、shuffling、token packing、distributed shards、deterministic cursor/state 连成一组对照：数据规模增加后，除了得到 `(B,T)`，还需要说明样本顺序、组织方式、分配关系与继续位置。

### Self-contained work copy

本节采用可手算的 **toy inputs（人为构造的小输入）**。新母版自带 `random`、`mlx.core` 导入、16.3原样的四参数 `get_batch` 定义，以及本节所需的小 helper 和输入定义，不读取16.1的语料缓存，不依赖旧Notebook变量，也不 `%run` 旧课。

之后实际使用时，先安全复制为新的 `notebooks/work/*_work.ipynb`，选择项目 Python 3.12 / `.venv` kernel，再按母版顺序执行依赖定义与所需 usage。没有前节kernel状态时，也能从本节定义建立这些示意对象。项目仍使用 MLX 0.32.2，不增加第三方数据加载框架。

本轮只准备这些代码，不执行它们；下面的具体数值是手工构造或推导，不是采样运行输出。

### 1. Random windows: simple sampling does not promise coverage

沿用 `get_batch(data,B,T,rng)`：在一条已选定 split 内，每行从合法起点取T+1个连续token，构造长度T的X/Y。示意输入为ID 0至11，`B=2,T=4`，因此 `N_split-T=8`，起点范围为0至7，输出 `X/Y:(2,4)`。

当前起点采用 **sampling with replacement（有放回采样）**：某个窗口可能重复，另一个可能一直未抽到。行内文本顺序保持不变，但采样次数增加不等于所有候选窗口都恰好出现一次。

**Epoch（一轮遍历）** 要先明确遍历对象和规则。若目标是一份有限候选样本列表各访问一次，需要建立并遍历这份列表；16.3的随机窗口sampler本身没有提供这个保证。即使每个窗口ID只访问一次，重叠窗口仍可包含相同token位置。

### 2. Shuffling: reorder sample IDs, preserve sequence order

**Shuffling（打乱顺序）** 在此作用于文档、样本或窗口ID的顺序，不是把语言模型输入行内部的tokens打散。`rng.shuffle(order)` 在原list上修改顺序并返回 `None`，因此不能把它的返回值当成新的顺序列表。

对有限候选ID列表，保存其一个 **permutation（排列）** 并完整遍历，才能说这些候选ID各被访问一次。若中途停止、丢弃尾部或使用其他抽样规则，覆盖范围也要相应说明。列表顺序改变后，取出的每条序列仍保持原来连续的token顺序与shift关系。

数据规模很大或通过stream读取时，可能使用 **shuffle buffer（打乱缓冲区）**：只暂存有限数量样本，在缓冲区中随机选择并补入新样本。这种局部混排不等于把全部数据放进内存后做一次完整全局排列；buffer内容也可能成为恢复所需状态。

### 3. Token packing: fill rows and make boundary policy explicit

**Token packing（token打包）** 将较短序列组织成模型所需的较长行，以减少未利用位置。Packing仍需要决定文档边界如何影响输入可见性和监督目标，并不只有reshape或补齐shape的问题。

本节只用一个明确的连续流策略作示意：

```text
toy_docs = [[10,11,12], [20,21], [30,31,32,33]]
toy EOS = 99
```

**EOS（End Of Sequence，序列结束标记）** 在每份toy文档末尾追加一次，得到长度12的stream：

```text
[10,11,12,99,20,21,99,30,31,32,33,99]
```

99只是本节的synthetic token约定，没有加入Tiny Shakespeare的真实字符词表，也不送入当前MiniGPT。真实模型若采用特殊token，需要相应的词表、ID与embedding配置支持。

令T=4，每行读取T+1=5个token，并将起点推进T。这样前一行的最后target成为下一行的第一个input，连续相邻的next-token配对不会因行边界被漏掉。

| 行 | 读取的T+1 chunk | X | Y |
|---|---|---|---|
| 0 | `[10,11,12,99,20]` | `[10,11,12,99]` | `[11,12,99,20]` |
| 1 | `[20,21,99,30,31]` | `[20,21,99,30]` | `[21,99,30,31]` |

两行形成 `X/Y:(2,4)`。剩余 **carry（保留到后续继续处理的尾部）** 为 `[31,32,33,99]`：包含上一行末target的31，并保留之后三个尚未组成完整行的相邻配对。当前tail不足T+1，所以示例返回carry，等待更多token；不默默padding、绕回开头或丢掉这部分。

本toy策略接受跨文档target，例如99之后预测下一篇首token20；同一packed row中的后文也可在causal规则下看到前文。EOS标记边界，**不会自动屏蔽attention，也不会自动取消跨文档target**。若任务要求文档相互独立，就需要document/segment metadata，并明确可见性和目标规则。本节只说明其职责，不改变当前模型、不实现新的mask或loss，也不进入Step17。Train/validation仍需保持各自数据边界。

### 4. Distributed shards: storage and rank allocation are different layers

先区分几个术语：

| 术语 | 本节含义 |
|---|---|
| Storage shard | 数据在存储上的分片，例如不同文件或分块 |
| Rank | 分布式任务中某个进程的编号 |
| World size | 参与相应分布式任务的进程总数 |
| Loader worker | 为某个进程准备输入的工作单元，不自动等于rank |
| Local batch | 一个rank本次取得的batch |

存储上有多个shards，不自动保证不同rank不会读到重复样本。需要明确样本到rank的分配计划；它与train/validation用途划分也不同。

本节只准备一份纯Python计划，不启动分布式进程。给定手工指定的共享sample-ID顺序：

```text
order = [7,2,9,4,0,6,1,8,3,5]
world_size = 2
local_batch_size = 2
```

每个逻辑global batch有4个sample IDs；在这4个中按rank分给每个rank连续2个。**Sample ID（样本编号）** 用来查找一条record，与record内部的 **token IDs** 是不同层次。本节另备每个ID对应T+1个toy tokens的records，以展示ID查找如何最终得到X/Y。

| 逻辑batch | Rank 0 | Rank 1 |
|---|---|---|
| 0 | `[7,2]` | `[9,4]` |
| 1 | `[0,6]` | `[1,8]` |

这里明确采用 **drop tail（丢弃不足完整global batch的尾部）**：只使用前8个ID，剩余 `[3,5]` 本轮不消费。这样每rank都有两个完整local batches，但不能再声称本轮完整覆盖全部10个样本。补齐会可能引入重复，保留不等长尾部则要另行处理各rank工作量；规则必须显式选择。

每rank的 `X/Y:(2,T)`，对应的逻辑global shape为 `(4,T)`。这只是样本分配与shape关系，不涉及通信、梯度累积或真实分布式训练。

### 5. Deterministic cursor/state: position depends on a plan

**Cursor（游标）** 必须说明指向哪个顺序、哪个位置。本节的 `next_batch_index` 指向当前rank计划中“下一批”的下标，不是原语料token偏移，也不是训练update step。以上例，值为1时，rank 0下一批sample IDs是 `[0,6]`，rank 1对应 `[1,8]`。

内存plan snapshot保存完整 `tuple(order)`、epoch、rank/world_size、local batch size、T、tail policy、声明的toy data identity label及 `next_batch_index`。在toy records与runtime保持不变时，按同一配置重建计划，再由cursor定位下一批。该label是用于说明前提的元信息，不是内容完整性校验。

当当前epoch的完整order已经固定保存、后续没有额外随机变换时，定位这个epoch内的下一批不需要再次随机抽取。之后若要产生新排列、继续随机变换或进行其他随机操作，仍需要16.4所讲的相应RNG state。World size改变会改变local assignment，不能直接照搬旧local cursor。

真实input pipeline还可能需要：

| 状态 | 所描述的当前位置或条件 |
|---|---|
| Shard manifest / order | 有哪些数据分片、按什么次序读取 |
| Reader positions | 每个读取者读到分片中的哪里 |
| RNG states | 后续随机动作的当前位置 |
| Shuffle buffer | 已进入缓冲区但还没有交付的样本 |
| Packing carry及文档边界信息 | 已读入但尚未组成完整行的tokens及其语义 |
| Consumed position与prefetch pending | 下游已消费到哪里，还有什么已准备但未交付 |
| Topology / software metadata | rank/worker分配与版本等解释状态的条件 |

**Prefetch（预取）** 会使读取进度领先于实际消费进度。例如reader读到第12项、下游只消费到第8项，若仅保存reader位置，恢复后从第13项开始就会跳过第9至12项。需要保存未消费队列，或从明确的已消费边界确定性重建相应数据。

本节的内存plan snapshot不声称保存packing流、真实loader、模型/optimizer、待处理训练batch或完整训练状态，也不写磁盘文件。持久化和实际训练恢复仍留给Step20。

### Step 16 material boundary and correction ledger

本轮只准备并发布16.5，至此大Step16的五小节材料均已准备，16.5仍是当前初版，不写成全部独立验收。停止在大Step16，不进入Step17。

累计实际修正保持：16.1=**1/2**、16.2=**1/2**、16.3=**0/2**、16.4=**0/2**、16.5初始生成=**0/2**。每小节最多两次实际修正，跨轮、换对话不重置。后续继续按请求逐小节至大Step20全部完成后停止。

没有读取或下载语料，没有执行Notebook、测试、训练、实验、采样或用户work副本，没有代答练习。材料完成、手算示意与静态审阅均不代表学员掌握或运行验证。

参考：

- [Python 3.12 random.shuffle](https://docs.python.org/3.12/library/random.html#random.shuffle)：原地重排API。
- [Hugging Face Datasets streaming](https://huggingface.co/docs/datasets/en/stream)：有限shuffle buffer、流式状态与恢复限制。
- [Hugging Face Datasets distributed说明](https://huggingface.co/docs/datasets/en/use_with_pytorch#distributed)：共享顺序与rank分配的关系。
- [NVIDIA Megatron-LM GPT dataset](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py)：attention边界和目标处理分别配置的官方实现依据。
- [MLX Data Buffers, Streams and Samples](https://ml-explore.github.io/mlx-data/build/html/buffers_streams_samples.html)：数据访问与预取次序。

这些工业文档仅用于理解机制对照，本节不安装或调用其数据框架，也不增加项目依赖。
