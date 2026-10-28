# Step 16 — Tiny Shakespeare Data Pipeline

目标：把 Step 15 已经完成的 MiniGPT 从 toy token tensors 接到一份真实、可重复获取的小型 character-level language-model corpus。

本 Step 保留现有 roadmap 的 16.1–16.5 结构。Step 16.1、16.2 的材料与核心教学均已通过，各实际修正 **1/2**；当前只准备并发布 **16.3 Batch sampler** 初版，实际修正 **0/2**，完成后停在本节，不进入16.4。

16.2 的一次实际修正为用户确认的超时后完整回复补齐，本次补记，不因从 mlx-chap1 转到 mlx-chap2 而重置。材料状态不代表用户运行或掌握情况。

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

本节初始生成与发布计 **0/2**；16.1、16.2累计实际修正均保留 **1/2**。每小节最多两次实际修正，新对话不重置。

本轮只准备并发布16.3，结束后停在本节。16.4的seed、determinism与RNG-state resume，16.5的工业数据管线，以及loss/optimizer/training均不在本节范围内。不下载或读取语料，不执行Notebook、assertion、测试、训练、实验或用户work副本，也不代答练习。

API参考：[Python 3.12 randrange](https://docs.python.org/3.12/library/random.html#random.randrange)、[MLX stack](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stack.html)。项目继续固定`mlx==0.32.2`。
