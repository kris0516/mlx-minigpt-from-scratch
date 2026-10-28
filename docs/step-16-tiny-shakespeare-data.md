# Step 16 — Tiny Shakespeare Data Pipeline

目标：把 Step 15 已经完成的 MiniGPT 从 toy token tensors 接到一份真实、可重复获取的小型 character-level language-model corpus。

本 Step 保留现有 roadmap 的 16.1–16.5 结构。Step 15 全部材料已通过静态审核；当前只准备 **16.1 Acquire and inspect Tiny Shakespeare**，不进入 16.2。

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

Step 16.1 只覆盖 acquisition/cache、encoding 与 corpus-scale inspection。修正计数从 0/2 起。

不进入 Step 16.2：不创建 train/validation split、不讨论 held-out validation 的实现、不创建 batch sampler、block_size、shifted batches、loss、optimizer 或 training loop。

本轮材料准备时不实际下载 Tiny Shakespeare，也不执行任何 Notebook cell、检查、测试或训练。
