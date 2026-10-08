# Course Roadmap — Apple Silicon MLX MiniGPT from Scratch

> This document is the persistent curriculum plan for the repository.
> It is intentionally detailed so a future ChatGPT/Codex session can continue the course without reconstructing the teaching plan from chat history.

## Project goal

Build a small **Decoder-Only MiniGPT** from scratch on an Apple Silicon Mac using **Apple MLX**, while learning the historical and mathematical path that led from basic neural networks to modern GPT-style models.

The project is deliberately optimized for:

- Apple Silicon / M-series Mac
- VS Code
- Python 3.12
- Apple MLX as the core framework
- low memory usage
- low heat / avoiding unnecessary sustained GPU load
- avoiding Swap where practical
- understanding tensor flow, computation graphs, gradients, and hardware behavior
- learning the historical progression:
  **MLP → RNN → LSTM/GRU → Seq2Seq → Attention → Transformer → GPT**
- ending with a clean, reproducible, public GitHub repository

## Student starting point

Assume the learner:

- understands basic Python syntax;
- can recognize and perform simple matrix multiplication;
- should **not** be assumed to already know calculus, tensor algebra, automatic differentiation, probability, optimization, or neural-network internals;
- benefits from prerequisite mathematics being introduced **just in time**, with small numeric examples.

## Teaching cadence

### Current pacing rule

From **Step 4 onward**:

- each Step contains **at most 5 subsections**;
- each subsection should cover a meaningful cluster of tightly related ideas;
- the five-subsection cap must never be used to compress away prerequisite reasoning, examples, edge cases, or shape analysis; each subsection must be self-contained enough for the learner to reconstruct the idea later;
- do not split every API or formula into its own micro-step;
- each teaching turn should still remain digestible, but depth takes priority over artificial brevity;
- when geometry, optimization behavior, distributions, embeddings, loss curves, or performance trends are easier to understand visually, include a small plot and explain the plotting technique used.

Step 3 contains 7 historical subsections because it was completed before this pacing rule was adopted.

### Required teaching pattern

For each subsection:

1. explain the problem or historical motivation;
2. introduce only the minimum prerequisite mathematics needed;
3. use a small hand-computable numeric example when useful;
4. show a short MLX code experiment;
5. map the experiment to the eventual MiniGPT implementation;
6. write/update the corresponding tracked course notebook and notes in GitHub.

Avoid API-only explanations.

---

# Persistent notebook workflow

Tracked course notebooks are maintained under:

```text
notebooks/*.ipynb
```

These are **master notebooks**. The learner should not run or edit them.

Personal runnable copies live under:

```text
notebooks/work/*_work.ipynb
```

and are ignored by Git.

Typical workflow:

```bash
git pull

cp notebooks/<lesson>.ipynb \
   notebooks/work/<lesson>_work.ipynb
```

The learner runs and edits only the `_work.ipynb` copy.

This prevents notebook outputs, execution counts, and personal experiments from blocking future `git pull` operations.

---

# Current environment baseline

- Platform: macOS on Apple Silicon
- Architecture target: `arm64`
- Python: `3.12.x`
- Project virtual environment: `.venv`
- Environment manager: `uv`
- Notebook kernel: project `.venv`
- Core framework: Apple MLX
- Editor: VS Code

---

# Curriculum

## Step 1 — Apple Silicon sanity check ✅

**Goal:** verify that the project is running natively on Apple Silicon before installing or benchmarking MLX.

### 1.1 macOS / architecture probe

- `platform.system()`
- `platform.machine()`
- expected: `Darwin`, `arm64`

### 1.2 Python interpreter awareness

- Python executable vs Python source code
- why an x86_64/Rosetta interpreter would be undesirable here

### 1.3 Project path sanity

- verify the repository location
- keep the MiniGPT repository separate from unrelated Python projects

**Repository artifacts:**

- `scripts/step01_system_info.py`
- `docs/step-01-apple-silicon-python.md`

---

## Step 2 — Isolated Python, VS Code, Notebook, and MLX environment ✅

**Goal:** create a reproducible project-specific execution environment.

### 2.1 Project-local virtual environment

- create `.venv`
- pin Python 3.12
- explain why projects should not share unrelated virtual environments

### 2.2 VS Code interpreter

- bind VS Code to `.venv/bin/python`
- distinguish shell activation from editor interpreter selection

### 2.3 Project metadata

- `pyproject.toml`
- `requires-python`
- explain how parent-directory Python metadata can accidentally influence tools

### 2.4 Jupyter / ipykernel

- interpreter vs Notebook kernel
- bind Notebook to the same project `.venv`

### 2.5 MLX / Metal probe

- install MLX
- verify Metal availability
- default MLX device
- first matrix multiplication
- first appearance of `mx.eval()` without yet deeply explaining lazy evaluation

**Repository artifacts:**

- `pyproject.toml`
- `uv.lock`
- `notebooks/01_environment.ipynb`
- `docs/step-02-python-environment.md`

---

## Step 3 — Tensor foundations ✅

**Goal:** learn just enough tensor mechanics to read later Transformer code.

### 3.1 Scalar, vector, matrix, tensor, shape

- dimensionality
- reading shapes

### 3.2 Axis and indexing

- axis numbering
- `x[:, t, :]`
- connect `(B, T, C)` to Batch / Time / Channel

### 3.3 Reshape

- same element count, new logical shape
- product-of-dimensions rule

### 3.4 Reshape vs transpose

- regrouping vs axis permutation
- preview multi-head attention shape changes

### 3.5 Broadcasting

- implicit logical expansion
- `(B,T,C) + (T,C)`

### 3.6 Broadcasting compatibility rules

- compare shapes from right to left
- equal / one / missing-dimension rule

### 3.7 Reduction, `axis=-1`, and `keepdims=True`

- mean/sum as reductions
- negative axis indexing
- keeping a size-1 dimension for later broadcasting
- direct preparation for RMSNorm

**Repository artifacts:**

- `notebooks/02_tensor_basics.ipynb`
- `notebooks/03_reshape.ipynb`
- `notebooks/04_transpose.ipynb`
- `notebooks/05_broadcasting.ipynb`
- `notebooks/06_broadcasting_rules.ipynb`
- `notebooks/07_reduction_keepdims.ipynb`
- `docs/step-03-tensor-basics.md`

---

## Step 4 — From matrix multiplication to neural networks ✅

**Goal:** turn familiar matrix multiplication into the basic learnable components used inside a Transformer.

### 4.1 Linear layer ✅

- `y = x @ W.T`
- input features → output features
- feature mixing as weighted combinations
- batch use of the same weight matrix
- `nn.Linear(in_features, out_features)`
- MLX weight storage shape

### 4.2 Parameters, bias, and `nn.Module` ✅

- what a trainable parameter is
- fixed constants vs trainable arrays
- why bias exists geometrically
- what `nn.Module` provides
- how MLX discovers and organizes parameter trees

### 4.3 Non-linearity: ReLU and GELU ✅

- why stacked Linear layers collapse into one Linear map without activation
- simple function-composition example
- ReLU intuition
- GELU intuition and why GPT-like models use smooth nonlinear activations
- do not overclaim that GELU alone guarantees training stability

### 4.4 Build the first MLP ✅

- `C → 4C → GELU → C`
- expansion as a larger feature workspace
- per-token feature transformation
- contrast with Attention, which mixes information across positions
- implement a small MLX MLP and trace shapes

**Expected final code connection:** MiniGPT feed-forward sublayer.

---

## Step 5 — How a neural network actually learns ✅

**Goal:** see a parameter change because a prediction was wrong.

### 5.1 Prediction, target, and loss ✅

- model output vs desired output
- squared error with tiny scalar examples
- why training needs a single objective value

### 5.2 Minimal calculus for learning ✅

- derivative as local sensitivity / slope
- finite-difference intuition
- derivative of simple scalar expressions
- chain rule introduced only as needed

### 5.3 Gradient and automatic differentiation ✅

- scalar loss with many parameters
- gradient as one derivative per parameter
- reverse-mode intuition
- first `mx.grad` / `mx.value_and_grad` experiment on a tiny function

### 5.4 Gradient descent loop ✅

- learning rate
- `parameter = parameter - lr * gradient`
- tiny manual SGD loop
- verify that loss decreases

### 5.5 What is and is not “learning” ✅

- parameters change; architecture does not
- data + loss + optimizer define the training signal
- prepare for later language-model training

---

## Step 6 — From text to a next-token learning problem ✅

**Goal:** convert raw text into a supervised sequence prediction task.

### 6.1 Vocabulary and character tokenizer ✅

- unique characters
- token IDs
- `stoi` / `itos`
- encode / decode

### 6.2 One-hot vs learned embedding ✅

- discrete identity vs dense learned representation
- correct misconception: embedding dimension does not need to exceed vocabulary size

### 6.3 Next-token shift ✅

- input `x`
- target `y` shifted by one position
- why each position supplies a training target

### 6.4 Context window / block size ✅

- fixed training windows
- sequence length `T`
- what context means in an autoregressive model

### 6.5 Sampling batches ✅

- random windows for the teaching model
- reproducibility with seeds
- compare toy random sampling with industrial shuffling / packing / deterministic resume

---

## Step 7 — Embeddings and sequence representations ✅

**Goal:** turn token IDs into the `(B,T,C)` hidden representation used throughout the model.

### 7.1 `nn.Embedding` as learned lookup table ✅

- lookup, not matrix multiplication at the API level
- embedding weight shape
- gathering rows by token ID

### 7.2 Embedding geometry ✅

- vectors, Euclidean distance, dot product, cosine similarity
- do not assume human-interpretable clusters must emerge
- embeddings optimize next-token loss, not visualization aesthetics

### 7.3 Build `(B,T,C)` ✅

- Batch
- Time
- Channel / hidden dimension
- why `C` is the model width

### 7.4 Position information ✅

- why recurrence naturally encodes order
- why unmasked self-attention without positional signals is permutation-equivariant
- causal-mask caveat: mask itself introduces order-dependent visibility
- learned positional embedding for this teaching implementation

### 7.5 Token + position addition ✅

- broadcasting `(B,T,C) + (T,C)`
- why addition is used instead of concatenation in this MiniGPT
- limitations of learned absolute positions

---

## Step 8 — RNN: the first explicit neural sequence memory 🚧

**Goal:** understand why recurrent neural networks were invented and what recurrence buys us.

### 8.1 Why ordinary MLPs are awkward for variable-length sequences ✅

- order matters
- history is missing
- fixed-size input limitation

### 8.2 Vanilla RNN equation 🚧

- `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`
- tiny hand-calculated example
- hidden state as evolving sequence memory

### 8.3 Time loop implementation

- `for t in range(T)`
- `x[:, t, :]`
- Batch parallelism vs Time dependency

### 8.4 Small MLX RNN cell

- implement a minimal teaching RNN
- trace `x_t`, `h_{t-1}`, `h_t` shapes
- use shared parameters across time

### 8.5 What RNN solves and what it costs

- sequence order becomes natural
- state carries history
- computation along time is inherently sequential

---

## Step 9 — BPTT, vanishing gradients, LSTM, and GRU

**Goal:** understand why vanilla RNNs struggle with long-range dependency and how gating was introduced.

### 9.1 Backpropagation through time

- unfold the recurrent graph
- chain rule across many time steps
- why products of derivatives appear

### 9.2 Vanishing and exploding gradients

- repeated multiplication examples such as `0.5^n` and `1.5^n`
- long-range credit assignment
- gradient clipping as a partial engineering remedy

### 9.3 LSTM cell state and gates

- forget gate
- input gate
- output gate
- cell state as a more direct memory path
- tiny numeric update example

### 9.4 GRU

- merge/simplify LSTM gating
- fewer states and parameters
- conceptual comparison, not a full industrial implementation

### 9.5 Why LSTM/GRU still did not solve everything

- time-step serial dependency remains
- long information paths remain
- GPU parallelism remains constrained

---

## Step 10 — Seq2Seq and the birth of Attention

**Goal:** see Attention emerge as a solution to a real encoder-decoder bottleneck.

### 10.1 Encoder-decoder Seq2Seq

- source sequence
- encoder state
- decoder state
- variable-length input/output

### 10.2 Fixed-vector bottleneck

- compressing a long source into one final state
- why this becomes difficult for long sequences

### 10.3 Bahdanau-style attention intuition

- decoder state asks which encoder states matter now
- content-based lookup
- attention weights as a distribution

### 10.4 Score → softmax → weighted sum

- tiny numeric example
- weighted combination of values
- first precise attention computation without Q/K/V terminology

### 10.5 Historical bridge to Transformer

- Attention initially sat on top of RNNs
- question: if direct retrieval works, why keep recurrence everywhere?

---

## Step 11 — Self-Attention from first principles

**Goal:** derive Q/K/V instead of memorizing the final formula.

### 11.1 Why separate Query, Key, and Value

- one representation serving three roles is restrictive
- learned projections create task-specific lookup spaces

### 11.2 Q/K/V Linear projections

- `X W_Q`, `X W_K`, `X W_V`
- shapes
- small hand-computable matrices

### 11.3 `QK^T` as compatibility score

- dot product
- distinction from cosine similarity unless vectors are normalized
- produce the `T × T` score matrix

### 11.4 Scaling and softmax

- why dot-product variance grows with head dimension
- `1/sqrt(d_k)`
- softmax as normalized positive weights

### 11.5 Weighted Value retrieval

- `A @ V`
- each output position becomes a context-dependent representation
- complete a full tiny single-head hand calculation

---

## Step 12 — Causal self-attention

**Goal:** turn general self-attention into an autoregressive language-model mechanism.

### 12.1 Autoregressive constraint

- predict next token using only present/past context
- future leakage as cheating

### 12.2 Causal mask

- upper triangle
- `-inf`
- why softmax maps masked scores to zero

### 12.3 Stable attention computation

- softmax numerical intuition
- what scaling helps with
- avoid simplistic claim that scaling only exists to “prevent gradient disappearance”

### 12.4 Implement a single causal attention head

- forward pass in MLX
- shape trace
- mask as non-trainable state / constant

### 12.5 Inspect an attention matrix

- print a tiny `T×T` example
- verify future positions receive zero probability

---

## Step 13 — Multi-Head Attention

**Goal:** understand the exact tensor surgery behind GPT-style multi-head attention.

### 13.1 Why multiple heads

- multiple learned subspaces
- avoid assigning fixed human meanings to specific heads

### 13.2 Fused QKV projection

- one `Linear(C, 3C)`
- split Q/K/V
- parameter and shape advantages

### 13.3 Split channels into heads

- `C = H × D`
- `(B,T,C) → (B,T,H,D)`

### 13.4 Transpose for batched attention

- `(B,T,H,D) → (B,H,T,D)`
- batched matrix multiplication
- distinguish logical head independence from literal “one GPU core per head”

### 13.5 Concatenate heads and output projection

- transpose back
- reshape to `(B,T,C)`
- `c_proj` mixes head outputs

---

## Step 14 — Transformer Block: residuals, normalization, and MLP

**Goal:** assemble the full repeated computation unit used in GPT.

### 14.1 Residual connection

- `x + F(x)`
- update-as-correction intuition
- gradient highway intuition
- residual does not mathematically guarantee zero information loss

### 14.2 LayerNorm → RMSNorm historical path

- activation scale problem
- RMS calculation
- epsilon
- learned scale `gamma`
- difference from LayerNorm at a high level

### 14.3 Pre-Norm vs Post-Norm

- where normalization sits
- why modern deep Transformers often use pre-norm variants
- connect directly to our Block code

### 14.4 Feed-forward MLP inside the Block

- revisit `C → 4C → activation → C`
- per-position computation
- Attention = cross-position mixing; MLP = within-position feature transformation

### 14.5 Implement and test one Block

- shape invariants
- no sequence-length change
- parameter count
- numerical sanity checks

---

## Step 15 — From the original Transformer to GPT / Decoder-Only Transformer

**Goal:** explain exactly what GPT keeps and removes from the 2017 Transformer.

### 15.1 Original Transformer architecture

- encoder stack
- decoder stack
- self-attention
- cross-attention

### 15.2 Why autoregressive GPT can be decoder-only

- no separate source encoder
- causal self-attention over the same token stream

### 15.3 MiniGPT top-level architecture

- token embedding
- position embedding
- Block × N
- final norm
- LM head

### 15.4 Implement `MiniGPT`

- clean MLX `nn.Module`
- shape assertions
- avoid accidental trainable constants such as causal masks

### 15.5 Full forward shape trace

- `(B,T) → (B,T,C) → ... → (B,T,V)`
- parameter count
- connect every line to its historical purpose

---

## Step 16 — Tiny Shakespeare data pipeline

**Goal:** replace toy tensors with a real small language-model corpus.

### 16.1 Acquire and inspect Tiny Shakespeare

- download / cache
- encoding
- corpus size

### 16.2 Train/validation split

- why held-out validation exists
- avoid training on validation tokens

### 16.3 Batch sampler

- `block_size`
- shifted targets
- output shapes

### 16.4 Reproducibility

- random seeds
- deterministic sampling expectations
- what must be saved for resume

### 16.5 Toy sampling vs industrial input pipelines

- random windows
- shuffling
- token packing
- distributed shards
- deterministic cursor/state

---

## Step 17 — Language-model loss and the first real training loop

**Goal:** make MiniGPT actually learn next-token prediction.

### 17.1 Logits, probabilities, and cross-entropy

- logits are not probabilities
- softmax
- negative log-likelihood
- one tiny numeric example

### 17.2 Flatten `B×T` supervision

- `(B,T,V) → (B*T,V)`
- `(B,T) → (B*T)`
- one training step contains many token-level targets

### 17.3 MLX `nn.value_and_grad`

- Module parameter tree
- loss + gradients as a function transformation
- contrast with a simplistic “gradient attached to every parameter” mental model

### 17.4 Adam optimizer

- first moment
- second moment
- state
- why optimizer state consumes memory

### 17.5 Uncompiled training step

- forward
- loss
- reverse-mode autodiff
- optimizer update
- explicit evaluation boundary

---

## Step 18 — MLX execution model: lazy evaluation, `mx.eval`, and `mx.compile`

**Goal:** understand what makes MLX distinctive on Apple Silicon.

### 18.1 Lazy evaluation

- operations build a graph
- Python line execution does not necessarily mean immediate GPU execution

### 18.2 `mx.eval` as an execution/materialization boundary

- force required arrays to materialize
- iterative graph growth if state is never evaluated
- why this matters for memory

### 18.3 `mx.compile`

- trace/compile graph
- optimization and fusion intuition
- first-call compile cost
- shape/dtype-dependent cache behavior

### 18.4 Mutable training state under compile

- model state
- optimizer state
- dynamic inputs/outputs
- avoid treating changing parameters as constants

### 18.5 Memory instrumentation

- active memory
- peak memory
- cache memory
- compare uncompiled vs compiled small runs

---

## Step 19 — Stable low-heat training

**Goal:** train the model responsibly on an M-series laptop rather than maximize benchmark throughput.

### 19.1 Hyperparameter baseline

- model width
- layers
- heads
- block size
- batch size
- learning rate

### 19.2 Train / validation loop

- periodic validation loss
- train/eval behavior where relevant

### 19.3 Training stability

- initialization
- gradient norms
- optional gradient clipping
- detect NaN / divergence

### 19.4 Apple Silicon resource control

- unified memory awareness
- avoid Swap
- conservative batch/model sizing
- sustained heat vs short bursts

### 19.5 Baseline training run

- record speed
- memory
- loss curve
- save reproducible baseline numbers

---

## Step 20 — Checkpointing and resume

**Goal:** make training recoverable and reproducible.

### 20.1 Save model weights

- parameter tree serialization
- file organization

### 20.2 Save optimizer state

- Adam moments
- step count

### 20.3 Save configuration and metadata

- architecture
- tokenizer/vocab
- random seed
- training progress

### 20.4 Resume training

- reconstruct model
- restore state
- continue from checkpoint

### 20.5 Resume verification

- small deterministic test
- catch architecture/config mismatch

---

## Step 21 — Autoregressive generation

**Goal:** turn trained logits into text.

### 21.1 Generation loop

- feed context
- take last-position logits
- choose next token
- append and repeat

### 21.2 Greedy vs sampling

- argmax
- categorical sampling
- diversity vs determinism

### 21.3 Temperature

- logit scaling
- low vs high temperature intuition

### 21.4 Top-k sampling

- restrict candidate set
- renormalize and sample

### 21.5 Context cropping

- respect `block_size`
- explain why naïve generation repeatedly recomputes the context
- preview KV cache

---

## Step 22 — Modern GPT improvements beyond our minimal implementation

**Goal:** place the teaching MiniGPT in the context of modern LLM architecture.

### 22.1 Weight tying

- embedding and LM-head relationship
- parameter savings / inductive bias

### 22.2 Learned positions → RoPE

- shortcomings of fixed learned absolute positions
- rotary position encoding intuition

### 22.3 GELU → gated MLPs / SwiGLU

- why many modern LLMs use gated feed-forward blocks

### 22.4 MHA → MQA / GQA and efficient attention

- KV memory cost
- grouping keys/values
- conceptual link to inference efficiency

### 22.5 Flash-style attention and fused kernels

- reduce memory traffic
- distinguish mathematical attention from execution algorithm
- decide which improvements are educational extensions vs core project scope

---

## Step 23 — Refactor notebooks into production-style source files

**Goal:** turn learning prototypes into a maintainable package.

### 23.1 Data module

Target:

```text
src/minigpt/data.py
```

### 23.2 Model module

Target:

```text
src/minigpt/model.py
```

### 23.3 Training module

Target:

```text
src/minigpt/train.py
```

### 23.4 Generation module

Target:

```text
src/minigpt/generate.py
```

### 23.5 Configuration and CLI

- dataclass/config structure
- train/generate commands
- keep notebooks as explanations, not the only executable implementation

---

## Step 24 — Testing, benchmarking, and Apple Silicon profiling

**Goal:** verify correctness and document the project's low-resource behavior.

### 24.1 Unit tests

- tensor shapes
- causal mask
- forward output
- save/load

### 24.2 Tiny overfit test

- deliberately overfit a tiny sequence
- powerful sanity test for the entire learning pipeline

### 24.3 Memory benchmark

- active / peak memory
- model-size and batch-size scaling

### 24.4 Performance / thermal-oriented benchmark

- tokens/sec
- conservative configurations
- no claim of scientific thermal measurement without proper sensors/instrumentation

### 24.5 Benchmark table for README

- hardware
- Python / MLX version
- model configuration
- memory and throughput

---

## Step 25 — GitHub-ready release

**Goal:** finish the repository as a reproducible educational open-source project.

### 25.1 Rewrite main README

- project purpose
- architecture diagram
- quick start
- learning path
- Apple Silicon focus

### 25.2 Documentation index

- link every Step
- historical timeline
- mathematical prerequisites

### 25.3 Reproducibility audit

- fresh clone
- `uv sync`
- notebook kernel instructions
- train and generate smoke test

### 25.4 Credits / references / license

- MLX
- foundational papers
- Tiny Shakespeare source
- MIT license consistency

### 25.5 Final release

- clean `main`
- release tag
- final repository structure
- optional GitHub release notes

---

# Optional extension track after the core course

These are not required to declare the MiniGPT project complete.

## Extension A — Better tokenization

- byte-level tokenizer
- BPE
- vocabulary-size tradeoffs

## Extension B — RoPE implementation

- complex/rotation intuition
- replace learned absolute position embeddings

## Extension C — KV cache

- why autoregressive inference otherwise recomputes old K/V
- implement a small cache
- memory/speed tradeoff

## Extension D — Mixed precision and quantization

- fp32 / fp16 / bf16 intuition
- inference quantization
- avoid conflating inference quantization with training compression

## Extension E — Scale-up experiments

- larger hidden size / more layers
- safe Apple Silicon limits
- compare learning quality vs memory / heat / speed

---

# Current progress

As of the latest curriculum update:

```text
Step 1   ✅ Complete
Step 2   ✅ Complete
Step 3   ✅ Complete
Step 4.1 ✅ Complete / taught
Step 4.2 ✅ Complete
Step 4.3 ✅ Complete
Step 4.4 ✅ Complete
Step 5.1 ✅ Complete
Step 5.2 ✅ Complete
Step 5.3 ✅ Complete
Step 5.4 ✅ Complete
Step 5.5 ✅ Complete
Step 6.1 ✅ Complete
Step 6.2 ✅ Complete
Step 6.3 ✅ Complete
Step 6.4 ✅ Complete
Step 6.5 ✅ Complete
Step 7.1 ✅ Complete
Step 7.2 ✅ Complete
Step 7.3 ✅ Complete
Step 7.4 ✅ Complete
Step 7.5 ✅ Complete
Step 8.1 ✅ Complete
Step 8.2 🚧 Current
```

Current lesson:

> **Step 8.2 — Vanilla RNN equation**
