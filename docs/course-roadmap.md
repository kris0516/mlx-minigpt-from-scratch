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

- a major Step will **usually contain about five subsections**, but this is a pacing convention rather than a hard minimum or maximum;
- the assistant chooses the natural grouping needed by the subject instead of forcing content into exactly five pieces;
- advance only **one numbered subsection per teaching turn** unless the learner explicitly asks for a different cadence;
- each subsection should be neither artificially short nor overloaded: include enough motivation, prerequisite reasoning, examples, shapes, caveats, and code to make the idea reconstructable later;
- do not split every API or formula into its own micro-step, and do not compress prerequisite explanation merely to satisfy a target count;
- when geometry, optimization behavior, distributions, embeddings, loss curves, or performance trends are easier to understand visually, include a small plot and explain the plotting technique used.

Step 3 retains its historical seven-subsection structure; later Steps may naturally vary around the usual five-subsection scale.

### Material review rule

- each lesson/subsection gets at most **two correction rounds** after its first prepared version;
- actual correction counts persist across conversations; starting a new session does not reset them;
- review focuses on substantive correctness, obvious prerequisite gaps, basic usability, and whether the material was actually published to the repository;
- do not spend extra correction rounds on minor formatting, wording, or stylistic preferences once the material is otherwise usable;
- a static audit or correction limit describes material quality control only; it never implies learner mastery or exercise completion.

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

Typical file workflow is:

```text
GitHub master lesson committed
→ git pull
→ create notebooks/work/<lesson>_work.ipynb
→ learner runs/edits only the work copy
```

### Interactive command protocol

For the normal post-setup lesson workflow, give the learner the two routine commands **at the beginning of the lesson response, as two separate shell code blocks**:

1. `git pull`
2. `cp notebooks/<lesson>.ipynb notebooks/work/<lesson>_work.ipynb`

Do not combine them into one shell block. Normally do **not** require the learner to paste terminal output between these two routine commands. Only switch to one-command-at-a-time troubleshooting if a command fails, Git reports a conflict, the work copy already exists in a way that matters, or another abnormal state appears.

Do not add recurring setup commands such as `mkdir -p notebooks/work`; the directory already exists. Use the normal copy command unless troubleshooting or the learner's explicit request requires another form. For the current continuation, the learner explicitly requests a new filename and protection against overwriting notes: use a timestamped `*_work.ipynb` name with `cp -n`.

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

## Step 8 — RNN: the first explicit neural sequence memory ✅

**Goal:** understand why recurrent neural networks were invented and what recurrence buys us.

### 8.1 Why ordinary MLPs are awkward for variable-length sequences ✅

- order matters
- history is missing
- fixed-size input limitation

### 8.2 Vanilla RNN equation ✅

- `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`
- tiny hand-calculated example
- hidden state as evolving sequence memory

### 8.3 Time loop implementation ✅

- `for t in range(T)`
- `x[:, t, :]`
- Batch parallelism vs Time dependency

### 8.4 Small MLX RNN cell ✅

- implement a minimal teaching RNN
- trace `x_t`, `h_{t-1}`, `h_t` shapes
- use shared parameters across time

### 8.5 What RNN solves and what it costs ✅

- sequence order becomes natural
- state carries history
- computation along time is inherently sequential

---

## Step 9 — BPTT, vanishing gradients, LSTM, and GRU ✅

**Goal:** understand why vanilla RNNs struggle with long-range dependency and how gating was introduced.

### 9.1 Backpropagation through time ✅

- unfold the recurrent graph
- chain rule across many time steps
- why products of derivatives appear

### 9.2 Vanishing and exploding gradients ✅

- repeated multiplication examples such as `0.5^n` and `1.5^n`
- long-range credit assignment
- gradient clipping as a partial engineering remedy

### 9.3 LSTM cell state and gates ✅

- forget gate
- input gate
- output gate
- cell state as a more direct memory path
- tiny numeric update example

### 9.4 GRU ✅

- merge/simplify LSTM gating
- fewer states and parameters
- conceptual comparison, not a full industrial implementation

### 9.5 Why LSTM/GRU still did not solve everything ✅

- time-step serial dependency remains
- long information paths remain
- GPU parallelism remains constrained

---

## Step 10 — Seq2Seq and the birth of Attention ✅

**Goal:** see Attention emerge as a solution to a real encoder-decoder bottleneck.

### 10.1 Encoder-decoder Seq2Seq ✅

- source sequence
- encoder state
- decoder state
- variable-length input/output

### 10.2 Fixed-vector bottleneck ✅

- compressing a long source into one final state
- why this becomes difficult for long sequences

### 10.3 Bahdanau-style attention intuition ✅

- decoder state asks which encoder states matter now
- content-based lookup
- attention weights as a distribution

### 10.4 Score → softmax → weighted sum ✅

- tiny numeric example
- weighted combination of values
- first precise attention computation without Q/K/V terminology

### 10.5 Historical bridge to Transformer ✅

- Attention initially sat on top of RNNs
- question: if direct retrieval works, why keep recurrence everywhere?

---

## Step 11 — Self-Attention from first principles ✅

**Goal:** derive Q/K/V instead of memorizing the final formula.

### 11.1 Why separate Query, Key, and Value ✅

- one representation serving three roles is restrictive
- learned projections create task-specific lookup spaces

### 11.2 Q/K/V Linear projections ✅

- `X W_Q`, `X W_K`, `X W_V`
- shapes
- small hand-computable matrices

### 11.3 `QK^T` as compatibility score ✅

- dot product
- distinction from cosine similarity unless vectors are normalized
- produce the `T × T` score matrix

### 11.4 Scaling and softmax ✅

- why dot-product variance grows with head dimension
- `1/sqrt(d_k)`
- softmax as normalized positive weights

### 11.5 Weighted Value retrieval ✅

- `A @ V`
- each output position becomes a context-dependent representation
- complete a full tiny single-head hand calculation

---

## Step 12 — Causal self-attention ✅

**Goal:** turn general self-attention into an autoregressive language-model mechanism.

### 12.1 Autoregressive constraint ✅

- predict next token using only present/past context
- future leakage as cheating

### 12.2 Causal mask ✅

- upper triangle
- `-inf`
- why softmax maps masked scores to zero

### 12.3 Stable attention computation ✅

- softmax numerical intuition
- what scaling helps with
- avoid simplistic claim that scaling only exists to “prevent gradient disappearance”

### 12.4 Implement a single causal attention head ✅

- forward pass in MLX
- shape trace
- mask as non-trainable state / constant

### 12.5 Inspect an attention matrix ✅

- print a tiny `T×T` example
- verify future positions receive zero probability

---

## Step 13 — Multi-Head Attention ✅

**Goal:** understand the exact tensor surgery behind GPT-style multi-head attention.

### 13.1 Why multiple heads ✅

- multiple learned subspaces
- avoid assigning fixed human meanings to specific heads

### 13.2 Fused QKV projection ✅

- one `Linear(C, 3C)`
- split Q/K/V
- parameter and shape advantages

### 13.3 Split channels into heads ✅

- `C = H × D`
- `(B,T,C) → (B,T,H,D)`

### 13.4 Transpose for batched attention ✅

- `(B,T,H,D) → (B,H,T,D)`
- batched matrix multiplication
- distinguish logical head independence from literal “one GPU core per head”

### 13.5 Concatenate heads and output projection ✅

- transpose back
- reshape to `(B,T,C)`
- `c_proj` mixes head outputs

---

## Step 14 — Transformer Block: residuals, normalization, and MLP ✅

**Goal:** assemble the full repeated computation unit used in GPT.

### 14.1 Residual connection ✅

- `x + F(x)`
- update-as-correction intuition
- gradient highway intuition
- residual does not mathematically guarantee zero information loss

### 14.2 LayerNorm → RMSNorm historical path ✅

- activation scale problem
- RMS calculation
- epsilon
- learned scale `gamma`
- difference from LayerNorm at a high level

### 14.3 Pre-Norm vs Post-Norm ✅

- where normalization sits
- why modern deep Transformers often use pre-norm variants
- connect directly to our Block code

### 14.4 Feed-forward MLP inside the Block ✅

- revisit `C → 4C → activation → C`
- per-position computation
- Attention = cross-position mixing; MLP = within-position feature transformation

### 14.5 Implement and test one Block ✅

- shape invariants
- no sequence-length change
- parameter count
- numerical sanity checks

---

## Step 15 — From the original Transformer to GPT / Decoder-Only Transformer ✅

**Goal:** explain exactly what GPT keeps and removes from the 2017 Transformer.

### 15.1 Original Transformer architecture ✅

- encoder stack
- decoder stack
- self-attention
- cross-attention

### 15.2 Why autoregressive GPT can be decoder-only ✅

- no separate source encoder
- causal self-attention over the same token stream

### 15.3 MiniGPT top-level architecture ✅

- token embedding
- position embedding
- Block × N
- final norm
- LM head

### 15.4 Implement `MiniGPT` ✅

- clean MLX `nn.Module`
- shape assertions
- avoid accidental trainable constants such as causal masks

### 15.5 Full forward shape trace ✅

- `(B,T) → (B,T,C) → ... → (B,T,V)`
- parameter count
- connect every line to its historical purpose

---

## Step 16 — Tiny Shakespeare data pipeline ✅

**Goal:** replace toy tensors with a real small language-model corpus.

All five subsection masters, documents, and full replies have independently passed the material reviewer's core static review. This is material acceptance, not learner completion, learning feedback, or runtime validation. Preserve the passed lessons and their accumulated correction counts.

### 16.1 Acquire and inspect Tiny Shakespeare ✅

- download / cache
- encoding
- corpus size

### 16.2 Train/validation split ✅

- why held-out validation exists
- avoid training on validation tokens

### 16.3 Batch sampler ✅

- `block_size`
- shifted targets
- output shapes

### 16.4 Reproducibility ✅

- random seeds
- deterministic sampling expectations
- what must be saved for resume

### 16.5 Toy sampling vs industrial input pipelines ✅

- random windows
- shuffling
- token packing
- distributed shards
- deterministic cursor/state

---

## Step 17 — Language-model loss and the first real training loop ✅

**Goal:** make MiniGPT actually learn next-token prediction.

### 17.1 Logits, probabilities, and cross-entropy ✅

- logits are not probabilities
- softmax
- negative log-likelihood
- one tiny numeric example

### 17.2 Flatten `B×T` supervision ✅

- `(B,T,V) → (B*T,V)`
- `(B,T) → (B*T)`
- one training step contains many token-level targets

### 17.3 MLX `nn.value_and_grad` ✅

- Module parameter tree
- loss + gradients as a function transformation
- contrast with a simplistic “gradient attached to every parameter” mental model

### 17.4 Adam optimizer ✅

- first moment
- second moment
- state
- why optimizer state consumes memory

### 17.5 Uncompiled training step ✅

- forward
- loss
- reverse-mode autodiff
- optimizer update
- explicit evaluation boundary

---

## Step 18 — MLX execution model: lazy evaluation, `mx.eval`, and `mx.compile` ✅

**Goal:** understand what makes MLX distinctive on Apple Silicon.

### 18.1 Lazy evaluation ✅

- operations build a graph
- Python line execution does not necessarily mean immediate GPU execution

### 18.2 `mx.eval` as an execution/materialization boundary ✅

- force required arrays to materialize
- iterative graph growth if state is never evaluated
- why this matters for memory

### 18.3 `mx.compile` ✅

- trace/compile graph
- optimization and fusion intuition
- first-call compile cost
- shape/dtype-dependent cache behavior

### 18.4 Mutable training state under compile ✅

- model state
- optimizer state
- dynamic inputs/outputs
- avoid treating changing parameters as constants

### 18.5 Memory instrumentation ✅

- active memory
- peak memory
- cache memory
- compare uncompiled vs compiled small runs

---

## Step 19 — Stable low-heat training 🚧

**Goal:** train the model responsibly on an M-series laptop rather than maximize benchmark throughput.

### 19.1 Hyperparameter baseline 🚧

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

# Current course-material status

> Status here tracks **repository lesson-material preparation**, not learner mastery, exercise completion, assessment, data acquisition success, or runtime validation.
> `✅` = master lesson/material has been prepared and its static review status has been confirmed where noted; `🚧` = current prepared lesson/material being used or revised; no icon = future planned lesson.
> Learner completion must only be recorded when the learner explicitly confirms completion or an agreed assessment has actually been completed.

```text
Steps 1–8.5      ✅ Material prepared
Steps 9.1–9.5    ✅ Material prepared / static audit passed
Steps 10.1–10.5  ✅ Material prepared / static audit passed
Steps 11.1–11.5  ✅ Material prepared / static audit passed
Steps 12.1–12.5  ✅ Material prepared / static audit passed
Steps 13.1–13.5  ✅ Material prepared / static audit passed
Steps 14.1–14.5  ✅ Material prepared / static audit passed
Steps 15.1–15.5  ✅ Material prepared / static audit passed (15.5 correction 1/2)
Step 16.1         ✅ Material prepared / static audit passed (correction 1/2)
Step 16.2         ✅ Material prepared / static audit passed (correction 1/2)
Step 16.3         ✅ Material and full-reply core independently passed / corrections 0/2
Step 16.4         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 16.5         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 17.1         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 17.2         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 17.3         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 17.4         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 17.5         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 18.1         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 18.2         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 18.3         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 18.4         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 18.5         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 19.1         ✅ Material reviewer independently confirmed core static PASS / corrections 0/2
Step 19.2         🚧 Current initial material published / corrections 0/2
Step 19.3+         Planned only
```

Current course-material checkpoint:

> **Step 19.5 — Baseline Training Run**

Current master: [86_step19_5_baseline_training_run.ipynb](../notebooks/86_step19_5_baseline_training_run.ipynb). Current Step document: [Step 19 — Stable low-heat training](step-19-stable-low-heat-training.md).

Learner-status note:

> All major Step 16, Step 17, and Step 18 materials have independently passed the material reviewer's core static review. The latest acceptance covers Step 18.5's actual five-file commit `57dc27ee121430b11802b7c6f54fa2617ccd710e`, fixed MLX v0.32.2 interfaces, and full teaching reply, with actual corrections **0/2**. This is material acceptance, not learner mastery or runtime feedback. Step 19.1 prepares one explicit, unmeasured hyperparameter candidate: model_width=64, layers=2, heads=4, block_size=64, batch_size=4, and learning_rate=3e-4. It connects these six choices to the existing MiniGPT, sampler, scalar loss, and Adam interfaces, keeping max_seq_len=block_size as this candidate's convention and vocabulary size as actual tokenizer metadata rather than a guessed constant. Pure Python configuration and argument-mapping helpers are self-contained and do not construct models, optimizers, arrays, batches, or training state. The material distinguishes structural constraints and hand-derived shapes from measurements; the old shape toys are not promoted to tuned training baselines. No low-heat, memory, performance, or convergence result is claimed. This turn stops at Step 19.1, before 19.2. Earlier independent acceptance does not imply acceptance of 19.1, learner completion, mastery, exercise completion, or runtime validation. The major Step 19 title is a course goal, not evidence that this candidate has been measured as stable or low-heat.

Actual correction ledger: Step 16.1 **1/2**, Step 16.2 **1/2**, Step 16.3 **0/2**, Step 16.4 **0/2**, Step 16.5 **0/2**, Step 17.1 **0/2**, Step 17.2 **0/2**, Step 17.3 **0/2**, Step 17.4 **0/2**, Step 17.5 **0/2**, Step 18.1 **0/2**, Step 18.2 **0/2**, Step 18.3 **0/2**, Step 18.4 **0/2**, Step 18.5 **0/2**, Step 19.1 initial version **0/2**. Step 16.2 includes the previously recorded reply completion after a timeout. Preserve the passed lessons and all accumulated counts across turns and conversations.

Active continuation boundary: subsequent requests proceed one subsection at a time through major **Step 20**, then stop. The existing Step 21–25 roadmap remains a future curriculum plan; this boundary does not authorize automatic or batch preparation of later lessons.


### Step 19.2 current boundary

Step 19.2 initial material has been published with corrections **0/2**. It provides a self-contained train/validation loop master using the 19.1 candidate, complete MiniGPT definitions, split-local sampling, fixed validation-window RNG, scalar loss, MLX v0.32.2 train/eval semantics, and the explicit compile/evaluation boundary. Validation does not call gradient transforms or an optimizer; it evaluates current parameters, whereas a training-step loss belongs to the pre-update parameters. The optional corpus acquisition is disabled by default; local cache availability is not assumed. The four-update/two-step validation schedule is a teaching example only, not a measured baseline. All code cells remain unexecuted and output-free. This is material preparation, not learner mastery or runtime validation. Stop before 19.3.


Step 19.2 dependency correction: first actual correction accepted, cumulative count **1/2**. The master now defines the scalar differentiable loss before use and invokes a local corpus bootstrap before model construction. The bootstrap uses the same character mapping to derive `vocab_size`, makes a contiguous 90/10 split, checks both sides have at least `block_size + 1` tokens, and defaults optional acquisition to disabled; missing cache fails clearly without toy fallback. The accepted train/validation loop is unchanged. No runtime verification or learner mastery is claimed. Stop at 19.2.


### Step 19.3 current boundary

Step 19.3 initial material is published with corrections **0/2**. It covers initialization, global gradient norms, optional global gradient clipping, and NaN/Inf/divergence detection. The master is self-contained, includes the complete MiniGPT and local data bootstrap, and leaves corpus acquisition disabled by default. No code was executed and no stability or training result is claimed. Stop before 19.4.


Step 19.3 first minimal correction accepted: cumulative actual correction count **1/2**. Per the fixed MLX v0.32.2 `mlx.utils` namespace, the master imports `tree_flatten` and `tree_map` from `mlx.utils` and uses those names for gradient-tree traversal/mapping. All other accepted stability logic remains unchanged. No code was executed; stop at 19.3 before 19.4.


### Step 19.4 current boundary

Step 19.4 initial material is published with corrections **0/2**. It covers unified memory awareness, Swap/system-memory versus MLX allocator counters, conservative sizing, evaluation boundaries, and sustained thermal load versus short bursts. The master includes a local bootstrap, with optional acquisition disabled by default and a clear missing-cache error. All resource-reading snippets are unexecuted; no training, system measurement, or performance/thermal claim is made. Stop before 19.5.


### Step 19.5 current boundary

Step 19.5 baseline-run material is published with initial correction count **0/2**. It prepares a future protocol for speed, loss curves, MLX allocator memory and reproducible JSON records. The 20-update schedule and measurement settings are candidates only; no training, validation, timing, memory measurement, corpus access, or system changes were performed. Missing corpus cache fails clearly and optional acquisition is disabled by default. All code cells are unexecuted. Stop before Step 20.


Step 19.5 first merge correction accepted: cumulative actual correction count **1/2**. The baseline master restores all referenced config/helpers, fixes the malformed bootstrap source, keeps host evaluation and finite checks outside compilation, uses a host-side stability step that checks/clips before optimizer update, preserves one warm-up plus 20 timed updates, and labels validation after completed updates 6/11/16/21. JSON metadata uses `mx.__version__`, records the clip threshold, and uses exclusive-create mode. No execution or measurement occurred. Stop at 19.5 before Step 20.


Step 19.5 second/final merge correction accepted: cumulative actual correction count **2/2**. The record cell's accidental line-leading commas were removed with indentation and exclusive-create JSON semantics preserved. The setup cell initializes a single-use marker; the protocol checks the marker and Adam step 0 before sampling, marks the run used immediately before the first sample, and does not clear the marker on failure. The baseline uses the uncompiled host-side stability step; timing excludes sampling, X/Y pre-evaluation, validation, and warm-up while including finite checks, clipping, Adam update, and state evaluation. The protocol retains one warm-up plus 20 timed updates and validation labels 6/11/16/21. No code was executed. Stop at 19.5 before Step 20.


## Step 20.1 current boundary

Step 20.1 — Save Model Weights has been published as `notebooks/87_step20_1_save_model_weights.ipynb`, with initial material generation count **0/2**. It covers parameter-tree serialization, MLX v0.32.2 `Module.save_weights`, SafeTensors, and safe output-directory organization. The full MiniGPT definitions are included; the real vocabulary size must be supplied explicitly from the existing token-ID mapping. No model was instantiated, no weights were saved, and no corpus was read/downloaded. Weights-only files do not represent complete optimizer/RNG/data training state. Stop before 20.2.


## Step 20.2 current boundary

Step 20.2 — Save Optimizer State is published as `notebooks/88_step20_2_save_optimizer_state.ipynb`, initial generation count **0/2**. It covers Adam first/second moments, step count, the optimizer state tree, and a future-work-copy NPZ + JSON serialization pattern with exclusive output-directory creation. Model weights and optimizer state remain distinct; hyperparameters, RNG, sampler position, data split and training history are separate recovery information. No model/optimizer was instantiated, and no state was saved or loaded. Stop before 20.3.


Step 20.2 first merge correction accepted: cumulative actual correction count **1/2**. MLX v0.32.2 Adam's zero-dimensional uint64 `step` and current `learning_rate` are MLX array leaves and are serialized in NPZ alongside m/v; JSON records array paths, shapes, dtypes and actual Python scalar leaves. The loader compares each rebuilt leaf against the already-initialized target optimizer state (array identity, shape/dtype, or exact non-array type) before assignment. Adam hyperparameters and scheduler rules remain separate. No code or state serialization was executed. Stop at 20.2.


## Step 20.3 current boundary

Step 20.3 — Save Configuration and Metadata is published as `notebooks/89_step20_3_save_configuration_metadata.ipynb`, initial generation count **0/2**. It records architecture/config, tokenizer/vocabulary fingerprint, initial seeds, training progress, and explicit weight/optimizer artifact references. The tokenizer fingerprint hashes the canonical `stoi` mapping, not just vocabulary size. Metadata output uses a new UTC-stamped directory with exclusive creation. The lesson distinguishes seed values from full RNG state and does not imply metadata alone restores training. No corpus was read/downloaded and no metadata/checkpoint was generated. Stop before 20.4.


Step 20.3 first correction accepted: cumulative actual correction count **1/2**. Tokenizer metadata now preserves the full `stoi` mapping and `itos_by_id` character table in addition to the fingerprint. A reconstruction helper rebuilds and validates the same mapping from saved metadata alone, without a prior kernel or corpus re-inference. No corpus was read and no metadata file was created. Stop at 20.3.


## Step 20.4 current boundary

Step 20.4 — Resume Training is published as `notebooks/90_step20_4_resume_training.ipynb`. This follow-up completes the interrupted initial delivery; cumulative delivery-correction count is 1/2 and no learner/runtime result is implied. The notebook defines metadata/tokenizer/architecture reconstruction, artifact hash/path checks, strict model weight loading, optimizer state restoration, Adam-step/progress matching, and a caller-supplied continuation loop. It includes its sampler dependency but never loads data or executes a restore. Without full RNG/sampler/data state, do not claim exact trajectory reproduction. Stop before 20.5.


Step 20.4 final merge correction accepted: cumulative delivery/correction count **2/2**. Artifact resolution explicitly separates the metadata file path from the original artifact root used for relative references. The master embeds the scalar shifted-target loss, global gradient norm, clipping and finite checks; a fresh gradient transform is bound to the restored model/optimizer. Before sampling it checks Adam step equals `completed_updates`; after every update it requires an exact +1 step and records the actual value. It does not replay warm-up or reuse the old baseline closure. No code or checkpoint was executed, loaded, restored, or trained. Stop at 20.4 before 20.5.


## Step 20.5 current boundary — Step 20 materials complete

Step 20.5 — Resume Verification is published as `notebooks/91_step20_5_resume_verification.ipynb`, initial generation count **0/2**. It prepares small deterministic checks for architecture/config mismatch, parameter-tree equality, same-input logits, optimizer state-tree equality, and Adam step/progress agreement. The fixed input is manually specified; no corpus is read. Tolerance-based agreement is distinguished from bitwise identity, and before/after comparison requires genuine saved reference evidence. All code cells are unexecuted and output-free. With this publication, the **Step 20.1–20.5 teaching-material scope is complete**. This is material completeness only, not checkpoint execution or runtime verification. Stop before Step 21.
