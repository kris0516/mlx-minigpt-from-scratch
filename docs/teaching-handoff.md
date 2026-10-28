# Teaching Handoff

This file is for any future ChatGPT/Codex session continuing the course.

## Read first

Before teaching or editing the project, read:

1. `docs/course-roadmap.md`
2. the most recent Step documentation
3. the corresponding tracked notebook under `notebooks/`

## Current state

Repository lesson-material preparation:

```text
Steps 1–8.5      material prepared
Steps 9.1–9.5    material prepared; static audit passed
Steps 10.1–10.5  material prepared; static audit passed
Steps 11.1–11.5  material prepared; static audit passed
Steps 12.1–12.5  material prepared; static audit passed
Steps 13.1–13.5  material prepared; static audit passed
Steps 14.1–14.5  material prepared; static audit passed
Step 15.1         material prepared; static audit passed
Step 15.2         material prepared; static audit passed
Step 15.3         material prepared; static audit passed
Step 15.4         material prepared; static audit passed
Step 15.5         current material prepared
Step 16           planned only; not current
```

**Material status is not learner status.** No claim is made that the learner has completed the Step 15.5 work notebook, exercises, or mastered the full MiniGPT trace. No runtime validation is asserted.

Do not enter Step 16 until the learner explicitly asks to proceed beyond Step 15.5.

## Non-negotiable teaching constraints

The learner asked for **small, digestible teaching turns**, but not absurdly fragmented micro-steps.

From Step 4 onward:

- a major Step usually has about five subsections, but this is **not** a hard minimum or maximum;
- choose the natural subdivision for the subject rather than forcing exactly five;
- advance only one numbered subsection per teaching turn unless the learner explicitly requests otherwise;
- keep each subsection substantial but digestible: neither artificially brief nor overloaded;
- introduce mathematics only when needed and explain new notation/API before first use;
- assume only simple matrix multiplication as the starting mathematical baseline;
- use small hand-computable examples;
- use plots where they materially improve intuition (e.g. geometry, distributions, loss curves, optimization, benchmarks), and teach the small plotting technique inline without turning the course into a plotting tutorial;
- explain theory → meaning/history → code;
- explain why a design exists, not only how to call an API;
- explicitly correct oversimplifications when needed;
- avoid dumping an entire architecture in one reply.

Review/audit policy:

- each lesson/subsection receives at most **two correction rounds** after its initial prepared version;
- corrections focus on major content correctness, obvious prerequisite discontinuities, basic material usability, and actual repository publication;
- do not keep revising small formatting, wording, or style issues once the lesson is materially usable;
- static audit status never implies learner mastery or exercise completion.

Primary language: Chinese, with important English technical terms included.

## Hardware / environment assumptions

- Apple Silicon Mac
- current teaching machine: M-series Mac with constrained memory relative to server GPUs
- project aims for low heat and low memory
- Python 3.12
- `.venv`
- `uv`
- Apple MLX
- VS Code + Jupyter Notebook

Do not switch the core implementation to PyTorch.

## Notebook workflow

Tracked master notebooks:

```text
notebooks/*.ipynb
```

The learner should not execute or modify them.

Local practice notebooks:

```text
notebooks/work/*_work.ipynb
```

These are ignored by Git.

Normal post-setup lesson workflow:

1. At the beginning of the lesson response, give `git pull` in its own shell code block.
2. Immediately after it, give the lesson-specific `cp notebooks/<lesson>.ipynb notebooks/work/<lesson>_work.ipynb` in a second, separate shell code block.
3. Do not combine the two commands into one code block.
4. Under normal conditions, do not stop and wait for terminal output between these two routine commands.
5. If Git/copying reports an error, conflict, unexpected existing work copy, or other abnormal state, then switch to one-command-at-a-time troubleshooting and wait for output after each diagnostic/fix command.
6. Do not invent recurring setup commands such as `mkdir -p notebooks/work`; the work directory already exists.
7. Use the normal plain `cp` command unless troubleshooting specifically requires another form.

Then the learner works only in the `_work.ipynb` copy.

When adding a lesson:

1. create/update the tracked master notebook;
2. update the relevant `docs/` note;
3. commit to `main`;
4. tell the learner the exact `git pull` and `cp` commands.

## Accuracy requirements

Avoid repeating these earlier oversimplifications:

- embedding width does **not** need to be >= vocabulary size;
- learned embeddings do **not** guarantee neat human-interpretable clusters;
- dot product is not automatically cosine similarity;
- unmasked self-attention without positional signals is permutation-equivariant, but a causal mask itself is position-dependent and breaks full arbitrary-permutation symmetry; do not claim causal masking and positional encoding are the same mechanism;
- attention heads are not literally mapped one-to-one to physical GPU cores;
- a residual connection helps information/gradient flow but does not mathematically guarantee no information loss;
- causal masks should not accidentally be registered as trainable parameters;
- MLX `Linear.weight` storage shape must be described according to the actual MLX API.

## Historical storyline

The course should maintain this causal development chain:

```text
MLP
→ RNN
→ BPTT limitations
→ LSTM/GRU
→ Seq2Seq bottleneck
→ Attention
→ Self-Attention
→ Transformer
→ Decoder-Only GPT
→ modern LLM improvements
```

Each new architecture should be introduced as a response to a concrete limitation of the previous one.

## Final repository target

Expected mature structure:

```text
mlx-minigpt-from-scratch/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── docs/
├── notebooks/
│   ├── <tracked course notebooks>
│   └── work/              # local ignored practice notebooks
├── scripts/
├── src/
│   └── minigpt/
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       └── generate.py
├── tests/
└── checkpoints/           # local / ignored
```

## Immediate next action

Current course-material checkpoint:

> **Step 15.5 — Full forward shape trace**

Step 15.5 material is prepared only. Do not infer learner mastery, exercise completion, or runtime validation, and do not create or teach Step 16 unless the learner explicitly asks to proceed. Step 14's completed static audit still does not imply runtime tests were executed or passed.

## 2026-10-08 continuity re-audit

Before Step 8.1, verified the live main tree, all 26 tracked master Notebooks (01–26), Steps 01–07 documentation, full roadmap, and learner agreements. See `docs/course-continuity-check-2026-10-08.md`. Do not conflate position-wise MLP with a flattened whole-sequence MLP. The former lacks cross-token mixing; the latter can mix tokens but has fixed-length input-width trade-offs.

## Step 9.2 accuracy boundary

Teach vanishing/exploding gradients from temporal derivative/Jacobian products. Use `0.5^n` and `1.5^n` only as scalar toy intuition; for vector RNNs emphasize state-dependent `D_k`, direction-dependent Jacobian action, transpose/order from the column-gradient convention, and that a single matrix can shrink some directions while growing others. Explain L2/global gradient norm before using MLX 0.32.2 `mlx.optimizers.clip_grad_norm`. Clipping mitigates excessive gradient magnitude but does not repair vanishing credit, tanh saturation, or long recurrent paths. Do not enter LSTM/Step 9.3 during Step 9.2 delivery.

## Step 9.3 accuracy boundary

Introduce sigmoid and Hadamard/element-wise multiplication before using LSTM gates. Use standard course notation `i/f/g/o`; `g_t` is a tanh candidate, not a sigmoid gate. Distinguish `h_t` and `c_t` without claiming a strict short-term/long-term split.

When connecting LSTM to BPTT, `diag(f_t)` describes only the **direct cell-state edge contribution** under the local partial derivative. It is not the complete LSTM total Jacobian. LSTM creates a learnable near-preservation path but does not guarantee zero vanishing/exploding gradients or unlimited memory.

For MLX 0.32.2, `nn.LSTM(input_size, hidden_size, bias=True)` accepts batched `(B,T,C)` input and optional `(B,H)` initial hidden/cell states, returning hidden and cell sequences `(B,T,H)`. Packed parameters use `Wx=(4H,C)`, `Wh=(4H,H)`, optional `bias=(4H,)`, split in `i,f,g,o` order.

Do not enter Step 9.4 during Step 9.3 delivery.

## Step 9.4 accuracy boundary

Teach GRU as a gated recurrent alternative with one hidden state, not as a guaranteed-better LSTM. Use the MLX 0.32.2 convention `r/z/n` and make the update-gate direction explicit: `z≈1` retains old hidden state, `z≈0` favors the new candidate.

Reset gate controls the previous-hidden contribution used to construct the candidate; it does not directly erase the hidden state. For BPTT intuition, `diag(z_t)` is only the direct retain-edge contribution, not the full GRU Jacobian.

For MLX 0.32.2, `nn.GRU(input_size, hidden_size, bias=True)` uses packed `Wx=(3H,C)`, `Wh=(3H,H)`, `b=(3H,)`, plus `bhn=(H,)` when bias is enabled, and returns hidden-state sequence `(B,T,H)`.

Do not enter Step 9.5 during Step 9.4 delivery.

## Step 9.5 accuracy boundary

Emphasize that LSTM/GRU gates improve state retention and gradient flow but do not remove recurrence. Distinguish the Time-axis serial dependency from parallelism available across batch rows and within each time step.

Do not claim recurrent models are wholly non-parallel: input-side packed projections can be computed across T, while hidden/cell-dependent state transitions remain sequential.

Describe fixed-width recurrent state as a practical representation bottleneck, not a dimension-count proof of absolute information impossibility. Gating does not shorten a long recurrent information path to constant length.

Use Step 9.5 only to close the recurrent-family limitations. Do not enter encoder-decoder Seq2Seq / Step 10.1 during this lesson.

## Step 10.1 accuracy boundary

Introduce `T_src/T_tgt`, source/target vocabularies, and `<BOS>/<EOS>` before using them. Teach classic recurrent Seq2Seq as an encoder reading the full source, with its final hidden state used to initialize the decoder.

Explain teacher forcing only as the common training-time choice of feeding the previous ground-truth target token; distinguish it from inference, where the decoder feeds back its own generated token.

Keep the shape bridge explicit: encoder `(B,T_src,C_src)->(B,T_src,H)->(B,H)` context state; decoder training path `(B,T_tgt,C_tgt)->(B,T_tgt,H)->(B,T_tgt,V_tgt)`.

Do not analyze the fixed-vector bottleneck in depth or introduce Attention during Step 10.1; those belong to later existing roadmap subsections.

## Step 10.2 accuracy boundary

Define the fixed-vector bottleneck as an encoder→decoder interface bottleneck: the encoder may produce `(B,T_src,H)` states, while classic Seq2Seq exposes only the final `(B,H)` state to the decoder.

Treat fixed H as a practical representation bottleneck, not a dimension-count proof that long histories are mathematically impossible to encode.

Make the long path explicit: source position `j` to decoder position `u` crosses `(T_src-1-j)+(u+1)=T_src+u-j` recurrent transitions in the classic final-state bridge.

Teacher forcing does not change this source interface. Larger H or stronger gated RNNs can mitigate capacity/optimization pressure without changing the decoder's access pattern.

Do not introduce Bahdanau Attention mechanics, scoring, weights, or weighted sums during Step 10.2; those belong to later existing roadmap subsections.

## Step 10.3 accuracy boundary

Teach Attention as dynamic access to the full encoder-state trajectory, motivated directly by the fixed-vector bottleneck. At decoder step `u`, use previous decoder state `h_(u-1)^dec` as the target-side context that is compared with each `h_j^enc` through a learned alignment/compatibility function.

Introduce raw alignment score `e_(u,j)` conceptually, then normalized attention weights `α_(u,j)` as a nonnegative source-position distribution summing to 1. Do not present attention weights as guaranteed human explanations.

Emphasize that different decoder steps can produce different source-position distributions and therefore different attention context vectors `c_u`.

Do not derive softmax, compute score→softmax numerically, or perform the weighted-sum arithmetic in Step 10.3; those are reserved for Step 10.4. Do not introduce Q/K/V terminology here.

## Step 10.4 accuracy boundary

Start from the already-introduced raw alignment scores `e_(u,j)`. Define softmax before using `mx.softmax`, explain exponential positivity and normalization, and make clear that softmax runs across source positions for each decoder step.

Use the tiny scores `[2,1,0]` example to obtain approximately `[0.665,0.245,0.090]`, then combine those weights with simple 2D encoder states to obtain a concrete context vector.

Keep shape logic explicit: one-step scores/weights `(B,T_src)`, encoder states `(B,T_src,H_enc)`, `weights[...,None]` -> `(B,T_src,1)`, weighted states `(B,T_src,H_enc)`, sum over source axis -> context `(B,H_enc)`.

MLX 0.32.2 supports `mx.softmax(scores, axis=-1)`. Do not introduce Q/K/V terminology or the Transformer historical bridge during Step 10.4; those belong to later roadmap material.

## Step 10.5 accuracy boundary

Teach the historical bridge without jumping into Self-Attention mathematics. Bahdanau-style Attention still sits on recurrent encoder/decoder states; it improves source access but does not itself remove recurrence.

Use Attention's direct content-based retrieval as the conceptual motivation for asking whether positions inside the same sequence can interact directly instead of relying mainly on recurrent propagation.

Explain the two structural motivations carefully: shorter conceptual dependency paths and better training-time parallelism across sequence positions when recurrent state transitions are removed.

Preserve the inference boundary: autoregressive generation remains token-by-token even when the model architecture has no recurrent state transition.

Modern terminology may distinguish encoder–decoder/cross-attention from self-attention, but Step 10.5 must not derive Q/K/V or Self-Attention score matrices. Stop before Step 11.1.

## Step 11.1 accuracy boundary

Start from `X=(B,T,C)` and explain Q/K/V as functional roles before introducing any projection matrices. Query means what the current position is looking for; Key means how a candidate advertises match-relevant features; Value means what content the candidate contributes if selected.

Do not claim Q/K/V separation is mathematically mandatory. A tied design can work, but it couples asking, matching, and payload roles in one representation space and is therefore more restrictive.

Make clear that `self` means the base representations for Query/Key/Value come from the same sequence, not that Q/K/V must be numerically equal. Use Cross-Attention as a contrast: Query may come from decoder representations while Key/Value come from encoder representations.

Step 11.1 may mention learned role-specific views conceptually, but must not derive the Q/K/V linear-projection equations, their exact projection shapes, or any Query–Key compatibility matrix. Stop before Step 11.2.

## Step 11.2 accuracy boundary

Teach Q/K/V as three independent trainable linear projections of the same base `X=(B,T,C)`. With the row-vector math convention, use `W_Q/W_K:(C,d_k)` and `W_V:(C,d_v)`, producing Q/K `(B,T,d_k)` and V `(B,T,d_v)`.

Make clear that the projection changes the feature axis and is shared across sequence positions; it does not reduce or aggregate the T axis.

Use a small hand-computable matrix example before the MLX API. For MLX 0.32.2, `nn.Linear(input_dims, output_dims)` stores weight as `(output_dims,input_dims)` and computes `x @ weight.T (+ bias)`.

Do not compute Query–Key compatibility scores, dot products, score matrices, scaling, or softmax during Step 11.2. Stop before Step 11.3.

## Step 11.3 accuracy boundary

Introduce dot product before matrix `QK^T`. For `q_i,k_j∈R^(d_k)`, define `s_(i,j)=q_i·k_j`, and make clear this is a raw compatibility score rather than a probability.

Explicitly distinguish dot product from cosine similarity: cosine divides by both vector norms, while raw dot product is sensitive to magnitude as well as direction. They are numerically equal only for unit-normalized vectors.

Keep matrix semantics exact: with Q/K `(T,d_k)`, `QK^T` is `(T,T)` and entry `(i,j)` is Query i against Key j. Batch shape is `(B,T,d_k) @ (B,d_k,T) -> (B,T,T)`.

Do not introduce scaling, `1/sqrt(d_k)`, or softmax during Step 11.3. Stop before Step 11.4.

## Step 11.4 accuracy boundary

Explain scaling from a clearly labelled statistical teaching assumption: independent, zero-mean, unit-variance Q/K components imply `Var(q·k)≈d_k` and standard deviation about `sqrt(d_k)`. Do not present this as an exact law for arbitrary trained representations.

Define scaled scores as `QK^T / sqrt(d_k)` and make clear scaling changes magnitude but not `(B,T,T)` shape or score ordering under a positive scalar.

Softmax must normalize each Query row across the Key-position axis. For scores `(B,T,T)`, use `mx.softmax(..., axis=-1)` and interpret `A[b,i,:]` as Query i's distribution over Keys.

Step 11.4 ends with attention weights A. Do not multiply by V, derive the final context/output, or complete the full single-head calculation until Step 11.5.

## Step 11.5 accuracy boundary

Start from attention weights `A:(B,T,T)` and Values `V:(B,T,d_v)`. Explain `o_i=sum_j A_(i,j)v_j` before matrix `O=A@V`, and keep the output width tied to `d_v` rather than `d_k`.

Complete one tiny single-head calculation end-to-end from X through Q/K/V, raw scores, scaling, softmax, A, and finally O. Keep the numbers small enough to hand-check.

Emphasize the semantic split: Q/K determine where/how strongly to retrieve; V carries the payload that is actually mixed into each output representation.

Step 11.5 must stop at general self-attention. Do not introduce causal masking, autoregressive visibility rules, Multi-Head Attention, output projection, residual connections, or Transformer blocks. Stop before Step 12.1.

## Step 12.1 accuracy boundary

Use shifted next-token indexing consistently: input position `t` uses `x_0...x_t` to predict `x_(t+1)`, so the current input token is legal context. Do not mix this with the alternate convention of predicting `x_t` from `x_<t` without explaining the shift.

Define autoregressive modeling as conditioning each predicted token only on its prefix. This is an information/probability constraint, not a requirement to use an RNN.

For self-attention, rows are Query positions and columns are Key positions. The causal visibility rule is `j<=i`; future `j>i` is disallowed. Explain future leakage as training-time cheating that exposes information unavailable during real generation.

Step 12.1 may illustrate allowed prefix positions, but must not construct the causal mask tensor, introduce upper-triangle `-inf`, or explain masked softmax. Stop before Step 12.2.

## Step 12.2 accuracy boundary

Map the Step 12.1 rule `j<=i` onto the score matrix: future positions `j>i` are the strict upper triangle, while diagonal `j=i` remains legal.

Teach additive masking as `0` for allowed entries and `-inf` for future entries, added to scaled scores before softmax. Explain `exp(-inf)=0` as the reason future attention weights become zero.

MLX 0.32.2 supports `mx.triu(x,k=1)` and `mx.where(condition,x,y)`; a `(T,T)` causal mask can broadcast over `(B,T,T)` scores.

Do not turn Step 12.2 into the later stability lesson or full-head implementation. Stable-softmax details, numerical-stability analysis, complete causal attention forward pass, and matrix inspection belong to Steps 12.3–12.5.

## Step 12.3 accuracy boundary

Teach stable softmax from its shift invariance: subtracting the same constant from all logits in a row leaves probabilities unchanged. Choose the row maximum so shifted logits are <=0 and exponentials are <=1.

Use a large-logit example such as `[1000,999,998]` to distinguish mathematical correctness from finite-precision overflow. Explain why very negative shifted logits underflowing toward zero is usually more manageable in softmax than positive overflow producing inf/NaN.

Keep causal masking compatible with the stability story: `-inf` entries remain `-inf` after subtracting a finite row maximum; causal diagonal ensures at least one finite allowed score per row.

Do not describe `1/sqrt(d_k)` as existing only to prevent vanishing gradients. Scaling primarily controls dot-product logit scale / softmax saturation and improves optimization conditions; subtract-max addresses exponential overflow. Stop before the complete causal-attention-head implementation in Step 12.4.

## Step 12.4 accuracy boundary

Implement one MLX `nn.Module` that joins the already-taught components in the correct order: Q/K/V projections, `QK^T`, `1/sqrt(d_h)` scaling, causal masking before softmax, stable `mx.softmax(...,axis=-1)`, then weighted Value retrieval.

Use a clear shape trace: input `(B,T,C)`, Q/K/V `(B,T,d_h)`, scores/weights `(B,T,T)`, output `(B,T,d_h)`. This lesson may set `d_k=d_v=d_h` for a minimal single-head implementation.

Treat the causal mask as structural, non-trainable information. The primary lesson implementation should construct it locally in forward so it does not enter the MLX parameter tree. If discussing a cached `self.causal_mask`, note that MLX module array attributes are parameters by default and must be explicitly frozen to stay out of `trainable_parameters()`.

Do not add Multi-Head Attention, concatenation, output projection, residual/normalization, Transformer blocks, or dedicated attention-matrix inspection. Stop before Step 12.5.

## Step 12.5 accuracy boundary

Teach matrix inspection with rows as Queries and columns as Keys. Verify the two causal invariants: each Query row sums to 1, and all strict-upper-triangle entries `j>i` are zero.

Use a deterministic hand-checkable case separately from the actual module. Equal legal logits can produce the exact expected lower-triangular uniform matrix `[1]`, `[1/2,1/2]`, `[1/3,1/3,1/3]`, etc. Make clear this is a designed expectation, not a generic learned-attention pattern.

For the Step 12.4 head, a diagnostic variant may return attention weights in addition to the output. In the Step 12.5 diagnostic example specifically, the input is all zeros and Q/K/V projections use `bias=False`; therefore Q=K=V=0 regardless of random weights, legal positions are exactly uniform within each causal prefix, future weights are zero, and the output is zero. Treat these as mathematically derived expected values until the notebook is actually executed, not as observed runtime results.

Optional heatmap code may visualize the matrix with Key positions on x and Query positions on y. Do not infer fixed human-semantic meaning from bright attention cells without additional evidence.

Stop after Step 12.5. Do not motivate, teach, or implement Multi-Head Attention / Step 13.1 in this lesson.

## Step 13.1 accuracy boundary

Start from the completed single causal head. Explain that one head owns one learned Q/K/V projection set, while multiple heads provide multiple independently learned interaction subspaces / feature views.

Do not claim a single head is mathematically incapable of representing multiple relations; frame multiple heads as increasing representational flexibility and reducing pressure on one shared projection space.

Do not preassign human meanings to heads. Examples such as syntax, pronouns, or position may only be discussed as possible observed behaviors with evidence, never as architectural identities.

A hand-designed two-head toy example may show that different projection matrices produce different causal attention patterns. State clearly that the hand-selected projections are for intuition only and real head projections are learned.

Stop before Step 13.2. Do not introduce fused `Linear(C,3C)`, QKV splitting, `C=H×D`, head reshaping/transposition, concatenation, or output projection.

## Step 13.2 accuracy boundary

Start from three already-known full-width projections `W_Q/W_K/W_V:(C,C)`. Explain that horizontally concatenating them gives a mathematical fused matrix `W_QKV:(C,3C)`, so one projection produces `(B,T,3C)`.

For MLX 0.32.2, `nn.Linear(C,3C)` stores weight as `(3C,C)`. Use `mx.split(qkv, indices_or_sections=3, axis=-1)` to recover Q/K/V, each `(B,T,C)`.

Do not claim fusion reduces parameter count: with the same bias choice, three `Linear(C,C)` layers and one `Linear(C,3C)` have identical trainable parameter counts (`3C^2` weights, plus `3C` biases if enabled). Fusion is primarily parameter packing / compute organization.

Do not overpromise a fixed runtime speedup; one larger projection can reduce dispatch/call overhead and organize matrix multiplication efficiently, but actual performance is implementation- and hardware-dependent.

Stop before Step 13.3. Do not split C into `H×D`, reshape to `(B,T,H,D)`, transpose head axes, concatenate heads, or introduce output projection.

## Step 13.3 accuracy boundary

Start from full-width Q/K/V tensors `(B,T,C)`. For equal-width heads define `C=H×D`, require `C % H == 0`, and set `D=C//H`. Clarify locally that H means number of heads, not the earlier RNN hidden-width notation.

Teach the exact index interpretation `Q_heads[b,t,h,d] = Q[b,t,hD+d]` before code. The reshape to `(B,T,H,D)` reorganizes channels; it does not add parameters, perform another learned projection, average values, or create extra elements.

Use MLX 0.32.2 `mx.reshape(q,(B,T,H,D))` for Q/K/V. Do not claim the operation is guaranteed to be physically zero-copy; teach only the logical tensor semantics.

Explain that GPT-style heads are represented as a tensor axis rather than requiring H separate Python modules. The learned head-specific freedom comes from different Q/K/V output channel blocks.

Stop before Step 13.4. Do not transpose to `(B,H,T,D)`, compute batched head-wise QK^T, discuss GPU-core mapping, concatenate heads, or introduce output projection.

## Step 13.4 accuracy boundary

Start from Q/K/V `(B,T,H,D)` and swap the T/H axes to `(B,H,T,D)`. Teach the exact mapping `Q_BHTD[b,h,t,d] = Q_BTHD[b,t,h,d]`; this is axis reordering, not another learned projection.

Use MLX 0.32.2 `mx.swapaxes(a,1,2)`. For scores, swap K's last two axes to `(B,H,D,T)` and use batched `mx.matmul`: `(B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)`.

Explain that the leading `(B,H)` axes behave as batch dimensions for matmul, so each `(b,h)` obtains its own T×T score matrix without a Python loop.

Logical head independence means each head's score matrix uses only its own D-dimensional Q/K channels and has no cross-head Q/K dot products. Do not equate this with one physical GPU core per head; hardware scheduling is implementation-dependent.

A `(T,T)` causal mask may broadcast over `(B,H,T,T)`. Stop before Step 13.5: do not transpose head outputs back, concatenate/reshape them to `(B,T,C)`, or add output projection.

## Step 13.5 accuracy boundary

Start from head-wise outputs `(B,H,T,D)`. Swap H/T back to `(B,T,H,D)`, then merge `(H,D)` into `C=H×D` to recover `(B,T,C)`. Teach the exact index relation `O_cat[b,t,hD+d] = O_heads[b,h,t,d]`.

Make clear that concatenating heads is a layout operation: it does not sum or average heads and has no learned parameters. The subsequent output projection `c_proj / W_O:(C,C)` is the learned operation that can mix channels originating from different heads.

Use a small hand-designed W_O example to demonstrate cross-head mixing, but label it as deterministic teaching arithmetic rather than learned runtime output. MLX `nn.Linear(C,C)` may be shown for the real API; do not claim its random initialized values in advance.

Parameter note: with bias disabled, fused QKV contributes `3C^2` weights and output projection contributes `C^2`, totaling `4C^2` for the attention projections. With biases on both, total is `4C^2+4C`.

Stop before Step 14.1. Do not introduce residual addition, normalization, feed-forward MLP, or a full Transformer Block.

## Step 14.1 accuracy boundary

Start from the shape-preserving Attention result: both x and F(x) are `(B,T,C)`, so the Transformer residual is same-shape element-wise `y=x+F(x)`. Do not rely on broadcasting as the conceptual mechanism.

Teach update-as-correction as an intuition: x is a baseline representation and F(x) is a learned correction. Do not claim F(x) must remain small.

For gradients, use the local Jacobian identity `J_y=I+J_F` and column-gradient convention `g_x=g_y+J_F^T g_y`. The direct `g_y` term motivates the gradient-highway intuition.

Do not turn that intuition into a guarantee. The counterexample `F(x)=-x` gives `y=0` and `J_y=0`, so residual connections do not mathematically guarantee non-vanishing gradients, invertibility, or zero information loss. Across many blocks, products of `(I+J_F)` can still have difficult behavior.

Residual addition itself has no trainable parameters. Stop before Step 14.2: do not teach LayerNorm/RMSNorm, RMS, epsilon, learned gamma, or normalization placement.

## Step 14.2 accuracy boundary

Normalize each Transformer token independently across its last feature axis C. Do not describe LayerNorm/RMSNorm as computing statistics across the batch or across sequence positions.

LayerNorm uses feature mean and population variance: `(x-mu)/sqrt(var+eps)`, then learned per-feature gamma and optionally beta. RMSNorm uses `x/sqrt(mean(x^2)+eps)` times learned gamma and does not mean-center.

Teach epsilon as a fixed positive numerical-stability constant, not a learned parameter or noise term. Teach gamma as learned per-feature scale that restores feature-specific amplitude after normalization.

Historical framing: LayerNorm (2016) predates and became canonical in the original Transformer era; RMSNorm (2019) simplifies the normalization by removing re-centering while retaining re-scaling. Do not claim RMSNorm is universally superior or mathematically equivalent to LayerNorm.

For MLX 0.32.2: `nn.LayerNorm(dims,eps=1e-5,affine=True,bias=True)` and `nn.RMSNorm(dims,eps=1e-5)`. RMSNorm has learned weight/gamma but no beta in this API.

Stop before Step 14.3. Do not decide whether normalization sits before or after the residual branch, and do not teach Pre-Norm versus Post-Norm yet.

## Step 14.3 accuracy boundary

Use a generic sublayer F and normalization N. Post-Norm is `y=N(x+F(x))`; Pre-Norm is `y=x+F(N(x))`. Both preserve `(B,T,C)`.

Historical framing: the original Transformer uses the classic Add & Norm / Post-Norm pattern, while many later deep Transformer/LLM architectures use Pre-Norm or related variants. Do not claim this makes Pre-Norm universally superior.

For gradient topology, teach `J_pre=I+J_F J_N` and `J_post=J_N(I+J_F)` in the local chain-rule ordering. With column gradients, Pre-Norm retains a direct `g_y` term that bypasses N and F; Post-Norm's path from output back to x still includes the normalization Jacobian.

Treat this as an optimization/gradient-flow motivation, not a guarantee that Pre-Norm cannot vanish/explode or must achieve better final quality.

Use only a generic sublayer in code. Stop before Step 14.4: do not introduce the `C->4C->activation->C` feed-forward MLP or compare Attention-vs-MLP roles yet.

## Step 14.4 accuracy boundary

Teach the Transformer feed-forward network as a shared position-wise function: each `(b,t)` token vector is transformed independently by the same parameters. Do not say the MLP itself performs cross-token mixing; Attention supplies that role.

Use the classic teaching path `C->4C->activation->C`, while stating that 4× is a common design choice rather than a mathematical requirement.

Explain why the nonlinear activation matters: without it, stacked affine/linear maps collapse to another affine/linear map. Use MLX 0.32.2 `nn.GELU()` for this lesson.

MLX `nn.Linear` acts on the last feature axis, so `(B,T,C)` can be processed directly with shared weights across positions; no Python token loop is required.

Parameter count for the lesson's `bias=True` MLP is `8C^2+5C`; with both biases disabled it is `8C^2`. The output returns to `(B,T,C)`, making residual addition shape-compatible.

Connect the MLP only as a concrete instance of the generic F from Step 14.3. Stop before Step 14.5: do not assemble the complete Transformer Block, compute total Block parameters, or perform Block-level numerical sanity tests.

## Step 14.5 accuracy boundary

Implement one minimal Pre-Norm Transformer Block using two independent RMSNorm modules: `x1=x+Attn(norm1(x))`, then `y=x1+MLP(norm2(x1))`. Attention must reuse the already-taught causal multi-head tensor path and MLP must reuse `C->4C->GELU->C`.

Keep the Block shape invariant `(B,T,C)->(B,T,C)` and explicitly test that sequence length T does not change.

For this lesson's exact parameter convention, QKV/c_proj have no bias, MLP fc1/fc2 use bias, and each RMSNorm contributes C learned gamma values. The expected Block count is `12C^2+7C`.

Prepare but do not run tests for shape invariants, parameter-count equality, finite outputs, and causal invariance under editing only a future final token. Do not state or imply that any test passed during static preparation.

The causal test should only require earlier outputs to remain unchanged when the last future token changes. Do not require the changed token's own output to differ.

Passing these tests later would not establish correct training, gradients, convergence, language-model quality, or complete GPT correctness.

Stop before Step 15.1. Do not add embeddings, positional path, multiple-block stacking, GPT model classes, or language-model heads.

## Step 15.1 accuracy boundary

Frame the 2017 Original Transformer as an encoder–decoder Seq2Seq model, not as decoder-only GPT. Connect it explicitly to the earlier recurrent Seq2Seq history: the encoder/decoder task roles remain while recurrent state transitions are replaced by attention-based stacks.

Encoder self-attention uses source representations as Q/K/V and is not decoder-style causal attention (apart from padding or other masks). Decoder masked self-attention is causal over the target stream.

Cross-attention uses decoder-side Q and encoder-memory K/V. With target length `T_tgt` and source length `T_src`, the score matrix is rectangular `(B,H,T_tgt,T_src)` in the multi-head case.

Preserve the shifted-target training story: decoder target input is shifted right and masked self-attention prevents future-target leakage; cross-attention may use the complete source memory.

Do not confuse the Step 14.5 modernized teaching Block with the 2017 decoder layer. The original Transformer is classically encoder+decoder, Post-Norm / LayerNorm, ReLU FFN, and decoder cross-attention; Step 14.5 used Pre-Norm / RMSNorm / GELU and no cross-attention.

Stop before Step 15.2. Do not explain the decoder-only sufficiency argument, delete the encoder, or implement GPT / MiniGPT top-level modules.

## Step 15.2 accuracy boundary

Anchor the explanation in autoregressive factorization: one token stream is modeled as `p(x_0,...,x_T)=prod_t p(x_t|x_<t)`. The needed conditioning information is the already-existing prefix in that same stream.

Do not confuse shifted training input/labels with Original Transformer source/target streams. They are two shifted supervision views of the same underlying sequence, not separate encoder/decoder architectural inputs.

Explain that causal self-attention over the same stream gives position i access to `x_0...x_i` while predicting `x_(i+1)`. With no independent source memory, the encoder stack and encoder–decoder cross-attention can be removed.

Decoder-only is a historical/architectural lineage term, not a literal copy of the full 2017 decoder layer: GPT-style blocks keep causal self-attention and FFN-style processing but remove cross-attention.

Prompt-conditioned tasks may serialize prompt/source before the continuation so response tokens can attend to the full prompt as causal past. Do not claim this makes encoder–decoder models universally obsolete or inferior.

Preserve the training/inference distinction: training can compute positions in parallel under the mask; autoregressive inference remains token-by-token.

Stop before Step 15.3. Do not enumerate or implement the MiniGPT top-level token embedding, position embedding, Block × N, final norm, or LM head.

## Step 15.3 accuracy boundary

Teach only the top-level MiniGPT component inventory and shape flow: token IDs `(B,T)` -> token embedding `(B,T,C)` + position embedding `(T,C)` -> Block × N -> final norm -> LM-head logits `(B,T,V)`.

Token embedding is an MLX `nn.Embedding(V,C)` lookup table with stored shape `(V,C)`. Token IDs are categorical indices, not continuous features.

For this project, choose learned absolute positions with a table `(T_max,C)` and require `T<=T_max`. State explicitly that this is a MiniGPT teaching choice, not a universal modern-LLM requirement. The causal mask handles visibility and is not a substitute for positional representation.

Explain that `Block × N` normally means separately parameterized block instances with the same architecture, not one shared block repeatedly called.

Because the current Block is Pre-Norm, use a final RMSNorm after the stack before the LM head. Treat this as the current project design rather than a universal placement law.

The LM head maps `C->V` independently at each position and returns unnormalized logits `(B,T,V)`. Do not add softmax or loss in this lesson.

Stop before Step 15.4. Do not implement a `MiniGPT` class, construct the real Block list, add model-level shape assertions, or revisit causal-mask module-state registration.

## Step 15.4 accuracy boundary

Implement a clean MLX `MiniGPT(nn.Module)` using the already-taught token embedding, learned absolute position embedding, independently instantiated Transformer Blocks, final RMSNorm, and bias-free LM head.

Validate positive constructor dimensions, `dims % num_heads == 0`, rank-2 token IDs `(B,T)`, and `1<=T<=max_seq_len`. Keep token dtype/index-range as the caller/embedding lookup contract rather than adding device-synchronizing runtime checks.

MLX 0.32.2 recursively tracks child modules stored in list/dict/tuple structures, so `self.blocks=[TransformerBlock(...) for ...]` is valid and produces separately parameterized blocks. Do not create one shared block and repeat the same object reference unless weight sharing is intentionally desired.

MLX `Module.parameters()` recursively collects mx.array members. A directly assigned `self.causal_mask = mx.array(...)` would therefore become module state and, unless frozen, trainable. Keep causal masks and position IDs forward-local in the primary implementation.

If a structural array is cached in the future, explain that it must be explicitly frozen (or otherwise managed as non-trainable state). Frozen arrays remain state/parameters but are excluded from `trainable_parameters()`.

Use Python scalar metadata for head_dim, scale, max_seq_len, etc. Do not introduce weight tying in this lesson; token embedding and LM-head weights remain separate.

Stop before Step 15.5. Do not compute the entire MiniGPT parameter count or provide the full line-by-line historical forward trace yet.

## Step 15.5 accuracy boundary

Trace the exact Step 15.4 implementation from token IDs `(B,T)` through token/position embeddings, every Block internal tensor shape, final RMSNorm, and LM-head logits `(B,T,V)`.

Use the exact parameter convention already fixed by the implementation. Total trainable parameters must be `P = 2VC + T_max*C + N(12C^2+7C) + C = 12NC^2 + C(2V+T_max+7N+1)`.

For `V=32,C=16,H=4,N=2,T_max=16`, the hand-derived parameter total is 7664. Token embedding and LM head are not tied, so both VC terms count independently. Learned position embeddings count the full `T_max*C` table, not current T.

In the current fixed-C full-width QKV design, H does not change parameter count; it changes D=C/H and tensor layout, with divisibility required.

Do not count causal masks, position IDs, current sequence length, head_dim, scale constants, or other forward-local/Python metadata as trainable parameters.

Prepare a `tree_flatten(model.trainable_parameters())` count helper and a detailed forward shape trace helper, but do not run them or claim their assertions/results passed.

Connect each component back to its historical purpose across the course. Stop before Step 16: do not create/download Tiny Shakespeare data, tokenizer/data pipeline, batching, loss, optimizer, or training workflow.
