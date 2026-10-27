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
Step 12.1         current material prepared
Step 12.2+        planned only; not current
```

**Material status is not learner status.** No claim is made that the learner has completed the Step 12.1 work notebook, completed exercises, or mastered the autoregressive visibility constraint.

Do not enter Step 12.2 until the learner explicitly asks to proceed beyond Step 12.1.

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

> **Step 12.1 — Autoregressive constraint**

Step 12.1 material is prepared only. Do not infer learner mastery or exercise completion, and do not create, teach, or mark Step 12.2 as current unless the learner explicitly asks to proceed.

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
