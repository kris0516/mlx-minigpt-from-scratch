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
Steps 15.1–15.5  material prepared; static audit passed (15.5 correction 1/2)
Step 16.1         material prepared; static audit passed (correction 1/2)
Step 16.2         material prepared; static audit passed (correction 1/2)
Step 16.3         material and full-reply core independently passed; corrections 0/2
Step 16.4         material reviewer independently confirmed core static PASS; corrections 0/2
Step 16.5         material reviewer independently confirmed core static PASS; corrections 0/2
Step 17.1         material reviewer independently confirmed core static PASS; corrections 0/2
Step 17.2         material reviewer independently confirmed core static PASS; corrections 0/2
Step 17.3         material reviewer independently confirmed core static PASS; corrections 0/2
Step 17.4         material reviewer independently confirmed core static PASS; corrections 0/2
Step 17.5         current initial material prepared; corrections 0/2
Step 18+          planned only; not current
```

**Material status is not learner status.** All major Step 16 masters, documents, and full replies; Step 17.1's actual master, documentation, fixed-version MLX API contract, and full reply; Step 17.2's actual master, documentation, and full reply; Step 17.3's actual master, documentation, MLX v0.32.2 API, and full reply; and Step 17.4's actual master, documentation, fixed-version MLX API, and full reply have independently passed the material reviewer's core static review. This is not learner learning or execution feedback. No claim is made that the learner has a local corpus cache, executed Step 17 code, performed experiments, completed exercises, or mastered the material. No runtime validation is asserted. Step 17.5 has initial material only; the earlier independent acceptance does not automatically extend to it.

Preserve all passed Step 16 and Step 17.1–17.4 teaching and masters. This turn prepares and publishes only Step 17.5, then stops at major Step 17, before Step 18. Step 17.5 supplies a self-contained toy Module, already-shifted data, scalar loss, Module-aware gradient function, and explicit Adam settings to connect one uncompiled training step. It explains reverse-mode autodiff, model/optimizer state updates, and the required explicit evaluation boundary. The prepared call returns the loss from before the update; it is not a post-update loss measurement. All lesson code remains unexecuted, and no corpus, old kernel, sampler, or new dependency is needed.

### mlx-chap2 continuation ledger

- Step 16.1 material/core teaching passed; actual corrections **1/2**.
- Step 16.2 material/core teaching passed; actual corrections **1/2**. The learner confirmed one completion of the full reply after a timeout; this previously unrecorded correction is now included.
- Step 16.3 master and full-reply core independently confirmed PASS by the learner; actual corrections **0/2**. Do not reopen the passed lesson.
- Step 16.4 full reply, master, and Step 16 documentation independently confirmed core static PASS by the material reviewer; actual corrections **0/2**. This is material acceptance, not learner feedback. Do not reopen the passed lesson.
- Step 16.5 material/core teaching passed; actual corrections **0/2**. The material reviewer independently confirmed core static PASS for all major Step 16 masters, documents, and full replies; this cumulative acceptance is not learner feedback.
- Step 17.1 actual master, documentation, fixed-version MLX API contract, and full reply independently confirmed core static PASS by the material reviewer; actual corrections **0/2**. This is material acceptance, not learner feedback. Do not reopen the passed lesson.
- Step 17.2 actual master, documentation, and full reply independently confirmed core static PASS by the material reviewer; actual corrections **0/2**. This is material acceptance, not learner feedback. Do not reopen the passed lesson.
- Step 17.3 actual master, documentation, MLX v0.32.2 API, and full reply independently confirmed core static PASS by the material reviewer; actual corrections **0/2**. This is material acceptance, not learner feedback. Do not reopen the passed lesson.
- Step 17.4 actual master, documentation, fixed-version MLX API, and full reply independently confirmed core static PASS by the material reviewer; actual corrections **0/2**. This is material acceptance, not learner feedback. Do not reopen the passed lesson.
- Step 17.5 initial preparation; actual corrections **0/2**.
- Earlier notebook wording about an initial count is historical; this cumulative ledger does not reset when conversations change.
- Subsequent requests continue one subsection at a time through major **Step 20**, then stop. This turn ends at Step 17.5 and major Step 17; do not prepare Step 18 or any later lesson during this turn, and do not infer authorization for automatic batch preparation. Keep the wider Step 21–25 curriculum as a future plan.

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
- actual correction counts persist across conversations and must not reset in a new session;
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
7. Use the normal plain `cp` command unless troubleshooting or an explicit learner request requires another form. For mlx-chap2 the learner requests a new filename and no overwriting of notes: use a timestamped `*_work.ipynb` destination with `cp -n`. Keep `git pull` and copying in two separate terminal blocks.

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

> **Step 17.5 — Uncompiled training step**

Step 17.5 initial material is prepared and published only. All major Step 16 masters, documents, and full replies; Step 17.1's actual master, documentation, fixed-version MLX API contract, and full reply; Step 17.2's actual master, documentation, and full reply; Step 17.3's actual master, documentation, MLX v0.32.2 API, and full reply; and Step 17.4's actual master, documentation, fixed-version MLX API, and full reply have independently passed the material reviewer's core static review. Neither acceptance nor current preparation implies learner completion, mastery, experiments, exercise completion, or runtime validation. Stop at Step 17.5 and major Step 17, before Step 18. The material sequence now reaches the final planned subsection of major Step 17; 17.5 is not yet independently accepted by the material reviewer. Step 14's completed static audit still does not imply runtime tests were executed or passed.

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

## Step 15.5 correction 1/2

Minimal dependency repair only. Do not reopen the already-passed formula, 7,664 example, shape trace, or historical explanation.

The Step 15.5 notebook must be self-contained before helper/usage cells: explicitly import `mlx.core as mx`, then reuse the Step 15.4 definition-only cells for `CausalMultiHeadAttention`, `FeedForwardMLP`, `TransformerBlock`, and `MiniGPT`.

Keep definition cells separate from usage/assertion cells. Do not use the Step 15.4 usage cell as a mechanism for loading definitions, and do not `%run` the whole Step 15.4 notebook because that would execute example/assertion code.

Operational order in a 15.5 work copy: run definition-only cells first; then helper definitions; then, only when the learner chooses, run explicitly marked usage/assertion cells.

Correction count: 1/2. No runtime execution or Step 16 work is implied.

## Step 16.1 accuracy boundary

Use the Karpathy `char-rnn/data/tinyshakespeare/input.txt` source and cache it locally at `data/tinyshakespeare/input.txt`. The repository already ignores `/data/`; do not commit the corpus.

Acquisition must be cache-first and use only Python standard-library functionality already available (`pathlib`, `urllib.request`) so the lesson does not add a new requests dependency. Keep acquisition helper definition separate from the usage cell that can actually access the network.

Read cached text explicitly as UTF-8. Teach character count and UTF-8 byte count as distinct quantities.

Reuse the Step 6.1 character tokenizer: `sorted(set(text))`, `stoi`, `itos`, `encode`, `decode`. Do not introduce BPE, SentencePiece, GPT-2 tokenizer, `<UNK>`, or other tokenizer systems here.

For this character tokenizer, one corpus character becomes one token, so encoded length N equals `len(text)`. Convert the full encoded corpus to a 1-D MLX integer array `(N,)`, but do not create train/validation splits or batches.

The standard file is commonly reported by Karpathy's nanoGPT preparation script as 1,115,394 characters with vocabulary size 65; present this only as a reference expectation, not as a hard assertion. Actual local inspection remains authoritative.

Clarify that corpus length N and model context T/T_max are different. Stop before Step 16.2: no held-out split, no batch sampler/block_size, no shifted batch construction, no loss/optimizer/training.

Step 16.1 material/core teaching passed; actual correction count is **1/2**, carried forward without resetting. Do not actually download the corpus or execute any Notebook cell during material preparation.

## Step 16.2 accuracy boundary

Start from the Step 16.1 character-level `token_stream:(N,)`. The new notebook must be operable without sharing a kernel with 16.1, so include a minimal local-cache bootstrap helper that reads `data/tinyshakespeare/input.txt`, rebuilds the same character vocabulary/token stream, and never accesses the network.

If the cache is absent, fail clearly and direct the learner back to Step 16.1 acquisition. Do not duplicate network download logic in 16.2.

Use a contiguous 90/10 split: `N_train=floor(0.9N)`, `N_val=N-N_train`; output `train_tokens:(N_train,)` and `val_tokens:(N_val,)` with non-overlapping original index ranges.

Teach held-out validation as data excluded from optimizer parameter updates. Future validation evaluation may inspect model performance, but validation examples are not optimizer training examples.

Do not randomize individual tokens across splits. For this single continuous corpus, preserve contiguous text structure.

State the future sampler boundary without implementing it: every training window must lie fully inside train_tokens and every validation window fully inside val_tokens. No window may cross the split point.

Keep the Step 16.1 fixed character vocabulary. This lesson's held-out separation concerns sequence tokens / optimization examples; do not redesign tokenizer training.

Stop before Step 16.3. No block_size, batch sampler, random start indices, shifted `(x,y)` windows, loss, optimizer, or training loop.

Step 16.2 material/core teaching passed; actual correction count is **1/2**, including the learner-confirmed reply completion after a timeout. Do not reopen the passed lesson or reset the count. Do not execute corpus reads, split code, assertions, Notebook cells, tests, or training during material preparation.

## Step 16.3 accuracy boundary

Lesson master: `notebooks/69_step16_3_batch_sampler.ipynb`. The learner independently confirmed that the master and full reply pass the core-content review; actual correction count remains **0/2**. This does not change the accumulated 1/2 counts for Steps 16.1 and 16.2. Preserve this passed master.

The new work copy must operate in its own kernel. Include the imports and definition-only helpers needed to find the course root, read the existing local cache, rebuild the same character vocabulary/MLX int32 stream, and reuse the contiguous 90/10 split from Step 16.2. Keep path selection, cache reading/splitting, and sampling usage separate from definitions. Never `%run` a whole previous notebook or assume its variables exist.

Default to the repository-root `data/tinyshakespeare/input.txt`, but explicitly show the kernel CWD and selected absolute cache path. If Step 16.1 created its cache elsewhere, the learner points `CACHE_PATH` at that existing file. Missing cache produces a clear error directing the learner to Step 16.1 or the correct path; no download or file relocation occurs in 16.3.

Use the existing Step 6.5 interface `get_batch(data, batch_size, block_size, rng)` on one already-selected split. Explain B as batch size and T=block_size as this sampler's fixed input length. A supervised row requires T+1 consecutive tokens: X uses the first T, Y the last T. Distinguish block_size from the number of Transformer Blocks; future model use requires T<=max_seq_len.

Derive legal starts from the last target: `s+T<N_split`, hence `0<=s<N_split-T`. Use `rng.randrange(0, N_split-block_size)` with its exclusive upper bound. N_split=T+1 has exactly start 0; N_split<=T must fail clearly. Validate one-dimensional input and positive integer batch_size/block_size.

For row b and position t, preserve `X[b,t]=data[starts[b]+t]` and `Y[b,t]=data[starts[b]+t+1]`. Explain slicing and `mx.stack(..., axis=0)` before use. Output X/Y have shape `(B,T)` and preserve the input's integer dtype; starts is a Python list of length B, not a third tensor. Targets remain categorical IDs rather than one-hot vectors or embeddings.

Call the sampler separately with train_tokens and val_tokens, so the entire T+1 chunk including its last target stays inside its split. Starts are split-local; a validation start maps to original index split_index+start. Batch rows may overlap or repeat, while token order within each row is preserved. Shifted targets define supervision; the already-taught causal mask governs input visibility.

Prepare only a small B=2, T=8 usage and optional inspection of one decoded row per split. Use `random.Random()` without an explicit seed only to supply random starts. Do not conduct reproducibility experiments or teach seed/state restoration before 16.4. Do not run a MiniGPT forward, loss, optimizer, tests, training, or the user's work copy.

The historical scope of the Step 16.3 lesson stops before seed/state restoration; current progression is recorded above. Material preparation/publication is not evidence of runtime validation or learner mastery.

## Step 16.4 accuracy boundary

Lesson master: `notebooks/70_step16_4_reproducibility.ipynb`. The material reviewer independently confirmed core static PASS for its full reply, master, and Step 16 documentation; actual correction count remains **0/2**. This is not learner learning or execution feedback. Carry forward Step 16.1=1/2, Step 16.2=1/2, and the independently passed Step 16.3=0/2 without resetting or reworking them. Preserve the passed master.

The new work copy is self-contained: include the same Path/random/MLX imports, root/cache/split helpers, and four-argument `get_batch` definition from Step 16.3. Separate definitions, cache-path selection, local bootstrap, RNG setup, sampling usage, and state restoration. The learner can select the existing cache by absolute path; missing cache fails clearly. Do not assume shared kernel state or execute a previous notebook wholesale.

Teach a seed as initialization input and RNG state as the current continuation position. Use `S_0=Init(seed)` and `(starts_k,S_{k+1})=G(S_k;B,N_split-T)` only after defining each symbol. This is a whole-batch transition, not a claim that each randrange call consumes exactly one fixed PRNG step. A small hypothetical example with N=8, T=3, B=2 and successive starts [1,4] then [0,3] is a hand illustration, not measured output or a claimed output for a particular seed.

Same initial seed or restored state, matching sampler/call parameters and call order, and a compatible runtime support matching start sequences. Matching X/Y additionally requires identical token data, vocabulary mapping, and split. Preserve the `(B,T)` integer arrays, length-B Python starts list, shifted targets, and split-local boundaries. Do not claim cross-version, cross-device, or whole-training bitwise determinism from seeds alone.

Prepare explicit `train_rng=random.Random(2026)` and `val_rng=random.Random(2027)` objects, B=2 and T=8, and `mx.random.seed(2028)` for future MLX initialization. Explain before use that these are separate random sources: module-level `random.seed` does not seed existing explicit Random instances, and MLX seed does not seed Python sampling. MLX seeding must precede future model initialization to affect its random draws; this lesson neither constructs a model nor draws MLX random arrays.

Use each Python RNG continuously. Do not reseed or recreate it before each batch. Validation sampling consumes val_rng, leaving train_rng untouched. Different seeds do not guarantee non-overlapping windows or different outputs on every call. Batch counters count sampled batches, not optimizer updates, epochs, or low-level random draws.

Prepare an in-memory snapshot dictionary containing both Python `getstate()` values, batch_size, block_size, and both sampled-batch counters. Restore two replacement Random objects with `setstate()`, then restore configuration and counters; the next draw continues from the saved boundary. Explain that getstate/setstate themselves do not sample. After restoring, do not rerun the fresh-run seed setup cell. Resetting to the original seed restarts the stream; it does not continue from the saved point.

Keep the same data/split/vocabulary and sampler definition during this in-memory demonstration. The snapshot does not contain corpus data, MLX RNG, model weights, or optimizer state; it is not a complete checkpoint and does not survive a kernel restart. It cannot reconstruct Step 16.3's earlier unsaved, unseeded RNG sequence. Do not give a writable-list assignment recipe for `mx.random.state`; the project uses MLX 0.32.2 and the actual MLX checkpoint/state API belongs to later implementation work.

The future resume checklist may name exact data identity (filename and length alone are insufficient), vocabulary/split, sampler and B/T configuration, Python RNG states, progress, software/device/dtype, and future model weights, optimizer state, training step, and MLX RNG state. Explain only their roles here. Leave checkpoint serialization, optimizer internals, compiled state handling, and resume verification to their existing Steps 17/18/20; do not introduce Step 16.5 industrial input pipelines.

The historical scope of the Step 16.4 lesson stops before industrial input pipelines; current progression is recorded above. Material preparation/publication does not imply learner mastery or runtime verification.

## Step 16.5 accuracy boundary

Lesson master: `notebooks/71_step16_5_toy_vs_industrial_input_pipelines.ipynb`. The material reviewer independently confirmed core static PASS for all Step 16 masters, documents, and full replies; actual correction count for 16.5 remains **0/2**. Carry forward 16.1=1/2, 16.2=1/2, 16.3=0/2, and 16.4=0/2. Do not reset counts or rework passed masters; acceptance is not learner feedback.

Follow the existing five roadmap topics: random windows, shuffling, token packing, distributed shards, and deterministic cursor/state. Use Chinese explanations with English terms introduced before use. The new work copy is self-contained: random/MLX imports, the unchanged four-argument Step 16.3 get_batch definition, small lesson-specific helper definitions, and literal toy inputs. It needs no corpus cache, previous kernel variables, network access, or execution of another notebook. Separate definitions from prepared usage. Do not add third-party data-loading dependencies.

Explain that the existing random-window sampler samples starts with replacement and may repeat or omit windows; it does not establish a full-coverage epoch. A toy stream of IDs 0 through 11 with B=2, T=4 has starts 0 through 7 and preserves X/Y:(2,4). Shuffling should reorder finite sample/window IDs, not tokens inside a language-model sequence. Explain that rng.shuffle(order) mutates the list in place and returns None before using it. Visiting each candidate ID once requires a full traversal of that permutation; overlapping candidate windows may still share tokens. A bounded streaming shuffle buffer is not the same as a global permutation.

Packing uses a clearly separate synthetic convention: toy_docs=[[10,11,12],[20,21],[30,31,32,33]], toy EOS ID 99 appended after every document. EOS=99 is not inserted into Tiny Shakespeare's real vocabulary and is not passed to the course model. The resulting length-12 stream produces T+1 chunks with stride T=4: [10,11,12,99,20] and [20,21,99,30,31]. X/Y shape is (2,4). Carry [31,32,33,99] includes the preceding row's last target as the next row's first input; three adjacent pairs remain pending until further tokens are available. Do not silently pad, wrap, or discard this carry in the example.

This toy packing policy deliberately accepts cross-document next-token targets and causal visibility within a packed row. EOS marks a boundary but does not automatically isolate attention or suppress boundary targets. Independent-document policies need document/segment metadata and explicit attention/target rules; describe their role without changing the current model, implementing new masks/losses, or entering Step 17. Keep train/validation data separation in any packing or shard discussion.

Distinguish storage shards, distributed rank, world size, and loader workers. The distributed example is only a Python allocation plan, with shared hand-specified sample order [7,2,9,4,0,6,1,8,3,5], world_size=2, and local batch size=2. Allocate consecutive groups of two per rank inside each logical global batch of four. Use the first eight IDs and explicitly drop tail [3,5]; rank 0 gets [[7,2],[0,6]], rank 1 gets [[9,4],[1,8]]. Each rank has X/Y:(2,T), with logical global shape (4,T); no distributed process, communication, gradient accumulation, or training is implemented. Sample IDs identify records and are not the token IDs inside them. Do not simultaneously promise complete coverage, no duplication, and equal rank step counts when a non-divisible tail policy prevents that.

Define local plan cursor as next_batch_index, the index of the next batch to deliver from the current rank's plan. An in-memory plan snapshot may preserve tuple(order), epoch, rank/world_size, local B, T, tail policy, a declared toy data identity label, and next_batch_index. Rebuild the same plan against unchanged toy records/runtime to identify the next batch. A label is metadata, not a content-integrity proof. With the complete fixed order saved and no further random transforms, locating the next batch in the current epoch requires no additional RNG draw; later random operations still require the corresponding Step 16.4 RNG state. Changing world size changes local assignment, so an old local cursor cannot be copied blindly.

Do not describe this plan snapshot as saving a live packing stream, prefetch queue, actual distributed loader, model, optimizer, or full training state. Industrial continuation may require shard manifest/order, reader positions, all relevant RNG states, shuffle-buffer contents, packing carry/boundaries, consumed-position accounting, pending prefetched items or a deterministic reconstruction from a committed consumption boundary, and topology/software metadata. Reading through item 12 while only consuming through item 8 means saving only the reader position can skip items 9–12. Keep persistence and actual training resume implementation in Step 20.

The historical Step 16.5 lesson ends at major Step 16; current progression is recorded above. Its preparation did not read/download corpus, execute Notebook cells, tests, training, experiments, sampling, or a user work copy, answer exercises, or imply mastery/runtime verification. Future requests continue one subsection at a time through major Step 20, then stop.

## Step 17.1 accuracy boundary

Lesson master: `notebooks/72_step17_1_logits_probabilities_cross_entropy.ipynb`. The material reviewer independently confirmed core static PASS for the actual master, documentation, fixed-version MLX API contract, and full reply. Actual corrections remain **0/2**. Preserve 16.1=1/2, 16.2=1/2, and 16.3–16.5=0/2. All major Step 16 masters, documents, and full replies also independently passed the material reviewer's core static review; this is material acceptance, not learner learning or execution feedback. The scope below records the passed 17.1 lesson; the current checkpoint is recorded above.

Follow the existing 17.1 topics: logits are not probabilities, softmax, negative log-likelihood, and one small hand-computable example. Connect the raw MiniGPT logits `(B,T,V)` to the already-shifted integer targets `Y:(B,T)`, pairing `logits[b,t,:]` with `Y[b,t]`. Do not shift Y again. Preserve the raw-logit model interface and do not alter passed model or data notebooks.

The new work copy is self-contained with `mlx.core` and `mlx.nn` imports and explicit toy arrays. It does not require a corpus cache, previous kernel variables, sampler calls, model construction, model forward, or whole-notebook `%run`. Toy logits are manually specified interface examples, not real model outputs or measured results. No new project dependency is needed.

Define B/T/V, a fixed position `(b,t)`, class index v, correct class ID y, logits z, and probabilities p before using them. Logits are unconstrained real-valued scores; token IDs are categorical labels rather than numeric distances. Softmax normalizes over the vocabulary axis V (`axis=-1`), not over batch or time positions. Distinguish this axis from attention's key-position axis even though both examples use a final-axis softmax.

Explain exponentiation, natural logarithm, the summation symbol, and the logarithm quotient rule before their use. Use `p_v=exp(z_v)/sum_u exp(z_u)`, then `NLL=-ln(p_y)`. Larger probability for the correct class produces smaller NLL; model-assigned probability is not an empirically verified accuracy rate.

Use V=3, `z=[ln(1),ln(2),ln(3)]`, y=2, giving p=[1/6,1/3,1/2] and NLL=ln(2), approximately 0.6931. Define general categorical cross-entropy `H(q,p)=-sum_v q_v ln(p_v)` and show that hard-label one-hot q=[0,0,1] reduces it to NLL. The one-hot is a mathematical explanation; actual Y remains integer IDs. Do not introduce label smoothing or soft-target training.

Derive `NLL=logsumexp(z)-z_y`, explaining logsumexp before its API. For the hand-written stable calculation, first subtract m=max(z), then calculate `mx.logsumexp(shifted,axis=-1)-shifted[y]`. Explain row-wise `keepdims=True` if broadcasting is used. Do not implement loss as `-log(softmax(z)[y])` for general numerical use, and do not claim unlimited robustness under extreme floating-point values.

Use the verified MLX 0.32.2 `nn.losses.cross_entropy(raw_logits,Y,axis=-1,reduction='none')` contract. Explain raw logits, the explicit class axis, integer target IDs in `[0,V)`, and no reduction before use. Do not feed probabilities back as logits. The result retains per-position shape `(B,T)`; no B×T flattening, global mean, or other full-batch aggregation is taught here.

The prepared shape example uses chunk `[[0,2,1]]`, X=`[[0,2]]`, Y=`[[2,1]]`, and manually constructed toy logits `mx.log(mx.array([[[1.,2.,3.],[3.,2.,1.]]],dtype=mx.float32))`, shape `(1,2,3)`. Taking log of positive toy numbers is only a convenient way to make the hand calculation simple; it is not the MiniGPT algorithm for token IDs or word counts. Per-position losses have shape `(1,2)` with hand-derived values `[[ln(2),ln(3)]]`, not claimed runtime output.

The 17.1 preparation was limited to materials: no corpus read/download, Notebook execution, tests, sampling, experiments, training, user work copy execution, or exercise answers. That lesson stopped at 17.1 before 17.2 and did not teach flattening, gradient/value_and_grad, optimizer, training loop, or generation. Its material preparation and static review do not imply learner mastery or successful execution.

## Step 17.2 accuracy boundary

Lesson master: `notebooks/73_step17_2_flatten_bt_supervision.ipynb`. The material reviewer independently confirmed core static PASS for its actual master, documentation, and full reply. Actual corrections remain **0/2**. Preserve 16.1=1/2, 16.2=1/2 (including the previously recorded reply completion after a timeout), 16.3–16.5=0/2, and 17.1=0/2. Step 17.1's actual master, documentation, fixed-version MLX API contract, and full reply also independently received core static PASS; this is material acceptance, not learner learning or execution feedback. The scope below records the passed 17.2 lesson; the current checkpoint is recorded above.

Follow the existing 17.2 topics: `(B,T,V) -> (B*T,V)`, `(B,T) -> (B*T,)`, and many token-level targets in the batch used by a future training step. Explain B, T, V, N=B*T, and the flattened row index r before using them. Preserve the class axis V and the existing pairing `logits[b,t,:]` with already-shifted `Y[b,t]`; do not shift targets again.

Use a self-contained work copy with `mlx.core` / `mlx.nn` imports, `toy_chunks=[[0,2,1],[1,0,2]]`, X=`[[0,2],[1,0]]`, Y=`[[2,1],[0,2]]`, and manually specified toy logits `mx.log(mx.array([[[1.,2.,3.],[3.,2.,1.]],[[1.,1.,2.],[3.,2.,1.]]],dtype=mx.float32))`. B=2, T=2, V=3, N=4. No corpus cache, old kernel variables, sampler, model construction, forward, or new project dependency is required. Taking log of chosen positive values is only a convenient hand-calculation fixture, not the MiniGPT algorithm for producing logits.

Teach reshape as preserving logical element order and total element count. Keep V separate; merge only B and T. For this unchanged input order, r=b*T+t, `flat_logits[r,v]=toy_logits[b,t,v]`, and `flat_targets[r]=Y[b,t]`. Explain the trailing comma in the one-dimensional shape tuple `(N,)` before use. Then use `mx.reshape(toy_logits,(N,V))` and `mx.reshape(Y,(N,))`. A short reading note may explain that one inferred dimension can be written as -1; do not add a separate experiment. Do not imply reshape is always zero-copy or changes causal context. Both arrays must keep the same position order; do not transpose or reorder only one side.

Reuse raw-logit cross-entropy with integer targets, `axis=-1`, and `reduction='none'` to obtain `flat_losses:(N,)`. The original-rank call remains valid in MLX 0.32.2 and returns `(B,T)`; flattening is a way to organize supervision, not an API requirement. The flattened and original-rank losses correspond per position mathematically. For the four positions, `flat_targets=[2,1,0,2]` and hand-derived losses are `[ln2,ln3,ln4,ln6]`. These are derivations, not observed outputs or executed equality tests.

Introduce an equal-weight arithmetic mean only after explaining summation indices and division by the number of supervised positions: `L=(1/N)*sum_r ell_r=(1/(B*T))*sum_b sum_t ell_(b,t)`. Use `mx.mean(flat_losses)` to produce scalar shape `()`. Every position in this lesson is valid and equally weighted; do not expand into padding, masks, or weights. The denominator is N=B*T, not V. Distinguish three operations: reshape organizes positions; cross-entropy consumes each V-class vector and its target; mean aggregates the resulting position losses. Do not add a reduction='mean' shortcut or unrelated alternative implementations.

N supervised positions do not mean N parameter updates, N distinct token IDs, or N statistically independent examples. The 17.2 lesson only explained the loss organization used by a future training step. Its scope stopped at 17.2 before 17.3, without value_and_grad, gradients, optimizers, training loops, or generation. Its preparation was limited to materials: no corpus read/download, Notebook or user work copy execution, tests, numerical experiments, sampling, training, or exercise answers. Static review and publication do not imply learner mastery or runtime validation.

## Step 17.3 accuracy boundary

Lesson master: `notebooks/74_step17_3_nn_value_and_grad.ipynb`. The material reviewer independently confirmed core static PASS for its actual master, documentation, MLX v0.32.2 API, and full reply. Actual corrections remain **0/2**. Preserve 16.1=1/2, 16.2=1/2 (including the reply completion after a timeout), 16.3–16.5=0/2, and 17.1–17.2=0/2. This is material acceptance, not learner learning or execution feedback. The scope below records the passed 17.3 lesson and its historical stopping boundary; do not rework it.

Follow only the existing 17.3 topics: Module parameter trees; loss and gradients as a function transformation; and the difference from a model of gradients automatically attached to parameter arrays. Connect 4.2's Module/tree explanation, 5.3's mx.grad/mx.value_and_grad, 15.4–15.5's trainable parameter tree, and 17.2's scalar MLX mean loss. Do not reopen passed lessons or enter 17.4 Adam, parameter updates, the full training loop, or Step 18 execution/compilation topics.

Use a self-contained SharedLogitModel(nn.Module). Its __init__ calls super().__init__() and assigns `self.head={'bias':mx.array([0.,0.,0.],dtype=mx.float32)}`. Its __call__(self,X) obtains B,T from X.shape, V from the bias vector's shape, and returns `mx.broadcast_to(self.head['bias'],(B,T,V))`. Explain dictionary keys, containers and array leaves, callable instances, and broadcasting before code. Every position uses the same parameter vector. This toy depends only on X.shape, not token contents or context; it is an interface example, not MiniGPT or an initialization recommendation.

Keep the data outside the Module: `toy_chunks=[[0,2,1],[1,0,2]]`, X=`[[0,2],[1,0]]`, Y=`[[2,1],[0,2]]`; Y is already shifted. The work copy includes imports, the full toy class, explicit arrays, loss_fn, and prepared transform/call usage. No old kernel, prior Notebook execution, corpus, sampler, full MiniGPT definitions, or new dependency is required. All cells remain unexecuted and output-free.

Explain that parameters() returns the parameter tree, including frozen parameter leaves, while trainable_parameters() filters out frozen parameters. Public mx.array members can be collected through nested Modules/dicts/lists. Do not assert default tuple traversal. This toy freezes nothing, so both trees contain the single path `head -> bias`, with leaf shape (3,). Returned gradients follow the trainable tree: `grads['head']['bias']` is shape (3,), corresponding to the three shared parameter components, not N token losses or the full `(B,T,V)` logits array. Do not require `.grad`, `.backward()`, or `.zero_grad()` calls.

Define `loss_fn(model,X,Y)` so `logits=model(X)` is computed inside the differentiable function. Reuse 17.2's matching reshape to `(N,V)` and `(N,)`, raw-logit cross_entropy with axis=-1/reduction='none', and return mx.mean of the resulting (N,) losses as a scalar MLX array. Do not close over fixed 17.2 toy logits or convert the return value with item()/float(). Explain function objects versus calls, return, a function transformation returning a callable, and tuple unpacking before the usage code.

Use the fixed MLX 0.32.2 API `g=nn.value_and_grad(model,loss_fn)`, then `loss_value,grads=g(model,X,Y)`. The first model identifies whose trainable parameters are differentiated. Arguments to g are ordinary arguments forwarded to the supplied loss_fn; this wrapper does not insert model automatically. Use the same model instance in both places. nn.value_and_grad is the Module-aware helper, distinct from the general mx array-argument transformation; do not add argnums or other nonexistent helper arguments. No optimizer learning update occurs in the example. Do not claim the wrapper never writes Module state: fixed-version source internally uses model.update(params) to connect differentiated parameters to the computation, not to perform a learning update. The internal tracking mechanism need not become another teaching topic.

Let theta denote all trainable parameters, a length-3 vector in this toy, and write scalar L(theta;X,Y). Explain each partial derivative as the local loss change rate when that component changes and the others stay fixed. Gradient signs describe local effects, not an optimizer algorithm. With theta=[0,0,0], every position has p=[1/3,1/3,1/3], and L=ln3. Define q_(r,v) as the one-hot indicator for target flat_targets[r]. From ell_r=ln(sum_v exp(theta_v))-theta_y and the brief derivative/chain-rule reminders d(exp(u))/du=exp(u), d(ln(s))/ds=1/s, obtain d(ell_r)/d(theta_v)=p_v-q_(r,v). Average the four position contributions for target order [2,1,0,2]: gradient=[1/3-1/4,1/3-1/4,1/3-1/2]=[1/12,1/12,-1/6]. These are hand-derived expectations, not observed output or executed checks. Do not add a softmax Jacobian derivation.

API authority: fixed v0.32.2 python/mlx/nn/utils.py and python/mlx/nn/layers/base.py; the official helper page is https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html. Prepare and publish materials only: no corpus read/download, lesson code or Notebook execution, tests, numerical experiments, sampling, training, user work copy execution, or exercise answers. Stop at 17.3 before 17.4. Static review and publication do not imply learner mastery or runtime validation.

## Step 17.4 accuracy boundary

Lesson master: `notebooks/75_step17_4_adam_optimizer.ipynb`. The material reviewer independently confirmed core static PASS for its actual master, documentation, fixed-version MLX API, and full reply. Actual corrections remain **0/2**. Preserve 16.1=1/2, 16.2=1/2 (including the reply completion after a timeout), 16.3–16.5=0/2, and 17.1–17.3=0/2 across turns and conversations. This is material acceptance, not learner feedback. The scope below records the passed 17.4 lesson and its historical stopping boundary; do not rework it.

Follow only the existing 17.4 topics: Adam's first moment, second moment, optimizer state, and why that state consumes memory. Connect 17.3's scalar mean loss and returned trainable gradient tree to an optimizer consuming parameter-shaped gradients. Do not rework accepted lessons or assemble the 17.5 forward/loss/autodiff/update/evaluation training step. Do not teach AdamW, weight decay, schedules, compilation, mixed precision, distributed training, or checkpoint implementation.

The work copy is self-contained with `mlx.core as mx` and `mlx.optimizers as optim`, an explicit `parameters={'head':{'bias':mx.array([0.,0.,0.],dtype=mx.float32)}}` tree, and an `example_grads` tree at the same path containing float32 `[1/12,1/12,-1/6]`. These gradients are the hand-derived 17.3 interface example, not autodiff or runtime results from this turn. The scalar loss has already averaged the four token positions; do not divide these gradients by B*T again. No nn import, Module definition, old kernel, old Notebook execution, corpus, sampler, model forward, loss call, or differentiation call is needed.

Use k for the optimizer-update count, distinct from sequence length T and token position t. Define theta, g_k, learning rate alpha, beta1/beta2, m_k/v_k, epsilon, and element-wise arithmetic before the formulas. With zero initial moments, `m_k=beta1*m_(k-1)+(1-beta1)*g_k` and `v_k=beta2*v_(k-1)+(1-beta2)*g_k**2`. Explain EMA as retaining a fraction of the previous estimate and adding a fraction of the current observation. m retains signed gradient history; v accumulates squared gradients and is a raw second-moment estimate, not centered variance or a second derivative. Squaring, square root, division, and updating are element-wise; state leaves have parameter shape, not `(B,T,V)` or one entry per supervised token.

Explain zero-initialization bias before `m_hat=m/(1-beta1**k)` and `v_hat=v/(1-beta2**k)`. Hats label corrected quantities; the bias in bias correction differs from the model parameter named head.bias. Correction addresses the zero-initialization weighting and does not make a changing gradient process an exact oracle. Use the corrected update `theta_k=theta_(k-1)-alpha*m_hat/(sqrt(v_hat)+eps)` for the lesson's explicit `bias_correction=True`. The epsilon is outside the square root. The positive epsilon avoids a zero denominator in this float32 example; do not promise stability for every input or dtype.

Fixed MLX v0.32.2 Adam has required learning_rate, betas=[0.9,0.999], eps=1e-8, and bias_correction=False. The lesson explicitly selects `learning_rate=1e-3`, `betas=(0.9,0.999)`, `eps=1e-8`, and `bias_correction=True`; these are lesson settings, not all API defaults or tuned MiniGPT recommendations. Do not import another framework's defaults. False uses uncorrected m and v in the update; the default flag is a real formula choice. Betas, eps, and bias_correction are optimizer object configuration, not keys stored in its state dictionary.

The first hand calculation starts from theta0=[0,0,0], m0=v0=0, and g1=[1/12,1/12,-1/6]. Obtain m1=[1/120,1/120,-1/60], v1=[1/144000,1/144000,1/36000], m_hat1=g1, and v_hat1=g1**2. The corrected new parameter vector is approximately [-0.001,-0.001,+0.001] when the tiny positive epsilon is neglected for the displayed approximation; it is not exactly that vector. With bias_correction=False and all other settings unchanged, the first-step magnitude is about 0.0031623 instead. This comparison is hand reasoning only, not a second optimizer experiment. Adam does not promise every update decreases loss or that all future per-coordinate update magnitudes equal alpha.

Explain constructor/object, named arguments, a two-element tuple, method calls, dictionary indexing, and returned-tree assignment before the corresponding code. In a fresh optimizer, `optimizer.init(parameters)` prepares zero m/v state without performing a parameter update or incrementing step. The prepared isolated call `parameters_after=optimizer.apply_gradients(example_grads,parameters)` returns a new parameter tree and advances the optimizer's own state; it does not replace the input parameters tree. It is not a pure function with no side effects. Do not also call it a second time to inspect output. No model/update integration or complete training step is authored here.

The state contains scalar step and learning_rate plus parameter paths extended with m/v, e.g. `optimizer.state['head']['bias']['m']` and `['v']`, each shape (3,). After one prepared application, step would be 1; store the uncorrected m/v, not separate persistent m_hat/v_hat arrays. Source initializes moments with zeros_like(parameter), so this example has float32 state; do not generalize to automatic float32 state for arbitrary parameter dtypes. The state tracks compact summaries, not every past gradient. Recreating a fresh optimizer discards this history; persistence implementation belongs to Step 20.

For P optimized parameter scalars and d bytes per scalar, with parameter/gradient/moment dtypes equal, moments alone require approximately 2*P*d data bytes. Parameters plus a retained gradient plus both moments total approximately 4*P*d data bytes. This is a component accounting assumption, not measured process or peak memory. In the 3-scalar float32 example, moments use 24 data bytes; the four arrays together use 48 data bytes. Exclude scalar state, Python containers, activations, temporary arrays, retained old/new parameter versions, and allocator/cache overhead from these simple totals; actual memory can be larger. P is not the number of parameter leaves or B*T targets.

Authority: [MLX v0.32.2 optimizers.py](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/optimizers/optimizers.py), including Adam and the base Optimizer init/apply_gradients/state implementation. Prepare and publish materials only; all code cells remain unexecuted with empty outputs. No corpus read/download, Notebook execution, tests, training, experiments, sampling, user work copy execution, or exercise answers. Stop at 17.4 before 17.5; publication and static review do not imply mastery or runtime validation. Future requests proceed individually through major Step 20, then stop.


## Step 17.5 accuracy boundary

Current master: `notebooks/76_step17_5_uncompiled_training_step.ipynb`. Initial preparation count: **0/2**. The material reviewer independently accepted 17.4's actual master, documentation, fixed-version MLX API, and full reply with core static PASS and corrections **0/2**. Preserve 16.1=1/2, 16.2=1/2 (including the previously recorded reply completion after a timeout), 16.3–16.5=0/2, and 17.1–17.4=0/2 across turns and conversations. This is material acceptance, not learner learning or execution feedback. The sequence of major Step 17 materials now reaches its last planned subsection; do not infer external acceptance of 17.5 or learner completion.

Follow only the existing 17.5 topics: forward, scalar loss, reverse-mode autodiff, optimizer update, and an explicit evaluation boundary in one uncompiled training step. Stop at major Step 17, before Step 18. Give only the minimal evaluation semantics required to understand the step; do not teach graph-growth experiments, asynchronous evaluation, compilation, mutable state under compilation, memory instrumentation, or later curriculum topics. Do not rework passed material.

Use a self-contained work copy with `mlx.core as mx`, `mlx.nn as nn`, and `mlx.optimizers as optim`. Restore the complete 17.3 SharedLogitModel(nn.Module): `super().__init__()`, public `head={'bias':mx.array([0.,0.,0.],dtype=mx.float32)}`, and __call__ that obtains B,T from X.shape then broadcasts the shared bias to `(B,T,3)`. Explain class, self, callable instances, dictionary paths, and broadcast before code as needed. This toy only reads X.shape, not token content/context; it is an interface example rather than MiniGPT or an initialization recommendation.

Restore explicit chunks `[[0,2,1],[1,0,2]]`, X=chunks[:,:-1], Y=chunks[:,1:]. X/Y are `(2,2)`, and flattened targets are `[2,1,0,2]`. Keep data outside the Module and do not shift Y again. Define loss_fn(model,X,Y) with model(X) inside the function; reshape raw logits from `(B,T,V)` to `(N,V)` and Y to `(N,)` with N=B*T, call cross_entropy(axis=-1,reduction='none'), and return mx.mean(token_losses) as an MLX scalar array. Do not convert the differentiable return with item()/float(), close over fixed unrelated logits, or divide gradients by N again.

Introduce reverse-mode autodiff as following forward dependencies backward with the chain rule: local derivatives multiply along a path and contributions add when paths meet. It is not reversing token order, finding an inverse function, or finite differences. Define r (flattened position), v (class), y_r (integer target), z_(r,v) (logit), p_(r,v) (softmax probability), and q_(r,v) (one-hot indicator) before the math. Begin with dL/dL=1; mean contributes dL/dell_r=1/N; cross-entropy then gives dL/dz_(r,v)=(p_(r,v)-q_(r,v))/N. Because z_(r,v)=theta_v for every position in this toy, broadcast contributions sum over r to give the three parameter gradients. At theta0=[0,0,0] these are [1/12,1/12,-1/6]. Do not hand-fill the gradient tree in this lesson's step: the prepared value_and_grad call obtains it from the defined loss.

Outside the single-step function, create one consistent group: `model=SharedLogitModel()`, `loss_and_grad_fn=nn.value_and_grad(model,loss_fn)`, explicit Adam(learning_rate=1e-3,betas=[0.9,0.999],eps=1e-8,bias_correction=True), and optimizer.init(model.trainable_parameters()). The fixed MLX v0.32.2 bias_correction default remains False and learning_rate remains required; True is the lesson's inherited explicit choice from 17.4. init prepares a fresh zero state without increasing step. Do not recreate Adam inside every training step, and do not reuse a transformed function bound to an old model after replacing the model. Rebuilding the first-step example requires rebuilding model, transformation, and optimizer together.

Explain Python function objects, parameters, calls, tuple unpacking, method side effects, and return values before composing the function. Use `def train_step(model,X,Y,loss_and_grad_fn,optimizer): loss_before_update,grads=loss_and_grad_fn(model,X,Y); optimizer.update(model,grads); mx.eval(loss_before_update,model.parameters(),optimizer.state); return loss_before_update`, written as normal indented lines in materials. The transformed call contains the forward/loss and reverse-mode work; do not add a redundant forward/loss call ahead of it. The same model instance supplies differentiated parameters and is passed into loss_fn. Read the returned gradient tree rather than attaching .grad or calling backward()/zero_grad().

Optimizer.update(model,grads) differs from the 17.4 apply_gradients API: it installs new parameters into the Module and advances optimizer state, returning None. Never write model=optimizer.update(...). Give the minimum MLX evaluation fact before mx.eval: executing a Python array-operation line does not necessarily finish all numeric work. mx.eval accepts multiple arrays or array-containing dict/list/tuple trees, evaluates the requested arrays and their dependencies to completion, and returns None. Select the existing scalar loss, current model.parameters(), and optimizer.state so the explicit boundary covers both the reported loss and the state being carried forward. Do not describe evaluating only the pre-update loss as an explicit guarantee that the new parameters and optimizer state have completed. Uncompiled here only means the ordinary Python function is not wrapped with mx.compile; it does not mean CPU-only or per-line eager execution.

Distinguish parameter versions in the explanation: the returned loss is L(theta_before;X,Y), while after the update/evaluation boundary the model holds theta_after and optimizer holds its new m/v/step. Evaluating the previously built loss after optimizer.update does not turn it into a loss recomputed with theta_after. Do not label the returned scalar post-update loss or claim it demonstrates a loss reduction. No extra forward, post-update loss computation, or training loop is prepared.

The single first-call hand table uses loss=ln3, gradient=[1/12,1/12,-1/6], m1=[1/120,1/120,-1/60], v1=[1/144000,1/144000,1/36000], step=1, and theta_after approximately [-0.001,-0.001,+0.001] only when the tiny positive epsilon is neglected for the displayed approximation. The actual prepared Adam retains eps=1e-8 and floating-point rounding. Loss/step are scalar `()`, and parameter/gradient/m/v leaves are `(3,)`. The table is mathematical expectation, not observed output, a runtime assertion, or evidence of learning.

Prepare exactly one call to train_step. Explain array.item() before using it to convert the returned single-element loss to a Python scalar for logging after the explicit evaluation boundary. Keep item outside the differentiable loss and inside no gradient transform. Repeating the call later would advance both model and optimizer state, rather than resetting the first-step example. All definitions and imports must be recoverable in the new copy; no old Notebook execution, corpus, sampler, or prior runtime state is required.

Fixed-version authority: v0.32.2 python/mlx/nn/utils.py, python/mlx/optimizers/optimizers.py, python/src/transforms.cpp, mlx/transforms.cpp, and python/src/array.cpp. Only author and publish material; all notebook code cells stay unexecuted and output-free. No corpus read/download, Notebook or user-work-copy execution, tests, training, experiments, sampling, or exercise answers. Do not claim learner mastery or runtime validation. End at 17.5 and major Step 17; subsequent requests continue one subsection at a time through major Step 20, then stop.
