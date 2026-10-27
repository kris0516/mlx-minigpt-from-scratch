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
Steps 1–8.5  material prepared
Step 9.1      material prepared; static audit passed
Step 9.2      material prepared; static audit passed
Step 9.3      material prepared; static audit passed
Step 9.4      material prepared; static audit passed
Step 9.5      current material prepared
Step 10.1+    planned only; not current
```

**Material status is not learner status.** No claim is made that the learner has completed the Step 9.5 work notebook, completed exercises, or mastered the remaining limitations of LSTM/GRU.

Do not enter Step 10.1 until the learner explicitly asks to proceed beyond Step 9.5.

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

> **Step 9.5 — Why LSTM/GRU still did not solve everything**

Step 9.5 material is prepared only. Do not infer learner mastery or exercise completion, and do not create, teach, or mark Step 10.1 as current unless the learner explicitly asks to proceed.

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
