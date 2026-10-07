# Teaching Handoff

This file is for any future ChatGPT/Codex session continuing the course.

## Read first

Before teaching or editing the project, read:

1. `docs/course-roadmap.md`
2. the most recent Step documentation
3. the corresponding tracked notebook under `notebooks/`

## Current state

Current teaching progress:

```text
Step 1   complete
Step 2   complete
Step 3   complete
Step 4.1 complete
Step 4.2 complete
Step 4.3 complete
Step 4.4 complete
Step 5.1 complete
Step 5.2 complete
Step 5.3 complete
Step 5.4 complete
Step 5.5 complete
Step 6.1 complete
Step 6.2 complete
Step 6.3 complete
Step 6.4 complete
Step 6.5 complete
Step 7.1 complete
Step 7.2 current / lesson created
Next after learner finishes: Step 7.3 — Build the (B,T,C) representation
```

## Non-negotiable teaching constraints

The learner asked for **small, digestible teaching turns**, but not absurdly fragmented micro-steps.

From Step 4 onward:

- maximum 5 subsections per Step;
- the cap is organizational, not a brevity target: each subsection must be sufficiently complete and deep, including prerequisite reasoning, intuitive examples, shape tracing, and important caveats;
- each subsection should contain roughly 2x the content of the earliest Step-3 micro-lessons;
- introduce mathematics only when needed;
- assume only simple matrix multiplication as the starting mathematical baseline;
- use small hand-computable examples;
- use plots where they materially improve intuition (e.g. geometry, distributions, loss curves, optimization, benchmarks), and teach the small plotting technique inline without turning the course into a plotting tutorial;
- explain theory → meaning/history → code;
- explain why a design exists, not only how to call an API;
- explicitly correct oversimplifications when needed;
- avoid dumping an entire architecture in one reply.

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

Normal user workflow:

```bash
git pull
cp notebooks/<lesson>.ipynb notebooks/work/<lesson>_work.ipynb
```

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

Current lesson:

> **Step 7.2 — Embedding geometry**

After the learner finishes it, continue with Step 7.3 — Build the (B,T,C) representation. Do not jump ahead to Attention or training loops.
