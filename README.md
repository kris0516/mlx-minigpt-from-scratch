# mlx-minigpt-from-scratch

在 Apple Silicon Mac 上使用 Apple MLX，从基础矩阵乘法开始，逐步构建并理解一个 Decoder-Only MiniGPT 的开源教学项目。

项目目前处于从零教学构建阶段。核心实现使用 Apple MLX，并关注低内存占用、低发热与尽量避免 Swap。

教学主线同时穿插：

```text
MLP → RNN → LSTM/GRU → Seq2Seq → Attention → Transformer → GPT
```

## Course-material status

`✅` means repository lesson material has been prepared; it does **not** mean learner mastery, data acquisition/split success, exercise completion, or runtime validation.

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

Current course-material checkpoint: **Step 19.5 — Baseline Training Run**.

Current master: [86_step19_5_baseline_training_run.ipynb](notebooks/86_step19_5_baseline_training_run.ipynb).

Actual correction counts persist across conversations: Step 16.1 **1/2**, Step 16.2 **1/2**, Step 16.3 **0/2**, Step 16.4 **0/2**, Step 16.5 **0/2**, Step 17.1 **0/2**, Step 17.2 **0/2**, Step 17.3 **0/2**, Step 17.4 **0/2**, Step 17.5 **0/2**, Step 18.1 **0/2**, Step 18.2 **0/2**, Step 18.3 **0/2**, Step 18.4 **0/2**, Step 18.5 **0/2**, Step 19.1 initial version **0/2**. Step 16.2's count includes the learner-confirmed completion of a reply after a timeout. Passed lessons are not reworked and counts do not reset.

All major Step 16, Step 17, and Step 18 materials have independently passed the material reviewer's core static review. The latest acceptance covers Step 18.5's actual five-file commit `57dc27ee121430b11802b7c6f54fa2617ccd710e`, fixed MLX v0.32.2 interfaces, and full teaching reply, with actual corrections **0/2**. This is material acceptance, not learner mastery or runtime feedback. This turn prepares only **Step 19.1**, then stops before 19.2. Earlier acceptance does not automatically extend to 19.1.

Learner completion/mastery, local-cache availability, exercise completion, Notebook execution, and runtime validation are intentionally **not asserted** here. Step 19.1 prepares one explicit, unmeasured hyperparameter candidate: model_width=64, layers=2, heads=4, block_size=64, batch_size=4, and learning_rate=3e-4. It connects these six choices to the existing MiniGPT, sampler, scalar loss, and Adam interfaces, keeping max_seq_len=block_size as this candidate's convention and vocabulary size as actual tokenizer metadata rather than a guessed constant. Pure Python configuration and argument-mapping helpers are self-contained and do not construct models, optimizers, arrays, batches, or training state. The material distinguishes structural constraints and hand-derived shapes from measurements; the old shape toys are not promoted to tuned training baselines. No low-heat, memory, performance, or convergence result is claimed. Only course materials are authored and published; no Notebook, training, code tests, experiments, measurements, sampling, or user-work-copy code has been run, and no corpus has been read or downloaded. Subsequent explicit requests continue one subsection at a time through major Step 20, then stop before Step 21.

## Persistent course documents

- [Full course roadmap](docs/course-roadmap.md)
- [Teaching handoff / continuation guide](docs/teaching-handoff.md)
- [Current Step 19 material](docs/step-19-stable-low-heat-training.md)

教学笔记、代码与实验将按学习进度逐步提交。


Step 19.2's train/validation loop material is published with initial corrections 0/2. The master separates train/validation data, distinguishes pre-update training loss from current-parameter validation loss, and uses MLX v0.32.2 mode/evaluation boundaries. Its sample loop is instructional only; no run results are claimed. Stop at 19.2 before 19.3.


Step 19.3 Training Stability material is published with initial correction count 0/2. It prepares initialization, global gradient norms, optional clipping and non-finite/divergence checks. The lesson has not been run and makes no stability or training-result claims; stop at 19.3 before 19.4.


Step 19.4 Apple Silicon Resource Control material is published with initial correction count 0/2. It distinguishes unified memory, MLX allocator counters, system memory/Swap, conservative sizing, evaluation boundaries and sustained heat. No code was run and no resource, thermal, speed or memory result is claimed. Stop at 19.4 before 19.5.


Step 19.5 Baseline Training Run material is published with initial correction count 0/2. It defines a future reproducible protocol for speed, loss curves, MLX allocator counters and JSON records; no run or measurements were performed. Stop at 19.5 before Step 20.


## Current checkpointing lesson

**Step 20.1 — Save Model Weights**: [Notebook master](notebooks/87_step20_1_save_model_weights.ipynb) · [Step 20 document](docs/step-20-checkpointing-and-recovery.md). Initial material generation count 0/2. No model weights were generated and no Notebook code was executed. Stop at 20.1 before 20.2.


## Current checkpointing lesson

**Step 20.2 — Save Optimizer State**: [Notebook master](notebooks/88_step20_2_save_optimizer_state.ipynb) · [Step 20 document](docs/step-20-checkpointing-and-recovery.md). Initial generation count 0/2. The lesson prepares optimizer-state serialization only; no real state file was created or loaded. Stop at 20.2 before 20.3.


## Current checkpointing lesson

**Step 20.3 — Save Configuration and Metadata**: [Notebook master](notebooks/89_step20_3_save_configuration_metadata.ipynb) · [Step 20 document](docs/step-20-checkpointing-and-recovery.md). Initial generation count 0/2. The lesson prepares metadata serialization only; no metadata file or checkpoint was generated. Stop at 20.3 before 20.4.
