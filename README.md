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
Step 18.5         🚧 Current initial material prepared / corrections 0/2
Step 19+           Planned only
```

Current course-material checkpoint: **Step 18.5 — Memory instrumentation**.

Current master: [81_step18_5_memory_instrumentation.ipynb](notebooks/81_step18_5_memory_instrumentation.ipynb).

Actual correction counts persist across conversations: Step 16.1 **1/2**, Step 16.2 **1/2**, Step 16.3 **0/2**, Step 16.4 **0/2**, Step 16.5 **0/2**, Step 17.1 **0/2**, Step 17.2 **0/2**, Step 17.3 **0/2**, Step 17.4 **0/2**, Step 17.5 **0/2**, Step 18.1 **0/2**, Step 18.2 **0/2**, Step 18.3 **0/2**, Step 18.4 **0/2**, Step 18.5 initial version **0/2**. Step 16.2's count includes the learner-confirmed completion of a reply after a timeout. Passed lessons are not reworked and counts do not reset.

All major Step 16 and Step 17 materials, and Steps 18.1 through 18.4, have independently passed the material reviewer's core static review. The latest acceptance covers Step 18.4's actual five-file commit `b3993d819c47c1a3dccf9bd71022db12bafcf7b0` and full teaching reply, with actual corrections **0/2**. This is material acceptance, not learner mastery or runtime validation. This turn prepares only **Step 18.5**, then stops at major Step 18, before Step 19. Earlier acceptance does not automatically extend to 18.5.

Learner completion/mastery, local-cache availability, exercise completion, Notebook execution, and runtime validation are intentionally **not asserted** here. Step 18.5 explains active, peak, and cache memory in the fixed MLX v0.32.2 allocator, including bytes/MiB, the difference between logical array payload and allocator statistics, and the separate scope of whole-system memory and Swap. Its self-contained master restores the complete tiny model, explicit shifted X/Y, scalar loss, gradient helper, Adam, and live training-state containers. It prepares one selected uncompiled or compiled mode per fresh kernel, with the same one-update warm-up and three measured updates, each followed by an external evaluation boundary covering the old-parameter loss and new model/optimizer state. Counter reads and comparison code are written only for later practice; no measurement, training, test, or experiment has been executed, and no measurement values or efficiency conclusion are supplied. Resetting a peak counter does not free memory; memory-cache buffers and compiled-graph cache are different things. Compilation is not a promise of lower memory or faster execution. All imports, definitions, and data are included, without an old kernel, corpus, sampler, or extra dependency. Subsequent requests continue one subsection at a time through major Step 20, then stop.

## Persistent course documents

- [Full course roadmap](docs/course-roadmap.md)
- [Teaching handoff / continuation guide](docs/teaching-handoff.md)
- [Current Step 18 material](docs/step-18-mlx-execution-model.md)

教学笔记、代码与实验将按学习进度逐步提交。
