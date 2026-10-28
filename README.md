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
Step 16.5         🚧 Current initial material prepared / corrections 0/2
Step 17+            Planned only
```

Current course-material checkpoint: **Step 16.5 — Toy sampling vs industrial input pipelines**.

Current master: [71_step16_5_toy_vs_industrial_input_pipelines.ipynb](notebooks/71_step16_5_toy_vs_industrial_input_pipelines.ipynb).

Actual correction counts persist across conversations: Step 16.1 **1/2**, Step 16.2 **1/2**, Step 16.3 **0/2**, Step 16.4 **0/2**, Step 16.5 initial version **0/2**. Step 16.2's count includes the learner-confirmed completion of a reply after a timeout. Step 16.3's accepted status is retained. The material reviewer independently confirmed core static PASS for the Step 16.4 full reply, master, and Step 16 documentation; this is material acceptance, not learner feedback. Passed lessons are not reworked.

All five subsections of major Step 16 now have prepared material; Step 16.5 is the current initial version, not a claim that every lesson has independently passed acceptance. This turn stops at the end of major **Step 16** and does not enter Step 17.

Learner completion/mastery, local-cache availability, sampling or experiment results, Notebook execution, and runtime validation are intentionally **not asserted** here. Step 16.5 uses self-contained toy inputs and requires no corpus read. Subsequent requests continue one subsection at a time through major Step 20, then stop.

## Persistent course documents

- [Full course roadmap](docs/course-roadmap.md)
- [Teaching handoff / continuation guide](docs/teaching-handoff.md)

教学笔记、代码与实验将按学习进度逐步提交。
