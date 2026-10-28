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
Step 16.4         🚧 Current initial material prepared / corrections 0/2
Step 16.5+           Planned only
```

Current course-material checkpoint: **Step 16.4 — Reproducibility**.

Current master: [70_step16_4_reproducibility.ipynb](notebooks/70_step16_4_reproducibility.ipynb).

Actual correction counts persist across conversations: Step 16.1 **1/2**, Step 16.2 **1/2**, Step 16.3 **0/2**, Step 16.4 initial version **0/2**. Step 16.2's count includes the learner-confirmed completion of a reply after a timeout. The learner independently confirmed that the Step 16.3 master and full reply pass the core-content review; the passed lesson is not reworked.

Learner completion/mastery, local-cache availability, sampling results, reproducibility experiments, Notebook execution, and runtime validation for Step 16.4 are intentionally **not asserted** here. Material preparation stops at Step 16.4 for this turn, before Step 16.5. Subsequent requests continue one subsection at a time through major Step 20, then stop.

## Persistent course documents

- [Full course roadmap](docs/course-roadmap.md)
- [Teaching handoff / continuation guide](docs/teaching-handoff.md)

教学笔记、代码与实验将按学习进度逐步提交。
