# mlx-minigpt-from-scratch

在 Apple Silicon Mac 上使用 Apple MLX，从基础矩阵乘法开始，逐步构建并理解一个 Decoder-Only MiniGPT 的开源教学项目。

项目目前处于从零教学构建阶段。核心实现使用 Apple MLX，并关注低内存占用、低发热与尽量避免 Swap。

教学主线同时穿插：

```text
MLP → RNN → LSTM/GRU → Seq2Seq → Attention → Transformer → GPT
```

## Course-material status

`✅` below means the repository lesson material has been prepared. It does **not** mean the learner has mastered the topic or completed its exercises.

```text
Steps 1–8.5  ✅ Material prepared
Step 9.1      🚧 Current material prepared / under correction
Step 9.2+        Planned only
```

Current course checkpoint: **Step 9.1 — Backpropagation Through Time**.

Learner completion/mastery for Step 9.1 is intentionally **not asserted** here.

## Persistent course documents

- [Full course roadmap](docs/course-roadmap.md)
- [Teaching handoff / continuation guide](docs/teaching-handoff.md)

教学笔记、代码与实验将按学习进度逐步提交。
