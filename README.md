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
Step 17.5         🚧 Current initial material prepared / corrections 0/2
Step 18+            Planned only
```

Current course-material checkpoint: **Step 17.5 — Uncompiled training step**.

Current master: [76_step17_5_uncompiled_training_step.ipynb](notebooks/76_step17_5_uncompiled_training_step.ipynb).

Actual correction counts persist across conversations: Step 16.1 **1/2**, Step 16.2 **1/2**, Step 16.3 **0/2**, Step 16.4 **0/2**, Step 16.5 **0/2**, Step 17.1 **0/2**, Step 17.2 **0/2**, Step 17.3 **0/2**, Step 17.4 **0/2**, Step 17.5 initial version **0/2**. Step 16.2's count includes the learner-confirmed completion of a reply after a timeout. Passed lessons are not reworked and counts do not reset.

All major Step 16 masters, documents, and full replies; Step 17.1's actual master, documentation, fixed-version MLX API contract, and full reply; Step 17.2's actual master, documentation, and full reply; Step 17.3's actual master, documentation, MLX v0.32.2 API, and full reply; and Step 17.4's actual master, documentation, fixed-version MLX API, and full reply have independently passed the material reviewer's core static review. This is material acceptance, not learner learning or execution feedback. This turn prepares only **Step 17.5**, then stops at major Step 17, before Step 18. The major Step 17 material sequence now reaches its final planned subsection; this does not assert independent acceptance of 17.5 or learner completion.

Learner completion/mastery, local-cache availability, experiment results, Notebook execution, and runtime validation are intentionally **not asserted** here. Step 17.5 supplies a self-contained toy Module, already-shifted data, scalar loss, Module-aware gradient function, and explicit Adam settings to connect one uncompiled training step. It explains reverse-mode autodiff, model/optimizer state updates, and the required explicit evaluation boundary. The prepared call returns the loss from before the update; it is not a post-update loss measurement. All lesson code remains unexecuted, and no corpus, old kernel, sampler, or new dependency is needed. Subsequent requests continue one subsection at a time through major Step 20, then stop.

## Persistent course documents

- [Full course roadmap](docs/course-roadmap.md)
- [Teaching handoff / continuation guide](docs/teaching-handoff.md)

教学笔记、代码与实验将按学习进度逐步提交。
