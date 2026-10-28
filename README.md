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
Step 18.4         🚧 Current initial material prepared / corrections 0/2
Step 18.5+          Planned only
```

Current course-material checkpoint: **Step 18.4 — Mutable training state under compile**.

Current master: [80_step18_4_mutable_training_state.ipynb](notebooks/80_step18_4_mutable_training_state.ipynb).

Actual correction counts persist across conversations: Step 16.1 **1/2**, Step 16.2 **1/2**, Step 16.3 **0/2**, Step 16.4 **0/2**, Step 16.5 **0/2**, Step 17.1 **0/2**, Step 17.2 **0/2**, Step 17.3 **0/2**, Step 17.4 **0/2**, Step 17.5 **0/2**, Step 18.1 **0/2**, Step 18.2 **0/2**, Step 18.3 **0/2**, Step 18.4 initial version **0/2**. Step 16.2's count includes the learner-confirmed completion of a reply after a timeout. Passed lessons are not reworked and counts do not reset.

All major Step 16 and Step 17 materials, and Steps 18.1 through 18.3, have independently passed the material reviewer's core static review. The latest acceptance includes Step 18.3's actual five-file commit `18ced2600e0b6465701b6d30e40b206c56db61ee`, official MLX v0.32.2 source, and full teaching reply, with actual corrections **0/2**. This is material acceptance, not learner learning or execution feedback. This turn prepares only **Step 18.4**, then stops before 18.5. The earlier acceptance does not automatically extend to 18.4.

Learner completion/mastery, local-cache availability, experiment results, Notebook execution, and runtime validation are intentionally **not asserted** here. Step 18.4 restores the complete small model, explicit shifted-token data, scalar loss, gradient function, and initialized Adam from 17.5 in a new self-contained master. It explains model and optimizer state, live container references, and the inputs/outputs state-capture contract of mx.compile. Current state values enter each call, and newly computed array leaves are filled back into the captured containers even when the compiled graph is reused. The prepared graph body returns the pre-update loss while advancing model and optimizer state; an explicit mx.eval outside the compiled call requests that loss and the new state to complete. Only one unexecuted compiled-step call is supplied. All imports, definitions, and data are included; no old kernel, corpus, sampler, or additional dependency is needed. This is prepared material, not a training run, a numerical test, or evidence of a speedup. Performance and memory experiments are outside this subsection. Subsequent requests continue one subsection at a time through major Step 20, then stop.

## Persistent course documents

- [Full course roadmap](docs/course-roadmap.md)
- [Teaching handoff / continuation guide](docs/teaching-handoff.md)
- [Current Step 18 material](docs/step-18-mlx-execution-model.md)

教学笔记、代码与实验将按学习进度逐步提交。
