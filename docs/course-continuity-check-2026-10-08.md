# Course Continuity Check — 2026-10-08

## Purpose and verification scope

Performed when restarting the course after the learner changed the selected GPT-6 configuration. Read the **live** GitHub `main` tree and all 26 tracked master notebooks existing before Step 8.1, as well as README, `docs/course-roadmap.md`, `docs/teaching-handoff.md`, the Step 01–07 documents, `pyproject.toml`, `.gitignore`, and `notebooks/work/README.md`.

The read confirmed exactly 26 masters, numbered 01–26, concluding with `notebooks/26_step07_5_token_position_addition.ipynb`. The planned next lesson is 8.1. This check is based on repository content; it is **not** a claim that every Notebook was executed on the student's Mac.

## Audited lesson index

- **Environment:** `01_environment.ipynb` and Step 1 system-probe script.
- **Tensor foundations:** `02_tensor_basics.ipynb` through `07_reduction_keepdims.ipynb`.
- **Neural-network foundations:** `08_step04_1_linear.ipynb` through `11_step04_4_mlp.ipynb`. Step 4.1 is the merged canonical lesson preserving matrix math, 1D vector/matmul rules, shared weights, and `nn.Linear` as a callable Module.
- **How learning works:** `12_step05_1_prediction_target_loss.ipynb` through `16_step05_5_what_learning_means.ipynb`.
- **Text-to-supervision:** `17_step06_1_character_tokenizer.ipynb` through `21_step06_5_sampling_batches.ipynb`.
- **Embeddings/position:** `22_step07_1_nn_embedding.ipynb` through `26_step07_5_token_position_addition.ipynb`.

## Teaching contract

- Chinese as primary teaching language with important English terminology.
- Student starting mathematics: basic matrix multiplication; add calculus, linear algebra, and probability only when needed.
- Maximum **five sub-lessons per major Step**; this is not a brevity constraint. Teach complete motivations, mathematics, worked examples, MLX code, shapes, caveats, and small appropriate plots.
- Historical progression: **MLP → RNN → BPTT → LSTM/GRU → Seq2Seq → Attention → Self-Attention → Transformer → GPT**.
- Apple Silicon, Python 3.12, uv, VS Code, native Apple MLX (not a PyTorch core), conservative laptop resources, avoid unnecessary memory/Swap.
- Git-tracked `notebooks/*.ipynb` are assistant-maintained unexecuted masters. Learner runs only ignored `notebooks/work/*_work.ipynb` copies.
- Each new lesson updates notebook, Step note, course roadmap, teaching handoff, and README.
- Interactive terminal guidance is sequential: **one shell command per assistant turn**, wait for and verify the user's terminal output, then provide the next command. `git pull` and the lesson `cp` command must never be bundled.

## Technical boundaries to preserve

- Position-wise MLP cannot exchange information between token positions; a fixed-window flattened MLP can, but has different scaling/variable-length trade-offs. Avoid saying all MLPs are mathematically incapable of sequence modeling.
- Recurrence `h_t=f_theta(x_t,h_(t-1))` introduces parameter sharing and an evolving compressed state, not guaranteed lossless/unlimited memory.
- Step 8.1 is historical motivation; precise trainable Vanilla RNN equation belongs to 8.2, time loop 8.3, MLX RNN Cell 8.4, costs 8.5.
- RNN's sequential time dependency and long-range credit assignment motivate the later BPTT/LSTM/GRU sequence of lessons.
- Embedding width need not exceed vocabulary; dot product is not cosine unless normalized; causal mask has its own position-dependent structure; fixed masks should not enter the trainable parameter tree inadvertently.

## Next

Step 8.1 new master: `notebooks/27_step08_1_why_mlp_is_awkward_for_sequences.ipynb`; after learner finishes, continue Step 8.2 **without skipping the RNN equation and hand calculations**.
