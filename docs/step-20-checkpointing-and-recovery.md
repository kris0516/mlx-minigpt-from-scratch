# Step 20 — Checkpointing and Recovery

This document tracks the checkpointing section of the MLX MiniGPT course. All claims below describe published teaching material, not learner mastery or runtime validation.

## Step 20.1 — Save Model Weights

Current master: [87_step20_1_save_model_weights.ipynb](../notebooks/87_step20_1_save_model_weights.ipynb). Initial material generation count: **0/2**.

The lesson explains parameter-tree serialization and file organization using the accepted full MiniGPT model and the fixed MLX v0.32.2 API. `nn.Module.parameters()` returns the nested parameter tree; `model.save_weights(path)` saves model weights based on the extension (`.npz` or `.safetensors`). The master chooses `.safetensors`, explicitly evaluates `model.parameters()` before saving, and creates a unique UTC-stamped run directory with `mkdir(exist_ok=False)` to avoid reusing an existing directory.

The model configuration requires a real `vocab_size` from the same token-ID mapping; the lesson does not read or download data and does not invent a vocabulary size. The master includes the complete MiniGPT definitions from the accepted Step 15.4 source. It does not call model construction or the save function during material preparation and creates no checkpoint. If a future learner constructs a fresh model and saves it without training, that file contains initialization weights, not a trained baseline.

Weights-only serialization does not automatically include Adam m/v/step, RNG states, sampler position, dataset split, training step, or all hyperparameters. Restoring the exact training trajectory requires more state and is outside 20.1. The save path is only a future-work-copy pattern. All code cells remain unexecuted and output-free. Stop at 20.1 before 20.2.


## Step 20.2 — Save Optimizer State

Master: [88_step20_2_save_optimizer_state.ipynb](../notebooks/88_step20_2_save_optimizer_state.ipynb). Initial generation count **0/2**. This lesson focuses on Adam moments and step count. It explains why model weights alone are insufficient to continue the same Adam trajectory, uses the actual optimizer state tree, and prepares a future-work-copy serialization pattern that stores every MLX array leaf in NPZ (including the zero-dimensional uint64 step and current learning_rate snapshot when present), while JSON stores array path/key/shape/dtype metadata and actual Python scalar leaves using MLX v0.32.2 tree utilities and explicit evaluation before saving.

The save/load helpers are definitions only; no model or optimizer is instantiated, and no real state file is written or loaded. The output directory is created exclusively to avoid overwriting old state. Restore requires a compatible model/optimizer structure; model weights, Adam hyperparameters, RNG, sampler position, data split, and training history are separate recovery information. This lesson does not claim complete or exact training restoration. All Notebook code cells are unexecuted and output-free. Stop at 20.2 before 20.3.


### Step 20.2 first correction (1/2)

The serializer now stores all MLX array leaves in NPZ, including Adam m/v, the zero-dimensional uint64 `step`, and the current `learning_rate` array snapshot when present in `optimizer.state`. JSON contains array path/key/shape/dtype metadata plus only actual Python scalar leaves and their Python types. The current learning-rate snapshot does not contain the scheduler rule, future schedule, or all Adam hyperparameters (`betas`, `eps`, `bias_correction`); those must be recorded separately.

Before assigning `optimizer.state = rebuilt`, the loader now compares the reconstructed state with the target optimizer's already-initialized state, path by path: array-vs-non-array identity, array shape and dtype, and exact non-array leaf type. Any mismatch raises before the target state is mutated. This is static material correction only; no state was saved/loaded and no code was executed.
