# Step 20 — Checkpointing and Recovery

This document tracks the checkpointing section of the MLX MiniGPT course. All claims below describe published teaching material, not learner mastery or runtime validation.

## Step 20.1 — Save Model Weights

Current master: [87_step20_1_save_model_weights.ipynb](../notebooks/87_step20_1_save_model_weights.ipynb). Initial material generation count: **0/2**.

The lesson explains parameter-tree serialization and file organization using the accepted full MiniGPT model and the fixed MLX v0.32.2 API. `nn.Module.parameters()` returns the nested parameter tree; `model.save_weights(path)` saves model weights based on the extension (`.npz` or `.safetensors`). The master chooses `.safetensors`, explicitly evaluates `model.parameters()` before saving, and creates a unique UTC-stamped run directory with `mkdir(exist_ok=False)` to avoid reusing an existing directory.

The model configuration requires a real `vocab_size` from the same token-ID mapping; the lesson does not read or download data and does not invent a vocabulary size. The master includes the complete MiniGPT definitions from the accepted Step 15.4 source. It does not call model construction or the save function during material preparation and creates no checkpoint. If a future learner constructs a fresh model and saves it without training, that file contains initialization weights, not a trained baseline.

Weights-only serialization does not automatically include Adam m/v/step, RNG states, sampler position, dataset split, training step, or all hyperparameters. Restoring the exact training trajectory requires more state and is outside 20.1. The save path is only a future-work-copy pattern. All code cells remain unexecuted and output-free. Stop at 20.1 before 20.2.
