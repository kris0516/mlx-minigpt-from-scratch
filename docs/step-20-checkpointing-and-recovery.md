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

## Step 20.3 — Save Configuration and Metadata

Master: [89_step20_3_save_configuration_metadata.ipynb](../notebooks/89_step20_3_save_configuration_metadata.ipynb). Initial generation **0/2**.

The lesson records architecture/configuration, tokenizer/vocabulary fingerprint, initial random seeds, training progress and explicit artifact references. It uses the full MiniGPT definitions and requires the real vocabulary size and token-ID mapping to be supplied from the existing project mapping; it does not read/download the corpus or invent vocabulary values. A canonical sorted `stoi` mapping is hashed with SHA-256 to distinguish vocabularies of the same size but different ID assignments.

The JSON writer creates a UTC-stamped directory using `mkdir(exist_ok=False)` and opens the manifest with exclusive-create mode `x` to avoid overwriting prior metadata. Artifact references require explicit existing files inside the chosen root and record relative path, size and SHA-256. Seed values are distinguished from full RNG state; the metadata explicitly records whether RNG state was saved and defaults that claim to false. Training progress must be supplied as real values; planned updates are not treated as completed updates.

Metadata alone is not a checkpoint and does not restore training. Weight files, optimizer state, Adam hyperparameters/scheduler, RNG state, sampler position, data split and training progress remain distinct pieces of information. All code cells are unexecuted and output-free. No metadata file, checkpoint, or training result was generated. Stop at 20.3 before 20.4.


### Step 20.3 first correction (1/2)

Tokenizer metadata now stores the complete `stoi` character-to-ID mapping and `itos_by_id` character table in addition to the existing SHA-256 fingerprint. A reconstruction helper rebuilds `stoi` directly from `itos_by_id`, verifies vocabulary size, character uniqueness, agreement with the saved `stoi`, and the fingerprint. Future work-copy code can therefore reconstruct the same token-ID mapping from the JSON alone, without an old kernel or re-inferring the vocabulary from corpus text. No corpus was read and no metadata file was generated during this correction.

## Step 20.4 — Resume Training

Master: [90_step20_4_resume_training.ipynb](../notebooks/90_step20_4_resume_training.ipynb). Interrupted initial delivery completed in the first delivery correction (**1/2**). This is material status only, not runtime validation.

The lesson defines a future resume order: read an explicitly selected metadata manifest; reconstruct the exact tokenizer mapping and architecture; validate referenced artifacts by path, size and SHA-256; reconstruct the matching MiniGPT; strictly load weights; initialize the matching Adam configuration/state tree; restore optimizer state using the 20.2 loader; compare Adam step against metadata `completed_updates`; then continue with an explicitly supplied compatible training split and RNG. Missing files or mismatched hashes, shapes, dtypes, architecture, optimizer paths or update counts must stop the process rather than guess or auto-select a latest file.

Initial seeds are not full RNG state. If Python/MLX RNG state, sampler cursor, data ordering/version, scheduler state, or compatible environment details were not saved, the user may be able to continue optimization from matched model/Adam state but must not claim exact trajectory reproduction. The notebook includes the full MiniGPT and sampler definitions, but does not load a corpus, instantiate model/optimizer objects, save/load a checkpoint, or run a resume loop. All code cells remain unexecuted and output-free. Stop at 20.4 before 20.5.


### Step 20.4 second/final correction (2/2)

Artifact resolution now separates `metadata_path` from the original `artifact_root` used when 20.3 created each artifact's `relative_path`; metadata's `run_metadata.json` directory is no longer assumed to be the artifact root. Path containment, file existence, size and SHA-256 checks remain in place.

The master now embeds the required shifted-target scalar cross-entropy, global gradient norm, clipping and finite-value checks. After restoring weights and optimizer state, `make_resumed_train_step(model, optimizer)` creates a fresh `nn.value_and_grad` callable bound to those exact restored objects; it does not use the old 19.5 global closure or create a replacement model. The continuation entry checks Adam step against `completed_updates` before sampling, and after every update requires the step to advance by exactly one before logging the actual step. It does not replay the baseline warm-up. This is a source-only correction: no notebook, checkpoint, data or training code was executed.


### Step 20.4 authorized extra minimal correction

The resume master imports `tree_map` from `mlx.utils`, required by its global gradient clipping helper. The orchestration example now takes `metadata_path` and the independent original `artifact_root` used by Step 20.3 artifact references, and follows the current resolver signatures and restore order. This is an explicitly authorized extra correction beyond the normal two-correction allowance. No notebook or checkpoint operation was executed.


## Step 20.5 — Resume Verification

Master: [91_step20_5_resume_verification.ipynb](../notebooks/91_step20_5_resume_verification.ipynb). Initial generation count **0/2**. This is the final Step 20 subsection.

The lesson prepares small deterministic checks for architecture/config compatibility, parameter-tree equality, same-input logits, Adam state-tree equality, and Adam step versus metadata `completed_updates`. It uses a fixed manually specified token-ID tensor and does not read corpus. It distinguishes tolerance-based numerical agreement from bitwise identity, and requires genuine pre-save reference outputs/state for before/after comparisons. Negative architecture mismatch cases are defined for future use. All code cells are unexecuted and output-free; no checkpoint is saved/loaded/restored and no verification result is claimed. With this publication, the Step 20.1–20.5 material scope is complete; this is not runtime validation. Stop before Step 21.
