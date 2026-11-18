# Complete Learning Index — Steps 1–25

> This index lists published curriculum source files, **not** runtime results or learner mastery. As of Step 25.4, materials are published through 25.4; final 25.5 is next. No Notebook, test, training, sampling, benchmark, fresh-clone or smoke test has been executed in the publication process.

## How to use this index

For each master: start with **git pull** in the repository root, then use a unique timestamped **cp -n** copy to the ignored **notebooks/work/** directory. Only run or edit that work copy, with Python 3.12, project .venv, MLX v0.32.2 on native Apple Silicon.

## Major Step documents and actual notebook masters

### Step 1 — Apple Silicon native environment

[Course note](../docs/step-01-apple-silicon-python.md) · [System probe](../scripts/step01_system_info.py)

- No numbered Notebook master for this Step; use its course note.

### Step 2 — Python / Notebook / MLX environment

[Course note](../docs/step-02-python-environment.md)

- [01_environment](../notebooks/01_environment.ipynb)

### Step 3 — Tensor foundations

[Course note](../docs/step-03-tensor-basics.md)

- [02_tensor_basics](../notebooks/02_tensor_basics.ipynb)
- [03_reshape](../notebooks/03_reshape.ipynb)
- [04_transpose](../notebooks/04_transpose.ipynb)
- [05_broadcasting](../notebooks/05_broadcasting.ipynb)
- [06_broadcasting_rules](../notebooks/06_broadcasting_rules.ipynb)
- [07_reduction_keepdims](../notebooks/07_reduction_keepdims.ipynb)

### Step 4 — Linear & MLP

[Course note](../docs/step-04-neural-network-basics.md)

- [08_step04_1_linear](../notebooks/08_step04_1_linear.ipynb)
- [09_step04_2_parameters_bias_module](../notebooks/09_step04_2_parameters_bias_module.ipynb)
- [10_step04_3_nonlinearity_relu_gelu](../notebooks/10_step04_3_nonlinearity_relu_gelu.ipynb)
- [11_step04_4_mlp](../notebooks/11_step04_4_mlp.ipynb)

### Step 5 — Learning and gradients

[Course note](../docs/step-05-learning-basics.md)

- [12_step05_1_prediction_target_loss](../notebooks/12_step05_1_prediction_target_loss.ipynb)
- [13_step05_2_minimal_calculus](../notebooks/13_step05_2_minimal_calculus.ipynb)
- [14_step05_3_gradient_autodiff](../notebooks/14_step05_3_gradient_autodiff.ipynb)
- [15_step05_4_gradient_descent](../notebooks/15_step05_4_gradient_descent.ipynb)
- [16_step05_5_what_learning_means](../notebooks/16_step05_5_what_learning_means.ipynb)

### Step 6 — Tokenizer and next-token supervision

[Course note](../docs/step-06-text-next-token.md)

- [17_step06_1_character_tokenizer](../notebooks/17_step06_1_character_tokenizer.ipynb)
- [18_step06_2_onehot_embedding](../notebooks/18_step06_2_onehot_embedding.ipynb)
- [19_step06_3_next_token_shift](../notebooks/19_step06_3_next_token_shift.ipynb)
- [20_step06_4_context_window_block_size](../notebooks/20_step06_4_context_window_block_size.ipynb)
- [21_step06_5_sampling_batches](../notebooks/21_step06_5_sampling_batches.ipynb)

### Step 7 — Embeddings and positions

[Course note](../docs/step-07-embeddings-sequence-representations.md)

- [22_step07_1_nn_embedding](../notebooks/22_step07_1_nn_embedding.ipynb)
- [23_step07_2_embedding_geometry](../notebooks/23_step07_2_embedding_geometry.ipynb)
- [24_step07_3_btc_representation](../notebooks/24_step07_3_btc_representation.ipynb)
- [25_step07_4_position_information](../notebooks/25_step07_4_position_information.ipynb)
- [26_step07_5_token_position_addition](../notebooks/26_step07_5_token_position_addition.ipynb)

### Step 8 — Vanilla RNN

[Course note](../docs/step-08-rnn-basics.md)

- [27_step08_1_why_mlp_is_awkward_for_sequences](../notebooks/27_step08_1_why_mlp_is_awkward_for_sequences.ipynb)
- [28_step08_2_vanilla_rnn_equation](../notebooks/28_step08_2_vanilla_rnn_equation.ipynb)
- [29_step08_3_time_loop](../notebooks/29_step08_3_time_loop.ipynb)
- [30_step08_4_mlx_rnn_cell](../notebooks/30_step08_4_mlx_rnn_cell.ipynb)
- [31_step08_5_rnn_tradeoffs](../notebooks/31_step08_5_rnn_tradeoffs.ipynb)

### Step 9 — BPTT, LSTM and GRU

[Course note](../docs/step-09-bptt-lstm-gru.md)

- [32_step09_1_bptt](../notebooks/32_step09_1_bptt.ipynb)
- [33_step09_2_vanishing_exploding_gradients](../notebooks/33_step09_2_vanishing_exploding_gradients.ipynb)
- [34_step09_3_lstm_cell_state_gates](../notebooks/34_step09_3_lstm_cell_state_gates.ipynb)
- [35_step09_4_gru](../notebooks/35_step09_4_gru.ipynb)
- [36_step09_5_recurrent_limits](../notebooks/36_step09_5_recurrent_limits.ipynb)

### Step 10 — Seq2Seq and attention origin

[Course note](../docs/step-10-seq2seq-attention.md)

- [37_step10_1_encoder_decoder_seq2seq](../notebooks/37_step10_1_encoder_decoder_seq2seq.ipynb)
- [38_step10_2_fixed_vector_bottleneck](../notebooks/38_step10_2_fixed_vector_bottleneck.ipynb)
- [39_step10_3_bahdanau_attention_intuition](../notebooks/39_step10_3_bahdanau_attention_intuition.ipynb)
- [40_step10_4_score_softmax_weighted_sum](../notebooks/40_step10_4_score_softmax_weighted_sum.ipynb)
- [41_step10_5_historical_bridge_to_transformer](../notebooks/41_step10_5_historical_bridge_to_transformer.ipynb)

### Step 11 — Self-attention Q/K/V

[Course note](../docs/step-11-self-attention.md)

- [42_step11_1_why_separate_qkv](../notebooks/42_step11_1_why_separate_qkv.ipynb)
- [43_step11_2_qkv_linear_projections](../notebooks/43_step11_2_qkv_linear_projections.ipynb)
- [44_step11_3_qkt_compatibility_score](../notebooks/44_step11_3_qkt_compatibility_score.ipynb)
- [45_step11_4_scaling_softmax](../notebooks/45_step11_4_scaling_softmax.ipynb)
- [46_step11_5_weighted_value_retrieval](../notebooks/46_step11_5_weighted_value_retrieval.ipynb)

### Step 12 — Causal mask and stable softmax

[Course note](../docs/step-12-causal-self-attention.md)

- [47_step12_1_autoregressive_constraint](../notebooks/47_step12_1_autoregressive_constraint.ipynb)
- [48_step12_2_causal_mask](../notebooks/48_step12_2_causal_mask.ipynb)
- [49_step12_3_stable_attention_computation](../notebooks/49_step12_3_stable_attention_computation.ipynb)
- [50_step12_4_single_causal_attention_head](../notebooks/50_step12_4_single_causal_attention_head.ipynb)
- [51_step12_5_inspect_attention_matrix](../notebooks/51_step12_5_inspect_attention_matrix.ipynb)

### Step 13 — Multi-head attention

[Course note](../docs/step-13-multi-head-attention.md)

- [52_step13_1_why_multiple_heads](../notebooks/52_step13_1_why_multiple_heads.ipynb)
- [53_step13_2_fused_qkv_projection](../notebooks/53_step13_2_fused_qkv_projection.ipynb)
- [54_step13_3_split_channels_into_heads](../notebooks/54_step13_3_split_channels_into_heads.ipynb)
- [55_step13_4_transpose_batched_attention](../notebooks/55_step13_4_transpose_batched_attention.ipynb)
- [56_step13_5_concatenate_heads_output_projection](../notebooks/56_step13_5_concatenate_heads_output_projection.ipynb)

### Step 14 — Transformer block

[Course note](../docs/step-14-transformer-block.md)

- [57_step14_1_residual_connection](../notebooks/57_step14_1_residual_connection.ipynb)
- [58_step14_2_layernorm_to_rmsnorm](../notebooks/58_step14_2_layernorm_to_rmsnorm.ipynb)
- [59_step14_3_pre_norm_vs_post_norm](../notebooks/59_step14_3_pre_norm_vs_post_norm.ipynb)
- [60_step14_4_feed_forward_mlp](../notebooks/60_step14_4_feed_forward_mlp.ipynb)
- [61_step14_5_implement_test_one_block](../notebooks/61_step14_5_implement_test_one_block.ipynb)

### Step 15 — Decoder-only MiniGPT

[Course note](../docs/step-15-transformer-to-gpt.md)

- [62_step15_1_original_transformer_architecture](../notebooks/62_step15_1_original_transformer_architecture.ipynb)
- [63_step15_2_why_decoder_only](../notebooks/63_step15_2_why_decoder_only.ipynb)
- [64_step15_3_minigpt_top_level_architecture](../notebooks/64_step15_3_minigpt_top_level_architecture.ipynb)
- [65_step15_4_implement_minigpt](../notebooks/65_step15_4_implement_minigpt.ipynb)
- [66_step15_5_full_forward_shape_trace](../notebooks/66_step15_5_full_forward_shape_trace.ipynb)

### Step 16 — Tiny Shakespeare data

[Course note](../docs/step-16-tiny-shakespeare-data.md)

- [67_step16_1_acquire_inspect_tiny_shakespeare](../notebooks/67_step16_1_acquire_inspect_tiny_shakespeare.ipynb)
- [68_step16_2_train_validation_split](../notebooks/68_step16_2_train_validation_split.ipynb)
- [69_step16_3_batch_sampler](../notebooks/69_step16_3_batch_sampler.ipynb)
- [70_step16_4_reproducibility](../notebooks/70_step16_4_reproducibility.ipynb)
- [71_step16_5_toy_vs_industrial_input_pipelines](../notebooks/71_step16_5_toy_vs_industrial_input_pipelines.ipynb)

### Step 17 — Language-model loss and Adam

[Course note](../docs/step-17-language-model-loss.md)

- [72_step17_1_logits_probabilities_cross_entropy](../notebooks/72_step17_1_logits_probabilities_cross_entropy.ipynb)
- [73_step17_2_flatten_bt_supervision](../notebooks/73_step17_2_flatten_bt_supervision.ipynb)
- [74_step17_3_nn_value_and_grad](../notebooks/74_step17_3_nn_value_and_grad.ipynb)
- [75_step17_4_adam_optimizer](../notebooks/75_step17_4_adam_optimizer.ipynb)
- [76_step17_5_uncompiled_training_step](../notebooks/76_step17_5_uncompiled_training_step.ipynb)

### Step 18 — MLX execution boundaries

[Course note](../docs/step-18-mlx-execution-model.md)

- [77_step18_1_lazy_evaluation](../notebooks/77_step18_1_lazy_evaluation.ipynb)
- [78_step18_2_eval_boundary](../notebooks/78_step18_2_eval_boundary.ipynb)
- [79_step18_3_mx_compile](../notebooks/79_step18_3_mx_compile.ipynb)
- [80_step18_4_mutable_training_state](../notebooks/80_step18_4_mutable_training_state.ipynb)
- [81_step18_5_memory_instrumentation](../notebooks/81_step18_5_memory_instrumentation.ipynb)

### Step 19 — Low-resource training plans

[Course note](../docs/step-19-stable-low-heat-training.md)

- [82_step19_1_hyperparameter_baseline](../notebooks/82_step19_1_hyperparameter_baseline.ipynb)
- [83_step19_2_train_validation_loop](../notebooks/83_step19_2_train_validation_loop.ipynb)
- [84_step19_3_training_stability](../notebooks/84_step19_3_training_stability.ipynb)
- [85_step19_4_apple_silicon_resource_control](../notebooks/85_step19_4_apple_silicon_resource_control.ipynb)
- [86_step19_5_baseline_training_run](../notebooks/86_step19_5_baseline_training_run.ipynb)

### Step 20 — Checkpoints and recovery

[Course note](../docs/step-20-checkpointing-and-recovery.md)

- [87_step20_1_save_model_weights](../notebooks/87_step20_1_save_model_weights.ipynb)
- [88_step20_2_save_optimizer_state](../notebooks/88_step20_2_save_optimizer_state.ipynb)
- [89_step20_3_save_configuration_metadata](../notebooks/89_step20_3_save_configuration_metadata.ipynb)
- [90_step20_4_resume_training](../notebooks/90_step20_4_resume_training.ipynb)
- [91_step20_5_resume_verification](../notebooks/91_step20_5_resume_verification.ipynb)

### Step 21 — Autoregressive generation

[Course note](../docs/step-21-autoregressive-generation.md)

- [92_step21_1_generation_loop](../notebooks/92_step21_1_generation_loop.ipynb)
- [93_step21_2_greedy_vs_sampling](../notebooks/93_step21_2_greedy_vs_sampling.ipynb)
- [94_step21_3_temperature](../notebooks/94_step21_3_temperature.ipynb)
- [95_step21_4_top_k_sampling](../notebooks/95_step21_4_top_k_sampling.ipynb)
- [96_step21_5_context_cropping](../notebooks/96_step21_5_context_cropping.ipynb)

### Step 22 — Modern GPT optional variants

[Course note](../docs/step-22-modern-gpt-improvements.md)

- [97_step22_1_weight_tying](../notebooks/97_step22_1_weight_tying.ipynb)
- [98_step22_2_rotary_position_embeddings](../notebooks/98_step22_2_rotary_position_embeddings.ipynb)
- [99_step22_3_swiglu_gated_mlp](../notebooks/99_step22_3_swiglu_gated_mlp.ipynb)
- [100_step22_4_mqa_gqa_attention](../notebooks/100_step22_4_mqa_gqa_attention.ipynb)
- [101_step22_5_flash_style_attention](../notebooks/101_step22_5_flash_style_attention.ipynb)

### Step 23 — Source refactor

[Course note](../docs/step-23-source-refactor.md)

- [102_step23_1_data_module](../notebooks/102_step23_1_data_module.ipynb)
- [103_step23_2_model_module](../notebooks/103_step23_2_model_module.ipynb)
- [104_step23_3_training_module](../notebooks/104_step23_3_training_module.ipynb)
- [105_step23_4_generation_module](../notebooks/105_step23_4_generation_module.ipynb)
- [106_step23_5_configuration_cli](../notebooks/106_step23_5_configuration_cli.ipynb)

### Step 24 — Tests and benchmark protocols

[Course note](../docs/step-24-testing-benchmarking.md)

- [107_step24_1_unit_tests](../notebooks/107_step24_1_unit_tests.ipynb)
- [108_step24_2_tiny_overfit_test](../notebooks/108_step24_2_tiny_overfit_test.ipynb)
- [109_step24_3_memory_benchmark](../notebooks/109_step24_3_memory_benchmark.ipynb)
- [110_step24_4_performance_thermal_benchmark](../notebooks/110_step24_4_performance_thermal_benchmark.ipynb)
- [111_step24_5_readme_benchmark_table](../notebooks/111_step24_5_readme_benchmark_table.ipynb)

### Step 25 — Release preparation

[Course note](../docs/step-25-release.md)

- [112_step25_1_readme_rewrite](../notebooks/112_step25_1_readme_rewrite.ipynb)
- [113_step25_2_documentation_index](../notebooks/113_step25_2_documentation_index.ipynb)

- [114_step25_3_reproducibility_audit](../notebooks/114_step25_3_reproducibility_audit.ipynb)
- [115_step25_4_credits_references_license](../notebooks/115_step25_4_credits_references_license.ipynb)

## Historical and mathematical milestones

| Span | Main bridge | Prerequisites that matter |
|---|---|---|
| Steps 3–7 | tensors → linear layers → token representations | matrix multiplication, reshape, broadcasting, reductions |
| Steps 8–10 | MLP → RNN/BPTT → Seq2Seq attention | chain rule, state recurrence, weighted average |
| Steps 11–15 | Q/K/V → causal MHA → GPT | dot products, scale 1/sqrt(d), row softmax, (B,H,T,D) |
| Steps 16–20 | real data → training → MLX eval → checkpoints | shifted labels, cross entropy, autodiff, explicit state |
| Steps 21–22 | logits → autoregressive generation → modern variants | RNG keys, context cropping, online softmax m/l/u |
| Steps 23–25 | source modules → tests/profiling → release docs | Python imports, CLI, model state and test hygiene |

## Key implementation boundaries

- [Source package](../src/minigpt/) implements untied learned-position dense causal baseline; Step 22 is optional exposition, not a silently activated RoPE/GQA/Flash/KV-cache implementation.
- [Step 24 source and benchmark protocol](step-24-testing-benchmarking.md) is static and has no observed measurements. [Benchmark table](benchmark-report-template.md) uses Not measured.
- [Current progress](course-progress.md), [Roadmap](course-roadmap.md), [Teaching handoff](teaching-handoff.md), [MIT License](../LICENSE).
