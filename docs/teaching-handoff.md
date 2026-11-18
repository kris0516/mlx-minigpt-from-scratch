> **Authoritative handoff: Step 25.5 material-complete, remote main published; no runtime validation or formal tag.** See course-progress.md and RELEASE_CHECKLIST.md.

# Teaching handoff — final material checkpoint

Step 21–25 teaching materials have now been prepared and published, with Step 23.1–25.5 each in a separate commit. Stop after 25.5; do not infer runtime tests or learner mastery. Use Chinese teaching plus English terms, motivation/minimum math/hand examples/shapes and unexecuted code. No course code execution, tests, training, validation, sampling, benchmarks, corpus access, checkpoint operations or work-copy execution. Exercises remain unanswered.

Use separate git pull and timestamped cp -n into the tracked notebooks/work directory. Do not overwrite existing notes or add recurring mkdir. Masters remain output-free, execution_count=null. Preserve historical accepted content and cumulative correction counts, with at most two substantive correction rounds.

Historical detailed teaching rules and counts are preserved in [the archived handoff](history/teaching-handoff-through-step20.md). Its former stops/current-state statements are superseded. The final consolidated handoff **below is already active** since Step 25.5 material publication. It records material continuity only, not learner completion or runtime test success.

## 23.1 checkpoint
Published `src/minigpt/data.py` and `notebooks/102_step23_1_data_module.ipynb`; proceed 23.2 model module without changing baseline architecture. Correction count new initial material 0/2. Historical ledgers unchanged.

## 23.2 checkpoint
Extracted unchanged baseline MiniGPT classes into src/minigpt/model.py. Next: 23.3. Initial corrections 0/2; historical counts preserved.

## 23.3 checkpoint
Published src/minigpt/train.py and notebooks/104_step23_3_training_module.ipynb; static only, initial corrections 0/2. Historical counts unchanged. Proceed in order.

## 23.4 checkpoint
Published src/minigpt/generate.py and notebooks/105_step23_4_generation_module.ipynb; static only, initial corrections 0/2. Historical counts unchanged. Proceed in order.

## 23.5 checkpoint
Added config.py, cli.py, __main__.py and master 106. Explicit CLI commands are future opt-in; weights-only-for-inference artifacts are not full training-resume checkpoints. Next: 24.1. Initial corrections 0/2; preserve prior counts.

## 24.1 checkpoint
Added tests/test_data.py, tests/test_model.py, tests/test_checkpoint.py, tests/test_generate.py and notebooks/107_step24_1_unit_tests.ipynb; static only, no runtime. Corrections initial 0/2.

## 24.2 checkpoint
Added scripts/tiny_overfit.py and notebooks/108_step24_2_tiny_overfit_test.ipynb; static only, no runtime. Corrections initial 0/2.

## 24.3 checkpoint
Published scripts/benchmark_memory.py and notebooks/109_step24_3_memory_benchmark.ipynb. Initial correction 0/2, no execution. Continue to next item.

## 24.4 checkpoint
Published scripts/benchmark_performance.py and notebooks/110_step24_4_performance_thermal_benchmark.ipynb. Initial correction 0/2, no execution. Continue to next item.

## 24.5 checkpoint
Published docs/benchmark-report-template.md and notebooks/111_step24_5_readme_benchmark_table.ipynb. Initial correction 0/2, no execution. Continue to next item.

## Step 25.1 checkpoint
README rewritten to newcomer-friendly quick start, baseline architecture, source and unmeasured benchmark table; existing historical docs preserved. Next 25.2. Initial corrections 0/2.

## Step 25.2 checkpoint
Published docs/INDEX.md with Step 1–25 navigation and actual current Notebook paths. Future 25.3–25.5 entries are not yet published, update index at finish. Initial corrections 0/2. Next 25.3.

## Step 25.3 checkpoint
Wrote docs/REPRODUCIBILITY.md and read-only scripts/static_release_audit.py. Remote pin/index/work-ignore checks were read; no audit script, install, tests or smoke commands run. Next 25.4, initial correction 0/2.

## Step 25.4 checkpoint
MIT LICENSE read and preserved; docs/CREDITS.md published with external references and provenance boundaries. No third-party files or data downloaded. Corrections initial 0/2. Next 25.5.


## Final consolidated continuation state (2026-11-18)

- Step 22.5 independent static acceptance is user-provided; historical records and correction counts are unchanged.
- At initial publication Step 23.1–25.5 substantive correction counts were 0/2 each; after the 2026-11-18 independent review correction, Step 23.1–23.5, 24.1–24.2 and 25.3–25.5 are now **1/2** each. Untouched 24.3–24.5 and 25.1–25.2 remain **0/2**. Material/static review never implies runtime or learner acceptance. Preserve older records and correction history across future sessions.
- [Complete masters and commit ledger](RELEASE_NOTES.md); [course INDEX](INDEX.md); [source refactor](step-23-source-refactor.md); [testing protocols](step-24-testing-benchmarking.md); [release checklist](RELEASE_CHECKLIST.md).
- Environment: Python 3.12, native Apple Silicon, MLX 0.32.2, uv.lock; work notebooks ignored. Two separate terminal commands start each lesson: git pull; timestamped cp -n to notebooks/work/.
- No Notebook or course code, test, training, sampling, benchmark, data access, checkpoint or work-copy operations performed while authoring Steps 23–25.
- Main published; optional experiments, performance measurements, full checkpoint compatibility, actual release tag and GitHub Release remain unresolved runtime gates. Do not automatically enter extension track.


## 2026-11-18 centralized static-review correction (current)

- Notebook 102–106 imports locate the repository upward from kernel CWD (root/notebooks/work) or raise explicit FileNotFoundError; no installation mode switch.
- Corrected Step 23 run_training local record vs accumulated Adam step and evaluation failure boundary; Step 23.4 full-vocabulary sampling and mode/key semantics; Step 23.5 CLI guard/head mapping and weights-metadata pairing caveat.
- Step 24.1 does not assert independent single-head coverage; Step 24.2 separates loss-drop smoke from strict toy memorization target (100% token accuracy AND final CE ≤ 0.1), unmeasured.
- Step 25 static checker handles partial `.py`/eligible Notebook Python-cell AST and selected structure, with old missing IDs and IPython magics reported; **not full schema validation, not run**. Step 25.4 credits link uses absolute GitHub URL.
- Step 22.5 and earlier untouched; all current changes are corrections for independent review, not proof of code execution or course mastery. No tag or Release.


## 2026-11-18 final two teaching clarifications (second correction, active tally)

- Step **23.3 = 2/2**: caller ensures the train-step closure and run-training receive the same model instance; train_rng/val_rng are separate `random.Random` instances. Neither identity relationship is enforced by `run_training`.
- Step **23.5 = 2/2**: `@dataclass(frozen=True)` constructor/ordinary-assignment constraint, `__post_init__` validation, `asdict` and the actual `MiniGPT(**model_cfg.as_model_kwargs())` CLI pathway are now explained.
- Historical first-round **1/2** records above are preserved. All other correction counts unchanged; Step 22.5 and earlier untouched. Two Markdown-only lesson supplements, no Notebook/test/training/generation/benchmark execution; no work copies, tag or Release.
