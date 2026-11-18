> **Authoritative material checkpoint: Step 23.1 published; follow course-progress.md.** No Notebook, model, training, benchmark, corpus or work-copy operations were executed.

# Teaching handoff during publication

Continue preparing and publishing the authorized Step 21–25 materials, one subsection per commit; stop after 25.5. Use Chinese teaching plus English terms, motivation/minimum math/hand examples/shapes and unexecuted code. No course code execution, tests, training, validation, sampling, benchmarks, corpus access, checkpoint operations or work-copy execution. Exercises remain unanswered.

Use separate git pull and timestamped cp -n into the tracked notebooks/work directory. Do not overwrite existing notes or add recurring mkdir. Masters remain output-free, execution_count=null. Preserve historical accepted content and cumulative correction counts, with at most two substantive correction rounds.

Historical detailed teaching rules and counts are preserved in [the archived handoff](history/teaching-handoff-through-step20.md). Its former stops/current-state statements are superseded. Final consolidated handoff will activate with Step25.5.

## 23.1 checkpoint
Published `src/minigpt/data.py` and `notebooks/102_step23_1_data_module.ipynb`; proceed 23.2 model module without changing baseline architecture. Correction count new initial material 0/2. Historical ledgers unchanged.

## 23.2 checkpoint
Extracted unchanged baseline MiniGPT classes into src/minigpt/model.py. Next: 23.3. Initial corrections 0/2; historical counts preserved.

## 23.3 checkpoint
Published src/minigpt/train.py and notebooks/104_step23_3_training_module.ipynb; static only, initial corrections 0/2. Historical counts unchanged. Proceed in order.

## 23.4 checkpoint
Published src/minigpt/generate.py and notebooks/105_step23_4_generation_module.ipynb; static only, initial corrections 0/2. Historical counts unchanged. Proceed in order.
