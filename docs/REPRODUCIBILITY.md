# Reproducibility Audit — Static Evidence Only

> **Status 2026-11-18: audit planning and available GitHub source inspection, not a completed fresh-clone or runtime audit.** No installation, model construction, Notebook execution, unit tests, benchmark, training, sampling, checkout or smoke tests were run in this publication session.

## Verified by reading remote tracked files

- At the Step 25.2 baseline HEAD `1f0001ad2c40877496fa3a2bcbebb1c78252012d`, `pyproject.toml` requires `>=3.12,<3.13` and `mlx==0.32.2`; `uv.lock` contains `mlx` version `0.32.2`.
- `.gitignore` ignores `/notebooks/work/*.ipynb`; no personal `notebooks/work/*.ipynb` files were tracked in the inspected GitHub tree.
- All 113 then-published master Notebook links in `docs/INDEX.md` resolved to paths in the inspected tree.
- Current code is a `src/minigpt/` package but `pyproject.toml` has no build-system/console-script install contract. **Use PYTHONPATH=src**, do not assume `pip install -e .` or bare `minigpt` command works.

These facts are static repository observations; they do not assert successful import, MLX Metal setup, Notebook execution or correct numerical behavior. Future commits require a fresh static re-check of links and output states.

## Future clean-room protocol — NOT EXECUTED

On a compatible native Apple Silicon Mac:

~~~bash
git clone https://github.com/kris0516/mlx-minigpt-from-scratch.git
cd mlx-minigpt-from-scratch
git rev-parse HEAD
uv sync --frozen
~~~

Select `.venv/bin/python` in VS Code; bind the Notebook kernel to that interpreter. Check native `arm64` and Python 3.12 using the existing Step 1/2 material. Do not edit tracked masters; use a timestamped `cp -n` under `notebooks/work/`.

Optional **read-only partial static** audit (still not a runtime test; command NOT executed in this session):

~~~bash
python scripts/static_release_audit.py
~~~

Optional **runtime tests**, only on later explicit authorization:

~~~bash
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_*.py' -v
~~~

Optional **training/generation smoke route**; only if the user supplies a real local UTF-8 corpus and authorizes execution:

~~~bash
PYTHONPATH=src uv run python -m minigpt.cli train --corpus data/tinyshakespeare/input.txt --output runs/unique-smoke-001 --updates 2
PYTHONPATH=src uv run python -m minigpt.cli generate --checkpoint runs/unique-smoke-001 --prompt 'To be' --new-tokens 3
~~~

A short smoke run is not a quality/convergence benchmark. The current CLI artifact is inference weights plus metadata; **no optimizer/RNG exact-resume state** is saved. Use Step 20 if a verified resumable checkpoint is required.

## Validation gates not completed

| Gate | Status | Evidence still required |
|---|---|---|
| Repo tree, pin, master links (Step 25.2 snapshot) | Static inspection done | Refresh for final SHA |
| `.py` AST + eligible plain-Python Notebook cell AST and limited nbformat/ID/output checks | Partial static checker source prepared; NOT RUN | Explicit static-check execution and review of skipped IPython magics / legacy missing IDs, if permitted |
| Fresh clone + `uv sync --frozen` | NOT RUN | Real Apple Silicon shell output |
| Import/Metal availability | NOT RUN | Project `.venv` environment observation |
| Unit / causal / save-load tests | NOT RUN | Captured test results |
| Tiny overfit | NOT RUN | Real initial/final loss curve |
| Training/generation CLI smoke | NOT RUN | Checkpoint files and output |
| Memory/throughput/Swap/thermal | NOT MEASURED | Raw environment-specific records |
| Full release tag and reproducible release signoff | NOT DONE | Tested SHA and tag/release action |

Do not mark a gate PASS based only on checked-in source code. No commands in this document were executed to prepare it.


## Step 25.5 material closure addendum

Masters 102–116, the source package, test/benchmark scripts and release documents have been authored and pushed to the remote GitHub main branch. The earlier Step 25.2 snapshot audit covered 113 masters; final tree should contain 116. **No static_release_audit.py, unit tests, smoke, sample generation, training, benchmarks or fresh clone were executed**; the checked boxes above remain limited to explicitly identified static evidence. The real tag and GitHub Release remain not created.


## 2026-11-18 independent review clarification — audit script scope

The revised `scripts/static_release_audit.py` (still **not executed**) checks Python source AST, parseable plain-Python Notebook code-cell AST, the nbformat major/minor fields, code-cell empty execution/outputs, present valid unique IDs and source/link shape. IPython magic/shell Notebook cells are **counted as skipped**; older masters with missing cell IDs are counted as legacy warnings rather than silently rewriting past material. It is **not** full `nbformat` JSON schema validation, and neither static parsing nor JSON checking proves MLX execution correctness. User work copies are not inspected.
