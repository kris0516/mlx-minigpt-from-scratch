# Final Publication Checklist — Step 25.5

This is a **material-publication** checklist, not a runtime signoff.

## Completed by static repository inspection / publication

- [x] Existing Step 22.5 master and independent acceptance record left unchanged.
- [x] Step 23.1–25.5 numbered lesson masters and Chinese teaching notes published in order, one Git commit per subsection.
- [x] Source modules in `src/minigpt/`, standard-library unittest files, short benchmarking scripts, and CLI documentation present.
- [x] `pyproject.toml` and `uv.lock` both require MLX v0.32.2; Python 3.12 constraint documented.
- [x] README and docs/INDEX.md show module map, full course index and correct status semantics.
- [x] GitHub tracked tree inspected: no user `notebooks/work/*.ipynb`, data or generated benchmark records included.
- [x] LICENSE remains MIT, attribution and third-party references documented.
- [x] All measurement template values remain Not measured, not fictitious zeroes.

## NOT DONE: require future explicit runtime authorization

- [ ] Independent full static AST/Notebook audit has been **executed** (script prepared only).
- [ ] Fresh clone + `uv sync --frozen` on a compatible Apple Silicon Mac.
- [ ] Actual import, MLX/Metal availability and unit tests.
- [ ] Tiny overfit, validation and train/generate smoke test.
- [ ] Save/load and full optimizer/RNG resume integration with real state.
- [ ] Memory, throughput, thermal and macOS Swap measurements.
- [ ] Final release-approval decision based on tests and independent review.
- [ ] Actual Git tag `v0.1.0` and optional GitHub Release, both **not created**.

No failed or omitted check was silently treated as a PASS. If a future user wants a production-style release, these unchecked gates must be completed and the target commit verified before tagging.
