# Local Notebook Workspace

This directory is for personal learning copies of the tracked course notebooks.

## Rule

- `notebooks/*.ipynb`: tracked course master notebooks
- `notebooks/work/*.ipynb`: local working copies, ignored by Git

Create a working copy before running or editing a lesson:

```bash
cp notebooks/02_tensor_basics.ipynb notebooks/work/02_tensor_basics_work.ipynb
```

Open and run the `_work.ipynb` copy in VS Code.

This keeps local execution counts, outputs, experiments, and extra cells from blocking future `git pull` operations.
