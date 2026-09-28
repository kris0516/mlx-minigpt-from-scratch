# Step 02 — 独立 Python 环境与 Notebook kernel

## 目标

让这个仓库拥有自己独立的 Python 3.12 环境，并确保：

```text
Terminal
VS Code Python interpreter
VS Code Notebook kernel
```

都指向同一个：

```text
mlx-minigpt-from-scratch/.venv/bin/python
```

## 当前配置

- Python: 3.12.x
- 虚拟环境: `.venv`
- Notebook kernel package: `ipykernel`

`ipykernel` 是让 Python 进程能够作为 Jupyter Notebook kernel 运行的包。

Notebook 的第一个验证文件：

```text
notebooks/01_environment.ipynb
```

它只检查解释器路径、Python 版本和 CPU 架构，不涉及 MLX。
