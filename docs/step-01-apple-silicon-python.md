# Step 01 — 确认 Apple Silicon 原生 Python 环境

## 目标

这一步只做一件事：

> 确认 VS Code 之后运行这个项目时，使用的是 **macOS + Apple Silicon arm64** 的 Python。

暂时不安装 MLX，也不写任何神经网络代码。

---

## 为什么先检查 `arm64`

Apple M 系列芯片本身使用 ARM64 架构。

如果 Python 进程意外运行在 Rosetta 的 x86_64 模式下，后面安装 MLX、调用 Metal GPU、分析性能时都可能出现不必要的问题。

所以第一步先确认：

```text
Mac 硬件        -> Apple Silicon
Python 进程架构 -> arm64
操作系统        -> macOS
```

---

## 本步代码

运行：

```bash
python scripts/step01_system_info.py
```

预期至少看到：

```text
System : Darwin
Machine: arm64
Python : 3.x.x
```

如果 `Machine` 显示为 `x86_64`，先不要继续下一步。

---

## 这一小步需要认识的 3 个概念

### 1. Python interpreter

`python` 不是“代码本身”，而是负责读取并执行 `.py` 文件的 Python 解释器程序。

之后 VS Code 选择哪个 Python interpreter，会直接决定项目实际用哪套 Python 环境和第三方包。

### 2. CPU architecture

这里我们只区分：

- `arm64`：Apple Silicon 原生架构
- `x86_64`：Intel Mac / Rosetta 常见架构

本项目目标是原生 `arm64`。

### 3. 标准库

本步只使用 Python 自带的 `platform` 和 `sys`。

所以即使还没安装 MLX，这段程序也应该能运行。

---

## 完成标准

只有下面三个条件同时满足，Step 01 才完成：

- [ ] VS Code 能打开本仓库
- [ ] 脚本能够成功运行
- [ ] 输出中 `System = Darwin` 且 `Machine = arm64`

下一步再处理项目自己的 Python 虚拟环境。
