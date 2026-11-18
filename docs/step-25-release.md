# Step 25 — GitHub-ready Educational Release

> Documentation and release preparation only. Runtime tests, training, benchmarking, fresh clone, and release tag are not presumed.

## Step 25.1 — Rewrite main README

# Step 25.1 — Rewrite main README

**终端命令一：**

~~~bash
git pull
~~~

**终端命令二（工作副本不覆盖）：**

~~~bash
cp -n notebooks/112_step25_1_readme_rewrite.ipynb notebooks/work/112_step25_1_readme_rewrite_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

MLX v0.32.2、Apple Silicon、Python 3.12。所有代码单元未执行。

## 1. 为什么重写 README？

旧 README 混有大量在 Step 19–20 时写下的停止点与进度声明，即使顶部另有权威 banner，新的读者仍可能误认为项目只到 19.5。现在换成面向实际使用的首页，依次呈现目标、可复现环境、架构图、快速开始、Source 模块、课程路径、benchmark 模板与许可。

README 是面向新访问者的**导航入口**，不是详细讲义、实验结果或 checkpoint 记录。Step 1–22 的历史课程内容仍保留在 docs 与 Notebooks，不因重写而丢失。

## 2. 必须准确写出的契约

真正的模型输入 IDs (B,T) → hidden (B,T,C) → logits (B,T,V)，C=H×d。默认 baseline 保留 learned absolute positions、untied output head、dense causal MHA 和 GELU MLP，不能把 Step 22 的可选 RoPE/GQA/Flash 示例宣传成已集成。

README 的 shell 命令必须分成安装、复制 work、未来主动运行三个类别。加载本地语料是可选显式动作，CLI 输出仅为 inference weights + metadata，不是完整 optimizer resume。

## 3. 纸面阅读示例

只用结构量：C=16、H=4 → d=4；若输入 IDs shape (2,3)，模型输出 shape 应为 (2,3,V)。这些符号足够解释架构图，不能由此填写 README 的实际速度或内存字段。Benchmark 表全部保留 Not measured。

## 4. 验收边界、练习与下一节

静态核对 README 内每个路径是否应存在、复制命令是否使用 cp -n 和时间戳，以及是否存在误导性的运行成功/低热声明。真正的 fresh clone、uv sync、训练或测试都未执行。

**练习（不附答案）：** ① 找出说明 baseline 不是 GQA 的句子；② 解释 Not measured 为何不能改成 0；③ 说明为什么 README 需要区分 release material status 与 runtime validation status。

初始修正 0/2；下一节 25.2 Documentation index。

[Published README](../README.md) uses actual src layout and unmeasured benchmark placeholders.


## Step 25.2 — Documentation Index

# Step 25.2 — Documentation index / 历史与数学导航

**命令 1：**

~~~bash
git pull
~~~

**命令 2：**

~~~bash
cp -n notebooks/113_step25_2_documentation_index.ipynb notebooks/work/113_step25_2_documentation_index_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

MLX v0.32.2 / Apple Silicon / Python 3.12。母版未执行，work 文件不覆盖。

## 1. 新增什么？

Step 25.1 的 README 解决新读者如何进入项目。Step 25.2 则为已有的 1–25 大 Step 建立 docs/INDEX.md，依照远程 tree 实际出现的 notebook 文件名生成路径，避免凭旧记忆编造编号。课程导航同时覆盖早期数学历史链和后期工程链。

索引是**资源导航**，不是先修知识的自动掌握声明。Step 25.3–25.5 此刻仍是计划材料，最终发布前要回写这三个母版的真实路径。

## 2. Shape / 依赖怎样从旧课通向工程课？

Tensor 学到 (B,T,C)，MHA 将它变为 Q/K/V 的 (B,H,T,D)，dense scores 是 (B,H,T,T)，最后 MiniGPT 将整数 IDs (B,T) 映射为 logits (B,T,V)。训练将 (B,T,V) 展平为 (B*T,V)，而生成逐轮只读取最后位置 (1,V)。这条 shape 主线也作为索引中的数学先修提示。

## 3. 纸面路径示例

若读者要理解 Step 22.5 的 running m/l/u，先查 Step 11 (QK/Value)、Step 12.3 (stable softmax)、Step 13 (head-major)、Step 18 (MLX lazy/eval)，最后查 Step 22.4 (GQA) 和 22.5。它是阅读路线，不是要求重新执行旧 Notebook。

## 4. 静态核对、范围和练习

此轮只比较 tree 中的实际文件名、按 Step 分组并生成 docs/INDEX.md，没有任何 Notebook import、运行或数据加载。没有把未来 25.3–25.5 的计划路径说成已经发布。

**练习（不附答案）：** ① 从 INDEX 找到 Step 13.4 的真实母版；② 为什么 Step 20 checkpointing 是 Step 25 CLI 的前置阅读；③ 为 22.5 写出至少四个先修链接。

材料初稿 0/2，下一节 25.3 Reproducibility audit。

[Published complete index](INDEX.md) lists the actual Notebook paths in the current tree and the Step-specific mathematical prerequisites.


## Step 25.3 — Reproducibility Audit

# Step 25.3 — Reproducibility audit（静态检查与待运行步骤）

**命令一：**

~~~bash
git pull
~~~

**命令二（cp -n 安全复制）：**

~~~bash
cp -n notebooks/114_step25_3_reproducibility_audit.ipynb notebooks/work/114_step25_3_reproducibility_audit_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

固定 Apple Silicon、Python 3.12、MLX v0.32.2；本次不执行 Notebook、测试、训练、采样或 benchmark。

## 1. 为什么要有复现审查？

源码已经发布不等于别人能在干净机器上成功安装并运行。Reproducibility（可复现性）涉及 Git SHA、依赖锁定、解释器、语料来源、明确种子、训练状态和运行结果。当前用户禁止安装和运行，因而这一节将可核实的**静态证据**与待验证的 runtime gates 分成两张清单，避免造假。

## 2. 真实源码核对的结果

从远程读取确认 pyproject 指定 Python >=3.12,<3.13 和 mlx==0.32.2；uv.lock 中同样锁定 mlx 0.32.2。notebooks/work/*.ipynb 被 Git 忽略，Step 25.2 时 GitHub tree 不包含用户 work 笔记，INDEX 中当时 113 个 notebook 链接均对应存在文件。

这属于静态可观察事实，不是 uv sync 通过。当前项目没有安装 console script：只有正确设置 PYTHONPATH=src，才能按文档直接使用 python -m minigpt.cli。

## 3. Shape 与 smoke test 的差别

未来 CLI 接受真实字符流 (N,) 与构造出的 X/Y (B,T)，MiniGPT 返回 logits (B,T,V)，loss 是 scalar ()。若只验证 shape 或 import 成功，仍不能证明 grad、Adam、保存/加载或生成正确；这些分别有 Step 24 的测试和 Step 25.3 的未来 smoke 路线。

例子：B=1,T=4,V=3，理论 logits (1,4,3)；只有真正执行后，才能宣称其值可读取。

## 4. 限制、练习与完成条件

完整静态证据、可复现安装与测试命令、未完成 gates 在 docs/REPRODUCIBILITY.md。新增 scripts/static_release_audit.py 只在未来主动调用时读取源码并解析语法；**当前只写入文件，没有调用它**。

**练习（不附答案）：** ① 为什么 uv.lock 存在不代表 uv sync 实际成功？② 解释 git SHA 和随机 seed 解决不同的复现问题；③ 为一次训练/生成 smoke test 设计最小记录表。

初始修正 0/2；下一节 25.4 Credits / license。

[Audit checklist and pending gates](REPRODUCIBILITY.md); [read-only source audit script](../scripts/static_release_audit.py).


## Step 25.4 — Credits and License

# Step 25.4 — Credits / references / MIT license consistency

**命令一：**

~~~bash
git pull
~~~

**命令二（新 work 副本，不覆盖）：**

~~~bash
cp -n notebooks/115_step25_4_credits_references_license.ipynb notebooks/work/115_step25_4_credits_references_license_$(date +%Y%m%d_%H%M%S)_work.ipynb
~~~

Apple Silicon、Python 3.12、MLX v0.32.2。所有源码/Notebook 未执行。

## 1. 为什么要专门核对归属？

开源教学项目混合了本仓库原始教材与代码、外部 MLX API、Tiny Shakespeare 数据来源、经典论文概念。Credits（鸣谢）与 citations（引用）说明来源，不等于给外部作品重新授权。许可一致性检查应以仓库真实 LICENSE 为准，不能看到 README 写 MIT 就猜它存在或可涵盖所有第三方内容。

## 2. 静态发现与来源层级

已读取真实 LICENSE，开头是 MIT License，记录 Copyright (c) 2026 kris0516；本节不改 LICENSE。src/minigpt 的示教实现使用 MLX v0.32.2 API，不把 Apple MLX 库本身当作本项目著作权。docs/CREDITS.md 列出 MLX、Python/uv、Tiny Shakespeare 以及 LSTM、Bahdanau attention、Transformer、RoPE、MQA/GQA、FlashAttention 论文链接。

没有在本轮从网上下载论文全文、外部代码或数据，也不声称完成每个第三方数据发行物的独立许可审查。

## 3. Shape 例子连接论文概念而不混淆实现

Transformer baseline 有 Q/K/V (B,H,T,d)、scores (B,H,T,T) 和 logits (B,T,V)。GQA 可能使用 Hq != Hkv，Flash-style 会改变执行策略；但本仓库默认 model.py 的原始 MHA 仍是 Hq=Hkv。引用 GQA/FlashAttention 论文并不意味着工程实现已经被替换。

## 4. 完成边界与独立练习

[docs/CREDITS.md](CREDITS.md) 是唯一较长引用清单，Notebook 只讲许可分层与误区。pyproject.toml 尚无打包发行的 license 元数据字段，因此不能声称 package wheel 已通过法律/分发审核；Step 25.5 要把它保留为待检查边界。

**练习（不附答案）：** ① 什么属于仓库 MIT 许可、什么属于外部 MLX 自身许可？② 为什么不自动下载语料是重要的来源边界？③ 引用 FlashAttention 与本项目已经集成 fused kernel 是同一回事吗？

仅静态来源核对，初始修正 0/2；下一节 25.5 Final release。

[Published credits and research sources](CREDITS.md), [MIT LICENSE](../LICENSE) unchanged.
