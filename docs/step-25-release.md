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
