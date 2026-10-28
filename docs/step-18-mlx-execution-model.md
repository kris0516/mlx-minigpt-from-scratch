# Step 18 — MLX Execution Model

沿用既有roadmap，本轮只准备并发布 **18.3 mx.compile**，初始实际修正计数 **0/2**；完成即停，不进入18.4。当前母版：[79_step18_3_mx_compile.ipynb](../notebooks/79_step18_3_mx_compile.ipynb)。

材料审核方已独立核验18.2实际六文件提交 `e3656db90239437cc0d50c24730920f1f783e59c`、固定MLX接口和完整教学回复，核心静态PASS、实际修正0/2。大Step16、17及18.1、18.2已通过材料保留，不返工；这是材料验收，不是学员掌握或运行反馈。18.3只登记当前初始材料编写发布，不继承前节外部验收结论。

## Step 18.1 — Lazy evaluation

母版：[77_step18_1_lazy_evaluation.ipynb](../notebooks/77_step18_1_lazy_evaluation.ipynb)。18.1真实六文件提交 `4dd066c839d8e5d371b6ba1fb9d039d50fd39efa`、固定MLX v0.32.2接口和完整教学回复已由材料审核方独立确认核心静态PASS，实际修正 **0/2**。下文保留已通过的教学及18.1当轮边界。

本节承接17.5显式求值边界，解释数组运算怎样建立计算依赖，以及为什么Python走过一行不等于对应GPU数值工作已完成。全部示例代码仅为材料，未执行、无运行输出。


### 0. 新Work Copy的依赖：只有MLX与本节常量

沿用项目 Python 3.12 / `.venv` kernel，以及 `pyproject.toml` 固定的 `mlx==0.32.2`。`import mlx.core as mx` 导入MLX数组模块，`as mx` 给它一个短名称。

新副本完整提供导入、一个2行3列的常量数组、全部中间运算，以及后文的普通Python函数。不需要旧kernel变量、`%run`旧Notebook、语料缓存、sampler、MiniGPT类、`nn.value_and_grad`或Adam对象；17.5的单步只作为已解释过的接口背景。

以后使用时，从本节导入单元开始按顺序使用即可。当前准备阶段只写这些单元，不执行它们。

```python
import mlx.core as mx

# Self-contained lesson import. All cells in this master are unexecuted.
```

### 1. Lazy Evaluation：先表达依赖，再在需要时取得数值

**Evaluation（求值）** 在这里指取得数组运算的数值结果，不是用验证集评估模型。**Lazy evaluation（惰性求值）** 允许数组运算先描述“这个结果应由哪些输入、通过什么运算产生”，在需要结果值时完成相应的数值工作。

先区分四个相互关联的东西：

| 名称 | 本节含义 |
|---|---|
| **Array object，数组对象** | Python变量可以持有的MLX对象；它可以表示已有数据，也可以表示尚待求值的运算结果 |
| **Metadata，元数据** | 描述数组的信息，例如shape和dtype，不是数组每个位置的具体值 |
| **Dependency，依赖** | 为得到一个结果，需要哪些输入数组，以及怎样对它们运算 |
| **Element values，元素数值** | 例如结果矩阵某个位置究竟是4还是7；读取这些值需要相应结果已经可用 |

看 `a = x + 1.0` 时，Python会执行右侧表达式，调用MLX的加法接口，并把返回的数组对象绑定给名字a。**Name binding（名称绑定）** 就是让变量名指向一个对象。这里调用、对象创建、shape/dtype推导、依赖记录等工作确实发生了；但返回一个数组对象，并不要求其所有元素的数值此时已经计算完成。

记录的依赖指向这次运算使用的数组对象，不是存下字符串 `x + 1.0`，以后再根据Python变量名查找输入。这个区别也解释了17.5中已建立的旧loss不会因为model后来换成新参数，就自动变为更新后的loss。

因此要分清：**Python已经处理完这次调用**，与**相应设备数值结果已经完成**。对交给GPU的数组工作，前者不能单独证明后者；有些数值工作可能尚未启动。也不能反过来把惰性求值说成“每一行都会立即异步启动一个GPU任务”。

惰性不表示整个程序什么都没做、也不表示零内存。例如从Python列表构造一个已有常量数组，需要处理这些已有数据。我们只把“记录运算关系”和“完成该运算的元素数值”区分开，不把所有MLX语句一概当作同一种行为。

固定版本依据：[v0.32.2 ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/ops.cpp)中的加法/乘法创建带有输入依赖的输出数组；[array.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.h)分别保存数组描述、依赖和结果可用状态。这里只用这些源码核对概念，不引入内部状态查询API。

### 2. 一个有两条分支的小例子：从矩阵到Scalar

本节的x是普通实数数组，shape为 `(2,3)`，意思是2行3列；这两个轴没有batch/token语义，x也不是MiniGPT的logits。s只是一个算术平均，不能把它叫作本节新增的语言模型loss。

先铺垫代码将用到的操作：

- `mx.array(..., dtype=mx.float32)`：从给定数据构造MLX数组，并明确使用32位浮点数。这里的嵌套Python列表每项是一行，不从外部取数据。
- **Elementwise operation（逐元素运算）**：对应位置分别运算。`x + 1.0` 给每个元素加1；`x * 2.0` 给每个元素乘2，`*`在这里不是矩阵乘法。
- **Scalar broadcasting（标量广播）**：把一个标量用于每个位置的运算。本例的shape仍为 `(2,3)`；这种数学语义不要求实际复制一整张常量矩阵。
- **Reduction（归约）**：把一组元素汇总为更少的结果。`mx.mean(c)` 在固定v0.32.2默认 `axis=None`、`keepdims=False`，对全部元素求平均且不保留被归约的维度，所以输出为scalar MLX array，shape `()`。

`None`在这个API中表示未指定某一条轴，因此使用全部轴；`False`表示不保留归约轴。本例用默认写法 `mx.mean(c)`，没有隐藏的按行平均或按列平均。

先给出明确的输入，再准备依赖关系：

```python
# EXPLICIT TOY CONSTANTS — prepared only, NOT executed.
x = mx.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    dtype=mx.float32,
)

# Shape (2, 3): two rows and three columns, with no token/batch meaning.
```

```python
# Build the relationships; do not display array values in this cell.
a = x + 1.0
b = x * 2.0
c = a + b
s = mx.mean(c)

# No bare array expression at the end: ordinary Notebook display can read values.
```

#### 手算的是数学结果，不是本轮的执行输出

用i表示行下标（0或1），j表示列下标（0、1或2）。例如 `x[i,j]` 指第i行第j列。上面的数学关系是：

$$
a_{ij}=x_{ij}+1,\qquad b_{ij}=2x_{ij},\qquad c_{ij}=a_{ij}+b_{ij}=3x_{ij}+1.
$$

双重求和表示把两行三列的六个位置全部相加：

$$
s=\frac{1}{6}\sum_{i=0}^{1}\sum_{j=0}^{2}c_{ij}.
$$

逐位置代入明确的x，可手算：

$$
a=\begin{bmatrix}2&3&4\\5&6&7\end{bmatrix},\qquad
b=\begin{bmatrix}2&4&6\\8&10&12\end{bmatrix},\qquad
c=\begin{bmatrix}4&7&10\\13&16&19\end{bmatrix}.
$$

$$
s=\frac{4+7+10+13+16+19}{6}=\frac{69}{6}=11.5.
$$

这些数值来自手工推导，不是运行输出，也不作为GPU完成工作的证据。代码中保留原来的两条分支，没有用手算的 `3*x+1` 偷换代码结构。

#### Compute Graph：箭头说明“依赖谁”

**Compute graph（计算图）** 是表达运算依赖关系的一种方式。图中的节点表示数组结果，箭头从输入指向依赖它的输出。本例x同时供给a与b，c需要两条分支的结果，s再汇总c：

```mermaid
flowchart TD
    X["x: (2,3)"] --> A["a = x + 1: (2,3)"]
    X --> B["b = x * 2: (2,3)"]
    A --> C["c = a + b: (2,3)"]
    B --> C
    C --> S["s = mean(c): ()"]
```

同一关系也可直接读表；即使Notebook前端没有渲染上图，表格仍完整表达依赖：

| 结果 | 直接依赖 | 运算 | 输出shape |
|---|---|---|---|
| x | 本节已给定的常量 | 构造输入数组 | `(2,3)` |
| a | x、标量1 | 逐元素加法 | `(2,3)` |
| b | x、标量2 | 逐元素乘法 | `(2,3)` |
| c | a、b | 逐元素相加 | `(2,3)` |
| s | c | 全元素平均 | `()` |

图省略了标量常量等实现细节。它是**数学依赖示意**，不是GPU执行时间线，也不保证a与b真的并行执行。一个高级API可以由多个内部运算组成；一行Python、图中的一个逻辑节点、一个底层GPU计算任务不能简单画等号。本节不分析具体设备调度。

### 3. 为什么还没读取元素，就能知道Shape和Dtype？

**Shape（形状）** 描述各轴长度；**dtype（数据类型）** 描述元素按哪一种数值类型表示。它们属于metadata。

本例的shape可以沿规则推出：x已知为 `(2,3)`；标量广播不改变a/b的shape；同shape的a与b相加得到同shape的c；对全部六个元素求平均并移除归约轴，s就没有剩余轴。这个推理不需要先读出c里面的4、7、10等结果。

Python中的 **attribute（属性）** 用点号读取：`c.shape`、`c.dtype`，这两处不加括号。固定MLX版本把shape暴露为Python tuple，把dtype暴露为数据类型描述；这两个读取本身不要求c的元素值完成求值。`print(...)` 是Python的显示函数；下面传给它的是标签字符串、shape tuple和dtype，而不是MLX数组本身。

本例x/a/b/c的shape均为 `(2,3)`，s为 `()`，这些数组的dtype为float32。**`()`表示零个轴的标量数组，仍有一个元素，不是空数组，也不是已经转换好的Python float。** 已经能读取这个shape，不能推出该标量的值已经完成计算。

固定依据：[v0.32.2 array.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.h)的shape/dtype访问器，以及[Python array binding](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/array.cpp)。

```python
# Metadata access only. This cell has not been executed.
print("x metadata:", x.shape, x.dtype)
print("a metadata:", a.shape, a.dtype)
print("b metadata:", b.shape, b.dtype)
print("c metadata:", c.shape, c.dtype)
print("s metadata:", s.shape, s.dtype)

# These attributes describe results; they are not evidence of numerical completion.
```

### 4. 普通Python函数已经返回，也不自动证明数值完成

这与17.5使用普通Python函数并不矛盾。先恢复三个Python含义：

- `def build_score(arr):` 定义一个函数；定义时不会执行函数体。`arr`是形参，调用时接收传入的数组对象。
- `build_score(x)` 才是调用，Python会执行函数体中的语句。名字 `build_score` 是本节自定义名称，不是一个额外的MLX API。
- `return score` 把score所指的对象交还给调用者。本例返回MLX数组对象，不在return处自动把它转成Python数字。

下面只是把同一个两分支表达式放进普通函数。`left`、`right`、`combined`、`score`是函数内部的变量名，分别对应之前的a、b、c、s；传入同一个x，数学关系和shape不变。函数返回后，调用者将该数组对象绑定给 `score_from_function`。

在这个只用数组运算并返回数组的函数中，**函数体执行结束**与**结果元素的数值完成**仍是两件事。普通Python的return不是额外的求值请求。本例没有任何编译包装，也没有训练步骤。

```python
def build_score(arr):
    left = arr + 1.0
    right = arr * 2.0
    combined = left + right
    score = mx.mean(combined)
    return score


# One prepared ordinary function call; NOT executed during material preparation.
score_from_function = build_score(x)
print("returned metadata:", score_from_function.shape, score_from_function.dtype)

# Returning an MLX array does not itself request its element value.
```

### 5. 看描述与读取值不同：接回17.5的求值边界

有些操作需要真正的元素值。**Implicit evaluation（隐式求值）** 指代码没有单独写 `mx.eval`，但某个取值操作为了给出结果而要求数组求值。这里仅辨认最常见的显示/取值方式，不增加求值或性能实验。

| 阅读到的表达式 | 要取得什么 | 与本例求值的关系 |
|---|---|---|
| `s.shape`、`s.dtype` | 结果的描述 | 这两个属性访问本身不要求元素值 |
| `print(s)` | 数组的数值表示 | 会要求s完成相应求值，才能显示值 |
| Notebook单元末尾的裸表达式 `s` | 常规交互显示中的对象表示 | 通常也会读取数组数值；不能把它视为纯metadata检查 |
| `s.item()` | 单个元素转为标准Python scalar | 需要时会求值；本例float32标量会转为Python float |
| `mx.eval(s)` | 明确要求s及所需依赖完成 | 沿用17.5接口，返回None，不返回一个新s |

`array.item()`是数组对象上的 **method（方法）**，所以有括号；固定版本要求数组只有一个元素。本例s满足条件。这个接口已经在17.5日志显示中出现，这里只是说明它为什么不能当作不影响求值的“查看一下”。

这也解释了前面代码为什么只打印 `.shape`、`.dtype`，并避免把MLX数组单独留在单元格最后一行。以后看到数组已经被打印，不能据此反推“之前每个加法语句都立即完成了数值计算”；打印本身可能正是要求数值的操作。

上述表格是接口阅读材料，不是本节额外的执行单元。固定版本中，数组的数值显示路径会调用求值，见[Python array binding](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/array.cpp)与[utils.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/utils.cpp)；`.item()`的取值路径见[convert.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/convert.cpp)。

#### 17.5的边界现在更容易理解

17.5的 `loss_and_grad_fn(model, X, Y)` 建立forward、scalar loss和梯度的关系；`optimizer.update(model, grads)`继续用梯度建立新参数与新状态的关系，并把对象中的引用更新好。仅从这两次Python调用已经返回，不能推出所有相应的设备数值已经完成。

所以17.5接着使用 `mx.eval(loss_before_update, model.parameters(), optimizer.state)`，明确要求这些结果及其依赖完成。这个接口等待指定结果可用后返回None；它不是对所有无关GPU工作的全局完成承诺。

那一节的训练函数之所以在返回前有明确的完成要求，是因为它**在函数体中写了求值边界**，不是因为普通Python函数的return天然具有这种保证。18.1只解释这个原因，不重建模型、优化器或训练单步，也不展开18.2的边界安排、图增长和内存主题。

### 本节的工作副本顺序、材料状态与停止点

以后使用本节新work副本时：

1. 在仓库根目录先 `git pull`，再将77母版用时间戳新文件名及 `cp -n` 安全复制到已有 `notebooks/work/`；只打开新副本。
2. 选择项目Python3.12 / `.venv` kernel，从本节 `import mlx.core as mx` 开始；其后输入常量和全部定义都已在副本中。
3. 按次序阅读数组对象与依赖、常量与两分支运算、手算及shape，再使用仅读取metadata的示例；不需要先运行17.5。
4. 普通函数示例依赖的只有本节mx和x。末尾表格用于辨认取值操作，17.5边界仅作阅读衔接，不需要在77副本里补入任何model或optimizer变量。

本节贯穿的关系是：**已知数学表达式和shape，足以继续组织依赖；需要元素值时，才必须保证相应数值工作完成。** Python名称绑定、函数返回、metadata可读与GPU结果完成之间，不能仅凭其中一项替另外一项作结论。

大Step17的全部材料验收保留；18.1现已获材料审核方独立核心静态PASS，实际修正 **0/2**。18.1验收后的累计实际修正：16.1=**1/2**、16.2=**1/2**（含已记录的超时后完整回复补齐），16.3至16.5、17.1至17.5及18.1各**0/2**。每小节最多两次实际修正，跨轮与换对话不重置。

只编写发布材料，没有执行Notebook、课程代码、测试、训练、实验、采样或用户work副本，没有读取/下载语料，没有代答练习；不声称学员掌握、执行完成或运行验证通过。不返工已合格课节。

**18.1当轮完成后停在18.1，没有进入18.2。** 后续仍仅按请求逐小节准备，至大Step20全部完成后停止。

### 参考与固定版本核对

- [17.5已验收母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/3fdec00b251765bda581c2b183e68d039c83c902/notebooks/76_step17_5_uncompiled_training_step.ipynb)：未编译单步、scalar loss与显式求值边界。
- [MLX v0.32.2 ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/ops.cpp)：add/multiply创建运算依赖，mean的组合实现。
- [MLX v0.32.2 array.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.h)：数组描述、shape/dtype读取，以及数值可用状态的区分。
- [MLX v0.32.2 Python ops](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/ops.cpp)与[Python array](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/array.cpp)：mean默认轴/keepdims、属性、数值显示绑定。
- [MLX v0.32.2 utils.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/utils.cpp)与[convert.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/convert.cpp)：数组显示与item读取触发相应求值。
- [MLX v0.32.2 Python transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp)与[core transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/transforms.cpp)：仅用于复核17.5已讲过的eval接口及等待语义。
- 官方API：[array.shape](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.shape.html)、[mean](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.mean.html)、[array.item](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.item.html)。在线页面可能更新，本课程接口与行为以 **v0.32.2源码** 为准。
- [Roadmap](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/course-roadmap.md)与[teaching handoff](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/teaching-handoff.md)：18.1当前材料状态、累计计数与停止边界。


## Step 18.2 — mx.eval as an execution/materialization boundary

母版：[78_step18_2_eval_boundary.ipynb](../notebooks/78_step18_2_eval_boundary.ipynb)。18.2实际六文件提交 `e3656db90239437cc0d50c24730920f1f783e59c`、固定MLX接口及完整教学回复已由材料审核方独立确认核心静态PASS，实际修正 **0/2**。下文保留已通过的教学及18.2当轮边界，不返工该节。它承接18.1惰性求值及17.5单步，解释选定数组与依赖完成、迭代状态的待求值链以及内存关系；全部示例仅为材料，未执行、无输出。

### 0. 新Work Copy：导入、数据和两个迭代片段都在本节

沿用项目 Python 3.12 / `.venv` kernel，以及固定的 `mlx==0.32.2`。只用 `mlx.core`；`import ... as mx`给数组模块一个短名称。

本节重新给出18.1的常量与分支表达式，并为两个三轮状态片段分别提供明确初值。所有代码只依赖本节导入和前面的本节定义，不需要旧kernel、`%run`旧Notebook、语料、sampler、MiniGPT类、`nn`或Adam对象。

两个迭代片段用于对照阅读“求值边界放在哪里”，不是计时、内存测量或训练实验。以后使用新副本时从本节开头按顺序取得定义即可；本轮不运行这些片段。

```python
import mlx.core as mx

# Self-contained import. Every code cell in this master is unexecuted.
```

### 1. Execution/Materialization Boundary：让选定结果的数值可用

**Execution boundary（执行边界）** 在本节指一个明确的要求：选定结果所需的数值工作，要在这个调用返回前完成。**Materialization（物化）** 指让数组结果具有可读取的元素数值；数组仍是MLX数组，shape和dtype也不因这个请求而改变。

18.1已经介绍 **evaluation（求值）**：沿依赖关系取得数组数值。固定接口是：

```python
mx.eval(*args) -> None
```

这一行是接口说明，不是待运行的Python语句。`*args`表示接收多个实参；`-> None`表示不返回一个结果数组。调用时可以给一个数组、多个数组，也可以给由dict/list/tuple组成的数组树。**Array tree（数组树）** 是用这些Python容器组织数组叶子的嵌套结构；例如17.5的模型参数树和优化器状态树。树中非数组叶子会被忽略。

`mx.eval`要求指定数组及取得它们所需的依赖完成，并等待这些指定结果可用后返回。这个等待针对请求结果，不是对一切无关GPU工作的全局完成承诺。它也不是把数组自动转成Python数字、复制到CPU、保存到磁盘，或保证给每个逻辑节点分配一块独占的新内存。

通常写成单独一行调用。不要写 `state = mx.eval(state)`：返回值是None，这样会把名字state改绑到None，丢掉通过这个名字访问原数组的途径。

已经完成的同一个数组再次被请求求值，并不代表又做了一次数学更新。更新由诸如 `state + 1.0` 的表达式定义；`mx.eval`负责完成已经建立的关系。

固定版本依据：[Python transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp)、[数组树处理](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/trees.cpp)与[核心eval实现](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/transforms.cpp)。

### 2. 选定输出决定所需依赖：恢复18.1的小分支

先恢复所用操作：`mx.array`从明确给出的列表构造数组，`dtype=mx.float32`指定32位浮点数。`+`和这里的 `*`是逐元素运算；标量1或2广播到各位置，shape不变。`mx.mean(c)`默认对全部元素求平均，且不保留归约轴，所以由 `(2,3)` 得到scalar shape `()`。

这里x只是2行3列的普通实数数组，不是token batch或logits，s也不是语言模型loss。所有数字直接写在本节：

```python
# Restore the complete branch example; prepared only, NOT executed.
x = mx.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    dtype=mx.float32,
)

a = x + 1.0
b = x * 2.0
c = a + b
s = mx.mean(c)

# Do not display array values here: display itself can request evaluation.
```

#### 先问“这一次选了谁”，再沿输入方向回溯

**Selected outputs（选定输出）** 是本次传给mx.eval的数组。一个输出的 **dependencies（依赖）** 是取得它所需要的输入及运算。

```mermaid
flowchart TD
    X["x: (2,3)"] --> A["a = x + 1: (2,3)"]
    X --> B["b = x * 2: (2,3)"]
    A --> C["c = a + b: (2,3)"]
    B --> C
    C --> S["s = mean(c): ()"]
```

箭头从输入指向依赖它的结果；求值请求从选定结果回看所需输入。这是逻辑依赖图，不是设备执行时间线。若前端不渲染图，可读下面的对应表：

| 请求 | 本例明确要求完成的结果关系 | 不能据此保证的结果 |
|---|---|---|
| `mx.eval(a)` | a及产生a所需的x/加法关系 | 另一条b分支，以及后继c、s |
| `mx.eval(s)` | s以及取得它所需的c、a、b、x关系 | 与这些结果无关的其他数组工作 |
| `mx.eval(x)` | 输入x本身；本例x来自已有常量 | 使用x的所有后继表达式不会因此自动获得完成保证 |

“不保证”不等于断言其他数组绝对没有执行过：它们可能在别处被求值，底层也可能有共享计算。接口保证的重点是所选结果及必要依赖，不是把整张有关联的图、或全局设备工作一概视为完成。

先准备一个只选a的边界。调用返回后a仍为shape `(2,3)` 的MLX数组；没有新数组返回给变量：

```python
# Request a and its required dependencies. Prepared only, NOT executed.
mx.eval(a)

# This request alone does not establish that b, c, or s have completed.
```

#### 多个实参与数组树：把需要完成的结果明确列出

**dict（字典）** 用键和值组织对象：下面的 `{"combined": c}` 表示键名combined对应数组c。这个键只是组织标签，没有特殊的求值含义。

接下来把s与这个小字典作为两个实参传入。同一请求既覆盖scalar s，也覆盖字典中的c；MLX会收集容器中的数组叶子。这与17.5一次列出loss、参数树和优化器状态树的接口形式一致。

本例c本来就是s所需的依赖，额外列出c不表示要再做一次加法；这里只用已经定义好的对象展示“数组 + 数组树”的接口。准备代码中 `selected_results` 的所有内容都来自本节：

```python
selected_results = {"combined": c}
mx.eval(s, selected_results)

# s is still an MLX scalar array with shape (); c still has shape (2, 3).
# The call returns None. No numerical output is recorded in this master.
```

手算可以帮助核对这份请求对应什么数值：

$$
c=(x+1)+2x=
\begin{bmatrix}4&7&10\\13&16&19\end{bmatrix},\qquad
s=\frac{4+7+10+13+16+19}{6}=11.5.
$$

这些是数学推导，不是本轮运行结果。`mx.eval`不改变数学表达式，不改变 `(2,3)` 与 `()` 的shape，也不会把scalar s自动变成Python float。

### 3. Iterative State：同一个变量名，为什么仍会留下更长的依赖链？

**State（状态）** 是一轮结束后留给下一轮继续使用的量。本节只用普通矩阵状态，不涉及训练参数更新。用大写 \(S_k\) 表示第k次更新后的状态：k=0是初值，k是更新次数，不是token位置或矩阵轴；它与前面的小写scalar变量s不同。

从同样的2行3列常量开始，每次给所有元素加1：

$$
S_0=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
S_{k+1}=S_k+1.
$$

这里的1是标量广播。先铺垫即将使用的Python循环：

- `for k in range(3):` 让缩进的循环体依次执行三次，k依次为0、1、2。
- `state_deferred = state_deferred + 1.0` 先处理右侧：取得一个依赖旧状态的新数组对象，再把左侧名字绑定到新对象。
- 这次赋值没有把旧数组改写成新值。新数组仍需要找到旧数组这个输入，这就是 **reference（引用）** 所保留的关系。

下面的版本在循环内部不要求数值。初始化与三次更新完整放在同一个单元，避免依赖旧工作副本中的state：

```python
# Version A: defer evaluation until after these three updates.
# Reading example only; this cell is NOT executed during preparation.
state_deferred = mx.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    dtype=mx.float32,
)

for k in range(3):
    state_deferred = state_deferred + 1.0

# No value printing, item(), or bare-array display in this cell.
```

#### Shape固定，不等于尚未完成的历史固定

用版本下标展开同名赋值，可以手算：

| 已定义的更新次数 | 当前状态的数学值 | Shape | 这段未求值递推包含的加1关系数 |
|---|---|---|---:|
| 0 | `[[1,2,3],[4,5,6]]` | `(2,3)` | 0 |
| 1 | `[[2,3,4],[5,6,7]]` | `(2,3)` | 1 |
| 2 | `[[3,4,5],[6,7,8]]` | `(2,3)` | 2 |
| 3 | `[[4,5,6],[7,8,9]]` | `(2,3)` | 3 |

最后一列统计的是代码表达的逻辑更新关系，不是内部primitive数、GPU任务数或内存测量。

数学依赖可以写成 \(S_0\to S_1\to S_2\to S_3\)，箭头仍表示后者需要前者。最后名字 `state_deferred` 虽然只指向 \(S_3\)，但在尚未求值时，\(S_3\) 需要 \(S_2\)，\(S_2\) 又需要 \(S_1\)，一直连接到初值。

因此，Python只反复使用一个变量名，并不会消除新数组所需的旧依赖。这不需要另建历史列表就会发生。若连续K次更新都没有显式或隐式求值，K表示这段连续更新的总次数，本例逻辑依赖链会继续延长，而每个状态的shape仍为 `(2,3)`。

这解释的是更多轮迭代时的趋势，不需要真的运行长循环来证明。本节只准备三轮小例子，没有大规模循环或内存压力任务。

#### 在链的末端请求当前状态

接下来准备 `mx.eval(state_deferred)`。它需要把取得当前 \(S_3\) 所必需、但尚未完成的依赖补齐，等待当前状态数值可用；它不是只算最后一个加法而跳过前两次必要关系。

这项调用没有再定义一次“加1”，因此当前数学结果仍是表中的 \(S_3\)，不是 \(S_4\)。同一已完成结果再次被求值，也不会凭空增加更新次数。

```python
# Finish the currently selected state and its necessary pending dependencies.
mx.eval(state_deferred)

# No assignment from mx.eval: state_deferred remains the current MLX array.
```

### 4. 在每轮末尾完成状态：下一轮从已有结果继续

下面给出同一数学递推的另一个边界安排。它再次从自己的明确初值开始，每轮先定义新的state，再要求这个state完成。`mx.eval`缩进在for循环体内，所以三轮都会遇到该边界。

两个版本都描述 \(S_{k+1}=S_k+1\)，数学末值和shape相同。区别是：版本A在三轮关系都建立后才请求当前结果；版本B在下一轮开始前，已经要求上一轮当前状态完成。

这是边界位置的阅读对照，不是运行比较，也不据此宣称哪种写法更快。每轮只有一次加法是为了看清状态衔接，不构成“任何程序每个小运算后都应求值”的调优结论。

```python
# Version B: make the current state available at each iteration boundary.
# It has its own initial value; it does not continue from state_deferred.
state_bounded = mx.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    dtype=mx.float32,
)

for k in range(3):
    state_bounded = state_bounded + 1.0
    mx.eval(state_bounded)

# Hand-derived final state: [[4, 5, 6], [7, 8, 9]], shape (2, 3).
# No execution, timing, memory measurement, or equality test is performed.
```

#### 当前结果保留，旧的生成依赖可以解除

本例使用普通数组，没有放入 `nn.value_and_grad` 等求导函数变换中。固定v0.32.2的普通数组求值路径会解除不再需要的生成运算/输入关系，同时保留当前数组数据；如果旧对象没有其他依赖或外部引用，就不必再因为当前状态的这条生成链而一直保留。

因此，完成 \(S_1\) 后可以用它的已有数值建立 \(S_2\)；完成 \(S_2\) 后再继续 \(S_3\)。**边界让当前数值成为下一轮的起点，避免当前状态持续挂着跨越越来越多轮的未完成生成历史。** 它没有把状态重置成 \(S_0\)，也没有改变加1的数学含义。

这个依赖释放说明限定在本节的普通数组迭代。函数变换为求导等目的保留依赖时有自己的规则；不能把这里的描述当作任意改动可微loss内部计算关系的规则。17.5已有“求导调用 → optimizer update → 显式求值”的顺序保持不变，本节不改写它。

固定依据：[核心求值路径](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/transforms.cpp)与[array.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.cpp)中的依赖解除实现。这里只说明源码支持的关系，不要求使用内部方法。

### 5. 为什么关系到Memory？先分清依赖记录与数值数据

**Memory（内存）** 在这里不只是一个矩阵的元素数据。至少要分清：

| 部分 | 与本例的关系 |
|---|---|
| 运算描述和输入引用 | 尚未求值的链需要记录“如何从旧状态取得新状态”；更多未完成更新会带来更多这些记录 |
| 已有数组数据和被保留的对象 | 初始数据、当前已完成结果，以及仍被其他变量/容器引用的旧数组，都可能继续存在 |
| 求值过程中的结果与临时数据 | 真正完成依赖时还需要相应数值存储；具体复用和生存期由实现及其他引用决定 |

所以“惰性”不等于不占内存。另一方面，链上有K次更新，也不能直接说已经分配K份完整结果矩阵：依赖描述存在，不等于每个尚未求值结果都已经拥有独立的完整数据缓冲区。

同样，单看一个state的 `(2,3)` shape，无法知道程序保留了多少依赖描述、其他数组或临时数据。把“单个state的逻辑数据大小 × K”写成此时的真实总内存或峰值，也没有依据。

为跨轮状态建立求值边界，有助于让不再需要的生成依赖退出当前状态的待完成历史。但它不是“清空全部内存”的操作：如果另一个Python列表保留了各轮旧数组，求值不会替你删除这些引用；分配器也可能保留可复用的缓存。因此不能承诺总内存恒定，或操作系统显示的占用立刻下降。

**Buffer（缓冲区）** 是用于存放数值数据的内存区域；**allocator（分配器）** 管理这些区域的取得和复用。这里提到缓存只是说明“依赖可以释放”不等于“所有内存立即归还”。本节不加入内存统计或缓存清理API。

固定依据：[array.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.h)分别保存数组描述、输入引用与共享数据；[Metal allocator](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp)包含缓冲区缓存复用逻辑。以上为源码解释，没有内存测量结果。

### 接回17.5：边界要覆盖真正带到下一步的状态

17.5已有这一行：`mx.eval(loss_before_update, model.parameters(), optimizer.state)`。它只作为本节的阅读衔接，不放进78的新执行单元；因此新副本不用补入model或optimizer定义。

| 被选定的对象 | 它在17.5中代表什么 | 为什么列入边界 |
|---|---|---|
| `loss_before_update` | 本步更新前forward得到的scalar loss | 要取得这次返回/日志所用的数值 |
| `model.parameters()` | 更新后模型当前持有的参数树 | 下一步forward要继续使用的模型状态 |
| `optimizer.state` | 更新后的m、v、step等数组状态 | 下一步Adam继续使用的历史状态 |

这与刚才的矩阵state有共同点：下一轮要接着使用的状态，不应被忽略在完成要求之外。只让loss求值，不能据此认定新的参数与优化器状态也已经完成；它们可能是loss之后才建立的更新关系，就像求值a不自动保证后继c、s完成。

17.5把这三类结果一起列出，明确了单步返回前要完成的范围。这里不重复Adam、逆向自动微分或训练循环，也不把“跨轮未完成依赖不持续延长”说成整个训练进程内存恒定的证明。

18.1的metadata与取值区别仍适用：`.shape`和`.dtype`本身不请求元素数值；打印数组、Notebook常规裸数组显示或 `.item()`可能要求相应结果求值。所以本节的未求值链示例中没有插入这些值显示操作，避免阅读时把隐式边界漏算进去。

### 工作副本顺序、材料状态与停止点

以后使用本节工作副本时：

1. 在仓库根目录先 `git pull`，再用时间戳新文件名及 `cp -n` 把78母版安全复制到已有 `notebooks/work/`；只打开新副本。
2. 选择项目Python3.12 / `.venv` kernel，从本节mx导入开始。先取得分支示例全部常量与定义，再阅读两个选定结果的求值调用。
3. 状态版本A单元自带初值；后一个独立求值单元只完成该版本当前结果，不额外定义一次更新。
4. 状态版本B也自带初值，不承接版本A的末状态。两段循环只有三轮，均用于阅读边界语义；不要求任何测试、长循环、计时或内存测量。
5. 17.5的model/optimizer表达式只在说明文字中出现，新副本没有对旧kernel变量的执行依赖。

18.2当轮只进行初始材料编写发布，随后已获独立材料核心静态PASS，实际修正保持 **0/2**。18.1的验收同样保持 **0/2**。当轮记录的累计实际修正：16.1=**1/2**、16.2=**1/2**（含已记录的超时后完整回复补齐），16.3至16.5、17.1至17.5及18.1各**0/2**，18.2初始生成=**0/2**。每节最多两次实际修正，跨轮与换对话不重置。

本轮只编写发布材料，未执行Notebook、课程代码、代码测试、训练、实验、采样或用户work副本，未读取/下载语料，未代答练习；不声称学员掌握或运行通过，不返工合格课节。

**完成即停在18.2，不进入18.3。** 后续仅按请求逐小节准备，至大Step20全部完成后停止。

### 固定版本依据与课程链接

- [18.1已验收母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/4dd066c839d8e5d371b6ba1fb9d039d50fd39efa/notebooks/77_step18_1_lazy_evaluation.ipynb)：惰性数组、依赖、metadata与取值区别。
- [17.5已验收母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/4dd066c839d8e5d371b6ba1fb9d039d50fd39efa/notebooks/76_step17_5_uncompiled_training_step.ipynb)：单步选择loss、新参数和优化器状态完成。
- [MLX v0.32.2 Python transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp)与[trees.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/trees.cpp)：eval多实参、数组树、非数组叶子与None返回。
- [MLX v0.32.2 core transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/transforms.cpp)：从选定结果沿输入依赖回溯、等待所选结果，以及普通数组的依赖解除路径。
- [MLX v0.32.2 array.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.h)与[array.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.cpp)：数组描述、输入引用、共享数据与普通求值后的数据保留。
- [MLX v0.32.2 Metal allocator](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp)：缓冲区复用与缓存；本节没有调用测量或缓存操作API。
- [官方mx.eval接口](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html)。在线文档可能更新，本课程行为以 **v0.32.2源码** 为准。
- [Roadmap](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/course-roadmap.md)与[teaching handoff](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/teaching-handoff.md)：18.2当前材料状态、累计计数及停止边界。

## Step 18.3 — mx.compile

母版：[79_step18_3_mx_compile.ipynb](../notebooks/79_step18_3_mx_compile.ipynb)。本节承接18.1／18.2，准备tracing与编译、优化和fusion直觉、首次使用成本及shape／dtype相关缓存。初始实际修正 **0/2**；全部单元未执行、无输出，完成即停在18.3。

### 0. 新Work Copy：导入、函数和输入全部在本节

沿用项目Python 3.12 / `.venv` kernel与 `pyproject.toml` 固定的 `mlx==0.32.2`。`import mlx.core as mx`导入数组模块，`as mx`提供短名称；这是本节唯一导入。

新副本完整恢复18.1的普通函数 `build_score`，重新给出全部手写输入，并在本节创建编译后的可调用对象。以后使用新副本时，从导入单元开始按顺序取得定义即可；无需旧kernel、运行旧Notebook、`%run`、语料、sampler、模型或优化器。

本节按编译未被显式禁用、后端支持编译的正常课程环境解释机制，不添加环境切换API。我们只写小型调用示例，不运行它们，也不安排计时、缓存命中观测或性能实验。

```python
import mlx.core as mx

# Self-contained import. Every code cell in this master is unexecuted.
```

### 1. 恢复一个熟悉的普通函数：数学含义保持清楚

先恢复必要的Python和数组接口：

- `def build_score(arr):`创建函数；定义时不执行函数体。`arr`是形参，调用时接收输入数组。
- 函数内的名字 `left`、`right`、`combined`、`score`只组织本次计算。`return score`返回MLX数组对象，不自动把它变成Python数字。
- `mx.array(..., dtype=mx.float32)`从明确写出的嵌套列表建立32位浮点数组；每个内层列表是一行。
- `+`与这里的 `*`是 **elementwise operations（逐元素运算）**。标量1.0和2.0广播到每个位置，shape不变；`*`不是矩阵乘法。
- `mx.mean`执行 **reduction（归约）**。固定版本默认 `axis=None, keepdims=False`，即平均全部元素且不保留归约轴，结果shape为 `()`。

本例x是普通实数矩阵，两个轴没有batch/token含义；score只是算术平均，不是新增的语言模型loss。函数仅使用显式传入的数组和固定数字，不修改外部对象；这样的 **pure function（纯函数）** 让我们先聚焦编译本身。

```python
# EXPLICIT TOY DATA — prepared only, NOT executed.
x = mx.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    dtype=mx.float32,
)


def build_score(arr):
    left = arr + 1.0
    right = arr * 2.0
    combined = left + right
    score = mx.mean(combined)
    return score


# Full local definition. No model, optimizer, hidden input, or old kernel state.
```

#### 从shape到手算：后面编译的是这个函数

用i表示行下标，j表示列下标。对于本例，i取0、1，j取0、1、2；下标只是指定矩阵位置。数学关系为：

$$
c_{ij}=(x_{ij}+1)+2x_{ij}=3x_{ij}+1,
\qquad
s=\frac{1}{6}\sum_{i=0}^{1}\sum_{j=0}^{2}c_{ij}.
$$

双重求和把六个位置全部加起来，再除以6。因此：

$$
c=\begin{bmatrix}4&7&10\\13&16&19\end{bmatrix},
\qquad s=\frac{4+7+10+13+16+19}{6}=11.5.
$$

| 对象 | 含义 | 本例shape |
|---|---|---|
| x／函数内arr | 本次输入矩阵 | `(2,3)` |
| left、right、combined | 对应位置的加法、乘法与合并结果 | `(2,3)` |
| score | 全部六个元素的平均值 | `()` |

这些数组在本例中使用float32。shape `()`表示零个轴、一个元素的MLX scalar array，不是空数组，也不是普通Python float。

**11.5是手算的数学预期，没有来自Notebook的执行输出。** 编译应保持同一计算的数学含义；浮点数只有有限精度，不能据此保证任意编译前后结果都逐bit相同。`3x+1`也只是便于手算的表达式，不能当作已经观察到的编译器改写。

### 2. Function Transformation：先得到一个可调用对象

17.3已经见过“函数作为输入，得到另一个函数”的 **function transformation（函数变换）**。这里 `mx.compile`接收原函数，返回 **callable（可调用对象）**：以后可以用括号和实参调用它。

先分清三个表达式：

| 表达式 | Python含义 |
|---|---|
| `build_score` | 取得函数对象，没有调用 |
| `build_score(x)` | 立即调用原函数，得到它返回的数组对象 |
| `mx.compile(build_score)` | 把函数对象交给编译变换，取得编译包装后的可调用对象 |

所以这里传的是不带括号的 `build_score`，而不是把 `build_score(x)`先算出的数组交给compile。下面用普通赋值保留原函数与编译对象两个名字，不需要引入装饰器语法。

固定v0.32.2的接口可简记为以下签名；它是阅读说明，不是执行单元：

```text
mx.compile(fun, inputs=None, outputs=None, shapeless=False) -> callable
```

`fun`是待变换的函数。`shapeless`是一个布尔选项，`False`表示本节保持默认的具体shape匹配；稍后解释缓存。显式写 `shapeless=False`是把默认值写清楚。`inputs`、`outputs`是额外的状态捕获选项，默认均为None，本节不使用；它们不是在这里填写x的位置。实际数组仍通过后面的 `compiled_score(x)`传入，状态捕获留在18.4。

依据：[MLX v0.32.2 Python compile绑定](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp)。

```python
# Create a callable. Pass the function object, not build_score(x).
compiled_score = mx.compile(build_score, shapeless=False)

# shapeless=False is the MLX v0.32.2 default, written explicitly here.
# This line does not establish that any input's numerical result is complete.
```

### 3. Tracing记录数组关系，求值仍有自己的边界

**Tracing（追踪）** 在这里指：为输入创建占位数组，沿原Python函数走一遍，让数组运算建立计算依赖。**Placeholder／tracer（占位数组／追踪数组）** 具有相应的shape和dtype，但不是这次输入元素值的计算结果。

例如，首次遇到本节的 `(2,3), float32` 输入规格，若还没有可用的匹配缓存，MLX会用同shape、同dtype的占位输入走过 `build_score`：先记录两个逐元素分支，再记录合并和mean。随后整理与优化图，把当前真实输入接入准备好的计算关系，返回本次调用的结果数组。

这不是读取Python源码字符串，也不是把整个Python程序的所有行为都搬到GPU。原函数体在tracing时确实会被Python执行；以后命中可复用图时，不需要再次按原函数体重做同样的tracing。

本节让函数体只建立数组关系并返回数组。不要在里面对tracer或由它产生的数组使用 `mx.eval`、`print(array)`、`.item()`等需要元素值的操作；占位追踪时没有这些输入数值可供读取。后面的显式求值放在编译调用外侧。

**Compilation（编译）** 准备和优化可复用的计算表示；**evaluation（求值）** 要求某次输入对应的结果数值完成。两者承担不同工作。下面先准备一次调用，并只保存它返回的数组对象：

```python
# First prepared call for this input specification; NOT executed here.
score = compiled_score(x)

# score is an MLX scalar array, shape (). No value display in this cell.
```

#### 三个阶段不要压成同一件事

| 代码位置 | 发生的接口工作 | 是否由此保证score数值完成 |
|---|---|---|
| `compiled_score = mx.compile(build_score, shapeless=False)` | 创建可调用的编译包装 | 否；还没有这次输入的调用 |
| `score = compiled_score(x)` | 对当前规格查缓存；必要时tracing和图优化，接入本次x，返回结果数组 | 否；返回数组对象不是求值完成承诺 |
| `mx.eval(score)` | 要求score及取得它所需的依赖完成，并等待所选结果可用 | 是，针对这次选定的结果 |

18.2的接口语义保持不变：`mx.eval`返回None，score仍是MLX数组；不要写 `score = mx.eval(score)`。请求范围是选定结果及必要依赖，不是全局无关GPU工作的完成保证。

固定源码在首次图准备后将占位输入替换为“可供求值”的真实数组，并返回它们。它没有把编译调用本身变成 `mx.eval`。依据：[v0.32.2 compile_trace、compile与compile_replace](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)。

```python
# Explicit completion request outside the compiled function.
mx.eval(score)

# The return value is None; score remains the selected MLX array.
# No timing, result assertion, or recorded numerical output.
```

### 4. Optimization与Fusion：把合适的工作组织在一起

**Graph optimization（图优化）** 是在保持计算含义的前提下整理计算表示，例如合并可判定的重复工作，或去掉特定的无效操作。并不是每张图都恰好有这些机会；本例也没有承诺发生哪一种具体改写。

**Kernel（计算内核）** 是设备执行的一段计算程序。**Fusion（融合）** 将符合条件的若干运算合并为一个执行区域，可能减少多个小操作分别调度的开销，以及部分中间结果的单独读写。

回看同一个位置：

$$
\ell_{ij}=x_{ij}+1,\qquad r_{ij}=2x_{ij},\qquad c_{ij}=\ell_{ij}+r_{ij}.
$$

这里的希腊字母ell（\(\ell\)）只代表left分支，r代表right分支。三个运算都可以在各位置分别完成。若实现将合适的部分融合，设备可以把相关算术组织在一起，不必把每条Python赋值都当作一次独立设备任务，也可能少保留或少搬运某些中间结果。

但 `mx.mean(combined)`需要把多个位置汇总，它是归约，不能只凭前面几项逐元素计算可融合，就宣称整个函数一定成为一个kernel。固定版本的普通逐元素融合候选包括Add、Multiply等；Reduce不在同一候选集合里。具体优化还受图结构、运算类型、形状和后端条件约束。

| 可以从机制得到的理解 | 本节没有得出的运行结论 |
|---|---|
| 编译可识别和组合符合条件的数组工作 | 本函数已经实际融合成了几个kernel |
| 融合有机会减少调度或部分中间读写 | 本函数必然更快、必然减少多少内存 |
| 数学上可用 `3x+1`理解combined | 编译器一定生成了这个手工化简式 |

“几行Python”“几个逻辑运算”和“几个设备kernel”不是同一个计数。我们只讲优化的动机与条件，不使用性能、内存或kernel探针。

依据：[v0.32.2 compile_simplify、is_fusable与compile_fuse](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)。

### 5. 首次使用与Cache：准备计算方案也有成本

**Cache（缓存）** 在这里保存可复用的编译计算表示。第一次没有匹配项时，需要额外准备；匹配项仍可用时，后续调用可以复用它。

把 **first-call cost（首次调用成本）** 放回MLX的惰性执行背景，需要区分两层：

1. 首次遇到某种输入规格且没有匹配图缓存时，调用阶段需要tracing、整理与优化图等工作。
2. 第一次实际要求结果求值时，设备后端还可能需要生成／编译相应代码，并准备计算管线。**Compute pipeline（计算管线）** 在这里只指设备执行该kernel所需的已准备配置，不是数据输入pipeline。

因此，“首次使用并取得数值”的工作可能分布在编译调用与随后求值两个阶段。固定Metal实现把部分kernel代码／管线准备放在求值路径；不能把所有首次成本都归到 `mx.compile(build_score)`那一行，也不能说 `compiled_score(x)`返回前所有GPU编译和数值计算必已完成。

复用能省去一些准备工作，但本次输入的数值仍要计算，调用和执行也不是零成本。图规模、运算和环境都会影响实际收益；本节没有计时，不能断言首次慢多少、后续快多少或必然提速。

依据：[图缓存与首次图准备](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)、[Metal Compiled::eval_gpu](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/compiled.cpp)与[设备library／kernel准备及缓存](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp)。

#### 5.1 Shape／dtype怎样参与默认缓存匹配？

先固定讨论前提：保持同一个编译函数、相同的参数组织与环境设置，相关缓存仍存在，使用本节默认 `shapeless=False`。这里的 **input specification（输入规格）** 包括输入数组shape和dtype等信息；**specialization（专门化版本）** 指为某类输入规格准备的计算表示。

固定源码会比较输入数组数量、每个输入的轴数、shape和dtype等。**ndim（轴数）** 是shape中轴的数量，例如 `(2,3)`和 `(3,2)`都是两个轴。函数身份、非数组常量／参数结构、默认设备与执行环境等也参与匹配；shape和dtype不是完整缓存规则的全部。

**作为函数实参的MLX数组，其元素数值本身不加入这一图缓存匹配条件。** 于是同规格的新数据可以沿用计算方案，但仍需把新数组接进去计算新的答案。这个结论限定于数组元素值，不能直接套到函数实参中的普通Python整数或浮点常量上。

下面给出三组新增常量，分别改变数值、shape与dtype。`mx.float16`表示另一种16位浮点格式，精度与float32不同；本节只用它说明dtype区别，不作精度实验或混合精度教学。仍用 `mx.array`显式写出数据，无随机数、无语料，也不需要新的转换API。

```python
# Same shape and dtype as x; different element values.
x_new_values = mx.array(
    [[2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0]],
    dtype=mx.float32,
)

# Still six elements and two axes, but a different shape.
x_new_shape = mx.array(
    [[1.0, 2.0],
     [3.0, 4.0],
     [5.0, 6.0]],
    dtype=mx.float32,
)

# Same shape and real-number entries as x, but a different input dtype.
x_new_dtype = mx.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    dtype=mx.float16,
)
```

#### 5.2 缓存的是方案，返回值仍对应本次输入

下表是根据默认接口与源码推导的阅读预期，不是实测的缓存命中记录：

| 输入 | 输入shape／dtype | 对已准备的 `(2,3), float32` 图缓存 | 输出shape／实数手算 |
|---|---|---|---|
| 初始x | `(2,3)`／float32 | 首次没有匹配项时准备该规格 | `()`／11.5 |
| x_new_values | `(2,3)`／float32 | 可以匹配同一规格项；使用本次新数值 | `()`／14.5 |
| x_new_shape | `(3,2)`／float32 | 不能直接匹配原shape项；寻找另一项，没有时重新准备 | `()`／11.5 |
| x_new_dtype | `(2,3)`／float16 | 不能直接匹配原dtype项；寻找另一项，没有时重新准备 | `()`／11.5的实数数学值 |

对第二行，输入每个位置加1，所以 `3x+1`的每个位置增加3，平均值由11.5变为14.5。也可直接手算combined为 `[[7,10,13],[16,19,22]]`，六项和87，平均为14.5。这说明可复用的是处理输入的关系，不是永远返回第一次的11.5。

第三行虽仍有六个元素、输出也仍是scalar，输入shape却不同。缓存匹配不能只看元素总数，更不能只看最后输出shape。第四行说明dtype也是输入规格的一部分；手算表表达实数数学，不保证不同浮点格式的运行结果逐bit一致。

如果后来重新传入已经缓存过的 `(2,3), float32`，原规格的匹配项仍在时可以再次使用。不能把“从A换到B再换回A”说成每次都从零完整编译。同样，图缓存未命中不等于所有后端缓存也全部失效。

下面只准备三个明确的调用，再在外侧列出所需结果的求值边界。它们是接口阅读代码，不包含断言、计时、循环压测或缓存计数：

```python
# Reuse the same callable created above; these calls are prepared, not run.
score_new_values = compiled_score(x_new_values)
score_new_shape = compiled_score(x_new_shape)
score_new_dtype = compiled_score(x_new_dtype)

# Numerical completion remains a separate, explicit request.
mx.eval(score_new_values, score_new_shape, score_new_dtype)

# All three results have scalar shape (). No measured output or speedup claim.
```

#### 5.3 默认False的边界，以及函数对象的生存期

`shapeless=True`是非默认选项。在固定v0.32.2的缓存匹配中，它放宽相同ndim下具体维度大小的比较；**ndim和dtype仍会参与匹配**，并不是“任意形状、任意dtype只编译一次”。也不是所有函数都支持它。本节保持False，不增加True的示例或实验。

版本核对以 [v0.32.2 Python绑定](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp)和[CompileCache::find](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)为准，不把其他版本或其他实现的默认值套过来。

本节在调用前创建一次 `compiled_score`，后面复用同一对象。重新执行函数定义会创建新的Python函数对象；不能把新对象的首次调用当成前一个对象缓存的必然延续。这里说明代码组织与缓存讨论的前提，不要求做重新定义或清缓存实验。

18.2的执行边界仍然有效：编译图可以复用，当前输入对应的结果仍须在需要数值时完成求值。17.5已经写好的未编译训练单步保持原样；本节不把model、参数或optimizer状态塞进这个纯函数示例，也不改写训练步骤。

### 工作副本顺序、材料状态与停止点

以后使用本节新副本时：

1. 在仓库根目录先 `git pull`，再用时间戳新文件名与 `cp -n`把79母版安全复制到已有 `notebooks/work/`；只打开新副本。
2. 选择项目Python3.12 / `.venv` kernel，从唯一的mx导入开始，取得本节x与完整 `build_score`定义。
3. 阅读函数对象与compile接口后取得 `compiled_score`。本节先给调用，再单列外侧 `mx.eval`，保持包装、调用、求值三层关系清楚。
4. 三组后续输入都在本节直接给出；后续调用复用前面同一个编译对象，不依赖任何旧Notebook变量。

以上是副本依赖恢复顺序，不是本轮运行记录或执行要求。全部代码单元保持未执行与空输出，没有运行Notebook、代码测试、训练、实验、采样或用户work副本，没有读取／下载语料，也没有代答练习。不声称学员掌握、运行通过或测得性能。

18.2的独立材料验收保持 **0/2**，18.3当前仅为初始材料编写发布。累计实际修正：16.1=**1/2**、16.2=**1/2**（含已记录的超时后完整回复补齐），16.3至16.5、17.1至17.5、18.1及18.2各**0/2**，18.3初始生成=**0/2**。每节最多两次实际修正，跨轮和换对话不重置。

**完成即停在18.3，不进入18.4。** 后续仅按请求逐小节准备，至大Step20全部完成后停止。

### 固定版本依据与课程链接

- [18.1已验收母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/e3656db90239437cc0d50c24730920f1f783e59c/notebooks/77_step18_1_lazy_evaluation.ipynb)：本节复用的普通函数、两分支数学、shape与惰性求值。
- [18.2已验收母版](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/e3656db90239437cc0d50c24730920f1f783e59c/notebooks/78_step18_2_eval_boundary.ipynb)：选定数组与依赖的显式完成边界。
- [MLX v0.32.2 Python transforms](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp)：compile签名与默认值、函数包装、数组实参与Python常量／结构处理。
- [MLX v0.32.2 compile.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)：占位追踪、图缓存匹配、优化与融合候选、把真实输入接入图。
- [MLX v0.32.2 Metal compiled.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/compiled.cpp)与[device.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp)：求值时的后端代码与计算管线准备，以及相应缓存。
- [MLX v0.32.2 ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/ops.cpp)：原小函数中的逐元素操作与mean语义。
- [官方mx.compile接口](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html)。在线文档可能更新，本课程行为以 **v0.32.2真实源码** 为准。
- [Roadmap](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/course-roadmap.md)与[teaching handoff](https://github.com/kris0516/mlx-minigpt-from-scratch/blob/main/docs/teaching-handoff.md)：当前材料状态、累计修正与18.3停止边界。
