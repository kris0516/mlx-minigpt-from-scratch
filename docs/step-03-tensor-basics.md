# Step 03 — Tensor Basics

## Step 03.1 — Scalar, Vector, Matrix, Tensor, Shape

这一小步只建立最基本的张量直觉。

### Scalar

单个数字。

```text
3.0
shape = ()
```

### Vector

一排数字。

```text
[1, 2, 3]
shape = (3,)
```

### Matrix

二维数字表。

```text
[[1, 2, 3],
 [4, 5, 6]]

shape = (2, 3)
```

### Tensor

深度学习中，tensor 可以泛指任意维度数组。

例如 3D tensor：

```text
shape = (2, 2, 3)
```

可以先理解成：

```text
2 个
  2 行
    3 列
```

本步暂时不学习 axis、reshape、transpose 或 broadcast。
