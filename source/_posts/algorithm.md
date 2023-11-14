---
title: 算法
date: 2023/2/13 20:46:26
categories:
  - [Algorithm]
---

.

<!-- more -->

# 二叉树

## 完全二叉树

1. 树的深度=一直遍历最左节点的长度
2. 左子树深度==右子树深度，左子树是全满的完全二叉树，如果左子树深度大于右子树深度，右子树是全满的完全二叉树



## 平衡二叉树

所有节点左右子树高度不大于1



# 字典序

就是按照字典排列顺序，英文字母按下面方式排列：

```
ABCDEFG HIJKLMN OPQRST UVWXYZ
abcdefg hijklmn opqrst uvwxyz
```



# 回溯

## 显回溯

pre是公共变量，遍历完这种情况要记得pop（），加入结果时记得copy（）

## 隐回溯

pre是函数内传递的变量，直接传就ok

## 去重

### 去res中重复元素

1. 排序数组

   可以用

   ```
   num[i]=num[i-1]跳过同层重复元素的选取（注意：子层不跳过）
   ```

2. set去重

### 排序问题去重

使用used数组

```
used=[True, ..., False, False]
```

# 位运算

## 基础操作

### 取反

异或操作，要取反的区域为1，不取反区域为0

# 优先队列

1. 使用heapq实现

   [8.4. heapq — Heap queue algorithm — Python 2.7.18 documentation](https://docs.python.org/2/library/heapq.html#basic-examples)

​	只能实现最小堆，通过再所有元素前加-号这个trick可实现最大堆

```
pq=[]
heapq.heappush(pq,a)
heapq.heappop(pq)
```

​	优先队列的元素可以是tuple

```
>>> h = []
>>> heappush(h, (5, 'write code'))
>>> heappush(h, (7, 'release product'))
>>> heappush(h, (1, 'write spec'))
>>> heappush(h, (3, 'create tests'))
>>> heappop(h)
(1, 'write spec')
```

