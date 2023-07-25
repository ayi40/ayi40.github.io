---
title: Python Basic
date: 2023/7/11 20:46:26
categories:
  - [Algorithm]
---

.

<!-- more -->
# sort
1. 怎么让数组在第一维度正序，第二维度倒序

  ```python
  li.sort(key = lambda x:(x[0],-x[1]),reverse=True)
  ```

  