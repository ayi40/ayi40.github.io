---
title: Relation-enhance Rec
date: 2023/6/17 20:46:25
categories:
  - [RecSys, KGRec]
---

relation-enhance KG

<!-- more -->

# RE-KGR

Paper: RE-KGR: Relation-Enhanced Knowledge Graph Reasoning for Recommendation

总结：把relation当做向量空间，同时考虑relation的方向性，最后基于路径概率预测

## Methodology

![image-20230617151926949](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617151926949.png)

given a CKG

### Embedding Layer

for every entity and relation, one-hot to dense vector

### RGC Layer

**First-order Aggregation**:

project each entity t to a different semantic space conditioned
to the relation r:

![image-20230617153022663](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153022663.png)

![image-20230617153033948](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153033948.png)

![image-20230617153045142](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153045142.png)

![image-20230617153143790](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153143790.png)

Here, $M_{r−1}$, $M_r$ are mapping matrices, and r and r−1 are a pair of inverse relations,such as AuthorOf and WrittenBy.

**High-order Aggregation**:

![image-20230617153235776](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153235776.png)

![image-20230617153301643](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153301643.png)

Here,  is the concatenation operator, and e(0) denotes initial embeddings.(dense connectivity)

### Local Similarity Layer

![image-20230617153403760](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153403760.png)

### Prediction Layer

Predict Based on path:

use $P_{UIIP}={(h,r,t)|(h,r,t)\in G}$ to describe an acyclic UIIP, the probability of the UIIP is:

![image-20230617153641218](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153641218.png)

we use Pui to denote all acyclic UIIPs that start and end with user u and item i.

![image-20230617153723353](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617153723353.png)

# PeRN

![image-20230627183244905](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230627183244905.png)