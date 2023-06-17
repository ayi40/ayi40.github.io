---
title: KR-GCN
date: 2023/4/26 20:46:26
categories:
  - [RecSys,KGRec]
---

.

<!-- more -->

# Background

previous study:

1. error propagation: consider all paths between every user-item pair might involve irrelevant one
2. weak explainability



# Model

![image-20230426111943016](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230426111943016.png)

4 parts:

1. the Graph encoding module: learn the representations of nodes in the heterogeneous graph. 
2. the Path Extraction and Selection module: extract paths between users and items from the heterogeneous graph and select higher-quality
   reasoning paths
3. the Path Encoding module: learn the representations of the selection reasoning paths.
4. the Preference Prediction module: predicts users’ preferences according to the reasoning paths.

## Graph Encoding - GCN

1. propagation and aggregation

initialized randomly

weighted sum aggregator :  the neighborhood nodes are aggregated via mean function.

![image-20230426113233853](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230426113233853.png)

2. weight sum to merge every layers

![image-20230426114004010](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230426114004010.png)

## Path Extraction

 we prune irrelevant paths between each user-item pair. 

 we extract multi-hop paths with the limitation that hops in every single path are less than l.