---
title: DHT-Relation Work
date: 2023/5/26 20:46:25
categories:
  - [GNN,EdgeLearning]
---

Relation work of DHT

<!-- more -->

# Edgeformers

EDGEFORMERS: G RAPH-E MPOWERED TRANSFORMERS FOR REPRESENTATION L EARNING ON T EXTUALE DGE NETWORKS

## Background

1. Edge-aware GNNs:

   studies assume the information carried by edges can be directly described as an attribute vector.

   	1. This assumption holds well when edge features are categorical
   	1. cannot fully capture contextualized text semantic

2. PLM-GNN

   text information is first encoded by a PLM and then aggregated by a GNN

   1. such architectures process text and graph signals one after the other, and fail to simultaneously model the deep interactions

3. GNN-nested PLM

   inject network information into the text encoding process

   1. cannot be easily adapted to handle text-rich edges

## Proposed method

1. we conduct edge representation learning by jointly considering text and network information via a Transformer-based architecture (Edgeformer-E).
2. perform node representation learning using the edge representation learning module as building blocks (Edgeformer-N)

![image-20230602122224210](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230602122224210.png)

### Network-aware Edge Text Encoding with Virtual Node Tokens

Given an edge $e_{ij}=(v_i,v_j)$

Use a transformer to deal with text

introduce two virtual node tokens to represent $ v_i$ and $v_j$ to transformer

 $ v_i$ 和 $v_j$ 是连接边两个node的embedding]

![image-20230602122208124](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230602122208124.png)

### TEXT-A WARE NODE REPRESENTATION LEARNING (EDGEFORMER-N)

#### node-Aggregating Edge Representations

![image-20230602122457727](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230602122457727.png)

### Enhancing Edge Representations with the Node’s Local Network Structure

add one more virtual node in edge learning :获取与边节点连接的邻居边的信息

![image-20230602122540378](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230602122540378.png)

邻居边的embedding经过一个新tranformer，获取<cls>节点的embedding，作为参加edge learning的虚拟节点