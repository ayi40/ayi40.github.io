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

# GRATIS

Paper：GRATIS: Deep Learning **G**raph **R**epresentation with T**a**sk-specifific **T**opology and Mult**i**-dimensional Edge Feature**s**

总结：计算一个全局representation-X，X经过MLP和reshape、softmax等操作变成和邻接矩阵大小相同的权重矩阵，然后得到一个edge出现的概率矩阵，概率大于一定阈值就补全边。

edge用向量表示而不是一个一维数（一维权重）。

## Methology

![image-20230617164254419](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617164254419.png)

![image-20230617164314513](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617164314513.png)

### Backbone

反正就是各种方法得到一个全局表示X

### Graph Definition

采用图原来的点和边

![image-20230617164457095](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617164457095.png)

### Task-specific Topology Prediction

用X计算出一个概率矩阵

![image-20230617164752154](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617164752154.png)

h（x）为mlp

如果概率大于某个阈值，增加新边

### **Multi-dimensional Edge Feature Generation**

根据边两端的节点计算边

![image-20230617165302836](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617165302836.png)

**VCR**

![image-20230617165318193](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617165318193.png)

**VVR**

![image-20230617165420670](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617165420670.png)

fifinally employ either a pooling layer or a fully-connected layer, to flatten $F_{i,x,j}$ and $F_{ *j,x,i*}$

![image-20230617165554284](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617165554284.png)

# SURGE

Paper： Knowledge-Consistent Dialogue Generation with Knowledge Graphs

总结：在KG大图中检索与文本相关的子图，用GCN计算node representation，用ENGNN计算
edge representation。然后用在后面的任务