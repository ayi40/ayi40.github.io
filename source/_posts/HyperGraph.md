---
title: HyperGraph
date: 2023/5/26 20:46:25
categories:
  - [GNN,hypergraph]
---

Hypergraph

<!-- more -->

# HGNN

use the matrix to represent hypergraph 

a hyperedge convolution operation is designed

can incorporate with multi-modal data and complicated data correlations(use below figure2 method to combine different data type )

## Hypergraph and adjacency matrix

hypergraph-一条边可以同时连接多个点

![image-20230605145232861](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230605145232861.png)

adjacency matrix:

![image-20230605145324681](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230605145324681.png)

## Method

### hypergraph learning statement

hypergraph defined as $G = (V,\varepsilon, W)$ 

the degree of vertex $v$ defined as $d(v)=\sum_{e\in \varepsilon}h(v,e)$——一个点与多少条边相连

the degree of hyperedge defined as $\delta (v)=\sum_{v\in V}h(v,e)$——一条边与多少个点相连

$D_e$ and $D_v$ denote the diagonal matrices of the edge degrees and the vertex degrees, respectively

example:

![image-20230605153837827](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230605153837827.png)

### Spectral convolution on hypergraph

![image-20230605155756968](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230605155756968.png)

频域理解不深，要完全看懂要好久，先简单略一下。



# HyperGCN

Hypergraph, a novel way of training a GCN for SSL on hypergraphs based on tools from sepctral theory of hypergraphs

主要是把hypergraph转为简单拉普拉斯图

## Method

![image-20230606094309361](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230606094309361.png)

### hypergraph generation

1. 对图上任意边$e\in E$, 令$(i_e,j_e):=argmax_{i,j\in e}|S_i-S_j|$ ,即返回同一个边上距离最远的两个顶点表示$(i_e,j_e)$为随机,切断点的联系。
2. 为剩下的边添加权重，权重为hyperedge的权重，构造简单的邻接矩阵。
3. 归一化计算拉普拉斯矩阵

### GNN

利用带权重的拉普拉斯矩阵计算GCN



# Hypergraph Convolution and Hypergraph Attention

