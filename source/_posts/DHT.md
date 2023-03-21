---
title: DHT
date: 2023/3/20 20:46:25
categories:
  - [GNN,EdgeLearning]
---

DHT, which transforms the edges of a graph into the nodes of a hypergraph.

ENGNN, use hypergraph after DHT to propagation

<!-- more -->

# Background

Before methods only capture edge information implicitly, e.g. used as weight.

# Contribute

1. propose DHT, Dual Hypergraph Transformation
2. propose a novel edge representation learning scheme ENGNN by using DHT.
3. propose novel edge pooling methods.

# Method

## DHT： how to transfer graph to hypergraph

![image-20230320171020430](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230320171020430.png)

### Step1: Get origin graph representation

Firstly, we get the initial node feature and edge feature.
$$
node \space  feature: X\in R^{n\times d}
$$

$$
edge\space feature: E\in R^{m\times d'}
$$

Than we use an incidence matrix M rather than an adjacency matrix to represent graph structure.
$$
incidence\space matrix: M\in \{0,1\}^{n\times m}
$$
So the origin graph is 
$$
G=(X,M,E)
$$

### Step 2: Use DHT to get hypergraph $G^*$

The hypergraph represent
$$
G^*=(X^*,M^*,E^*)
$$

$$
X^*=E
$$

$$
M^*=M^T
$$

$$
E^*=X
$$

$$
DHT:G=(X,M,E)->G^*=(E,M^T,X)
$$

While DHT is a bijective transformation:
$$
DHT:G^*=(E,M^T,X)->G=(X,M,E)
$$

## EHGNN: an edge representation learning framework using DHT

$$
E^{(l+1)}=ENGNN(X^{(l)},M,E^{(l)})=GNN(DHT(X^{(l)},M,E^{(l)}))
$$

So ENGNN consists of DHT and GNN, while GNN can be any GNN function.

After ENGNN, EHGNN, $E^{(L)}$ is returned to the original graph by applying DHT to dual hypergraph $G^∗$. Then, the remaining step is how to make use of these edge-wise representations to finish the task.



## Pooling

To be continue...

# Advantage

## DHT



1. low time complexity

![image-20230320165122363](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230320165122363.png)