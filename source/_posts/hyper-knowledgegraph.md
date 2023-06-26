---
title: Hyper Knowledge Graph
date: 2023/6/17 20:46:25
categories:
  - [GNN, hypergraph]
---

Hypergraph knowledge graph

<!-- more -->

# KHNN

Paper: Knowledge-Aware Hypergraph Neural Network for Recommender Systems

总结：用CKAN方法表示user和item，用hyperedge将l-hop的node全部连在一起，用（l-1）hop和l-hop concate卷积计算出l-hop节点的权重与l-hop节点相乘，再做conv最后得到该层的一维embedding，然后再aggregation

## Methodology

![image-20230617154902916](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617154902916.png)

### Knowledge-Aware Hypergraph Construction

**Initial Hyperedge Construction**

use user’s interacted items to represent user u

![image-20230617155650437](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617155650437.png)

use items, which have been watched by the same user, to construct the initial item set of item v 

![image-20230617155746509](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617155746509.png)

![image-20230617155754397](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617155754397.png)

**Knowledge Hyperedge Construction**

让l-hop neighbor 与(l-1)-hop neighbor在相连，即所有节点被一条hyper-edge相连，主要服务于下面的neighborhood convolution

![image-20230617161110960](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617161110960.png)

###  **Knowledge-Aware Hypergraph Convolution**

**Neighborhood Convolution**

![image-20230617162200596](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617162200596.png)

1. learn the transform matrix T from the entity vectors in both l-order and l-1-order hyperedges for vector permutation and weighting(entity vectors in l-order) . use 1-d conv to generate T , use another 1-d conv to aggregate the transformed vectors.

   ![image-20230617162545551](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617162545551.png)

   *conv*1 and *conv*2 are 1-dimension convolution but withdifffferent out channels.

2. for the initial hyper-edge

   ![image-20230617162625945](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617162625945.png)

3. add item v information

   ![image-20230617162651714](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617162651714.png)

4. combine

   ![image-20230617162708343](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617162708343.png)

5. aggregation

   ![image-20230617162731699](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617162731699.png)

# LGCL

Paper：Line Graph Contrastive Learning for Link Prediction

## Methodology

 ![image-20230617174642449](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617174642449.png)

# HPR

Paper: Empowering Knowledge Graph Construction with Hyper-graph for Personalized Recommendation

basic idea: You might like something that someone with similar preferences likes you.

总结：将相似度高的user作为hyper-edge（本质上是扩展了target user的1-hop neighbor），通过计算user的l-hop neighbor与target item的相似度来计算出user的embedding

## Methodology

we would like to calculate the probability of $u_1$ will interact with $i$

![image-20230617144000884](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617144000884.png)

### Hyper-graph learning

1. adopt the cosine similarity to estimate the relevance between users.
2. select some users with the highest similarity as the hyper-edge.

这里假设u1与u2最为相似，所以将u1、u2成为一条hypergraph

### Knowledge Graph Construction

given:

$\vec{i}$: the embedding of item i

$S_{u_l}^1$​ : the l-hop neighbor of user1, which takes the entities which with implicit interaction behaviour with users as head entities.

$S_{u_2}^l$ : the l-hop neighbor of user2, because there is a hyperedge consist of u1 and u2



1-hop cal:

gain information between item-i and the 1-hop neighbor of user

![image-20230617151011714](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617151011714.png)

![image-20230617150959919](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617150959919.png)

$N_u^1:S^1_{u_1} \or S^1_{u_2}$  user one-hop neighbor (include hyperedge)

multi-hop cal: get $q_u^{l}$

the user embedding:

![image-20230617151325200](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617151325200.png)

predict:

![image-20230617151352536](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230617151352536.png)



# HypE

Paper:Knowledge Hypergraphs: Extending Knowledge Graphs Beyond Binary Relations

score： convolution-based embedding method for knowledge hypergraph

总结：做KG图的连接预测，无GNN方法，主要是embedding计算方法。考虑到entity在triple中的i个位置，在这个位置有训练出来的filter，对embbeding进行转换，最后计算概率score，

计算成本较低。

