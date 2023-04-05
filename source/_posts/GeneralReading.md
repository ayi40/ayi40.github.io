---
title: GeneralReading
date: 2023/3/29 20:46:26
categories:
  - [RecSys, KGRec]
---

MKR,RKGE,HAGERec,entity2rec,HAKG

<!-- more -->





# MKR

## Framework

![image-20230329182212440](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230329182212440.png)

The framework of MKR is illustrated in Figure 1a. 

MKR consists of three main components: recommendation module, KGE module, and cross&compress units.

1. The recommendation module on the left takes a user and an item as input, and uses a multi-layer perceptron (MLP) and cross&compress units to extract short and dense features for the user and the item, respectively. The extracted
   features are then fed into another MLP together to output the predicted probability. 

2. Similar to the left part, the KGE module in the right part also uses multiple layers to extract features from the head and relation of a knowledge triple, and outputs the representation of the predicted tail under the supervision of a score function f and
   the real tail.

3. The recommendation module and the KGE module are bridged by specially designed cross&compress units. The proposed unit can automatically learn high-order feature interactions of items in recommender systems and entities in the knowledge
   graph.

u: MLP to update

v: cross&compress units

r: MLP to update

h: cross&compress units

## Loss function

![image-20230329184018259](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230329184018259.png)

# RKGE

RKGE first automatically mines all qualified paths between entity pairs from the KG, which are then encoded via a batch of recurrent networks, with each path modeled by a single recurrent network.

It then employs a pooling operation to discriminate the importance of different paths for characterizing user preferences towards items.

## framework

![image-20230329205324109](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230329205324109.png)

### Semantic Path Mining

Strategy 

1. We only consider user-to-item paths $P(u_i,v_j)$ that connect $u, i$ with all her rated items.
2. We enumerate paths with a length constraint.

### Encode path

use recurrent networks 

#### Embedding layer

generate the embedding of entities

#### Attention-Gated Hidden Layer

就是一个RNN网络的变种



# HAGERec

## Framework

![image-20230330191130895](C:\Users\37523\AppData\Roaming\Typora\typora-user-images\image-20230330191130895.png)

four components:

1. Flatten and embedding layer: flatten complex high-order relations and embedding the entities and relations as vectors.

2. GCN learning layer: uses GCN model to propagate and update user’s and item’s embedding via a bi-directional entity propagation strategy
3.  Interaction signals unit: preserves interaction signals structure of an entity and its neighbor network to give a more complete picture for user’s and item’s representation.
4.  Prediction layer:  utilizes the user’s and item’s aggregated representation with prediction-level attention to output the predicted score.

### Flatten and embedding

![image-20230330193624700](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230330193624700.png)

flatten high-order connection to the path: $u\rightarrow^{r1}v\rightarrow^{r2}e_{u1}\rightarrow^{r3}e_{v3}$

embedding: initialized embedding vectors.

### GCN learning unit

user and item use the same propagation and aggregate strategy.

![image-20230330200154040](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230330200154040.png)

![image-20230330200204043](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230330200204043.png)

$h^T, W, b$: learned parameters

neighbor sample(only get fixed number neighbor): $\alpha_{e_v,e_{nv}}$ would be regarded as the similarity of each neighbor entity and central entity. Through this evidence, those neighbors with lower similarity would be filtered.

###  Interaction signals unit

![image-20230330201306937](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230330201306937.png)

区别：上面是相加，下面事相乘

so GCN unit + interaction unit =

![image-20230330201407720](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230330201407720.png)

### Predict

![image-20230330201630216](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230330201630216.png)

# Entity2rec

## Framework

![image-20230331170104462](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230331170104462.png)

### node2vec

![image-20230331165454743](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230331165454743.png)

将图用random walk转化为word格式，用词袋模型计算vector。

### Property-specific knowledge graph embedding

在node2vec基础上加上relation embedding，基于p子图在p空间上优化node vector

maximize the dot product between vectors of the same neighborhood

![image-20230331170246301](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230331170246301.png)

Ze-negative sampling

N(e): neighbor of entity

### subgraph

#### Collaborative-content subgraphs

只保留单一relation，但连接性很差，对random walk效果不好

![image-20230331170405382](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230331170405382.png)

所有子图可以分成两张类型：feedback子图(user-item图)和其他子图

用下面方法来计算推荐分数：

![image-20230331170819790](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230331170819790.png)

R+(u) denotes a set of items liked by the user u in the past.

s(x): similarity socre

#### Hybrid subgraphs

$K_p^+=K_p \cup(u,feedback,i)$

![image-20230331171032188](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230331171032188.png)

![image-20230331171139039](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230331171139039.png)

# HAKG

## Framework

![image-20230401160018626](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230401160018626.png)

1. Subgraph Construction ：it automatically constructs the expressive subgraph that links the user-item pair to represent their connectivity;
2. Hierarchical Attentive Subgraph Encoding ： the subgraph is further encoded via a hierarchical attentive embedding learning procedure, which first learns embeddings for entities in the subgraph with a layer-wise propagation mechanism, and then attentively aggregates the entity embeddings to derive the holistic subgraph embedding; 
3.  Preference Prediction ： with the well-learned embeddings of the user-item pair and their subgraph connectivity, it uses non-linear layers to predict the user’s preference towards the item. 

### Subgraph Construction

path sampling and then reconstructs the subgraphs by assembling the sampled paths between user-item pairs

#### path sampling

use random walk get path from u to i and length<=6, uniformly sample K paths

#### Path Assembling

just assemb the K paths

### Hierarchical attentive subgraph encoding

#### entity embedding learning

![image-20230401173559684](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230401173559684.png)

##### Embedding Initialization

1. initial
2. $e_h^{(0)}=MLP(e_h \space concatenation \space t_h)$

##### Semantics Propagation

![image-20230401173656033](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230401173656033.png)

##### Semantics Aggregation

![image-20230401173801411](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230401173801411.png)

final entity embedding is $e_h^{(L)}$

constitute an entity embedding matrix H(u,i) for the whole subgraph :

$H_{(u,i)}=[e_1,e_2,\cdots,e_n]$

#### sub-graph embedding learning

use self-attention mechanism optimize the entities embeding of subgraph

Than use pooling method to get subgraph embedding

### Prediction

![image-20230401175552413](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230401175552413.png)









