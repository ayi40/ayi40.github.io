---
title: KGAT
date: 2023/2/24 20:46:25
categories:
  - [RecSys, KGRec]
---

.

<!-- more -->

# Background

利用KG作为辅助信息，并将KG与user-item graph 整合为一个图

## Background



![image-20230224155340260](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230224155340260.png)

**Previous model:** 

CF: behaviorally similar users would exhibit similar preference on items.

**focus on the histories of similar users who also watched $i1$, i.e., $u4$ and $u5$;**

SL:  transform side information into a generic feature vector, together with user ID and item ID, and feed them into a supervised learning (SL) model to predict the score. 

**emphasize the similar items with the attribute $e1$, i.e.$ i2$.**



**current problem:**

 existing SL methods fail to unify them, and ignore other relationships in the graph:

1.  the users in the yellow circle who watched other movies directed by the same person $e_1$.
2. the items in the grey circle that share other common relations with $e_1$.



## User-Item Bipartite Graph: $G_1$


$$
\{(u,y_{ui},i)|u\in U, i\in I\}
$$
$U$: user sets

$I$: item sets

$y_{ui}$: if user $u$ interacts with item $i$ $y_{ui}$=, else  $y_{ui}$=0.



## Knowledge Graph $G2$

$$
\{(h,r,t)|h,t\in E, r\in R\}
$$

$t$ there is a relationship $r$ from head entity *h* to tail entity $t$.

## $CKG$: Combination of $G1$ and $G2$

1.  represent each user-item behavior as a triplet $ (u, Interact,i)$, where$ y^{ui}$ = 1.
2. we establish a set of item-entity alignments

$$
A = \{(i, e)|i ∈ I, e ∈ E \}
$$

3. based on the item-entity alignment set, the user-item graph can be integrated with KG as a unified graph.

$$
G = \{(h,r,t)|h,t ∈ E^′,r ∈R^′\}
$$

$$
E^′ = E ∪ U
$$

$$
R^′ = R ∪ {Interact}
$$

# Methodology

![image-20230222185609261](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222185609261.png)

KGAT has three main components:

1. Embedding layer
2. Attentive embedding propagation layer
3. prediction layer

## Embedding layer

Using **TransR** to calculate embedding

**Assumption**: if a triplet (h,r,t) exist in the graph,
$$
e^r_h+e_r\approx e_t^r
$$
Herein, $e^h$, $e^t$ ∈ $R^d$ and $e^r$ ∈ $R^k$are the embedding for *h*, *t*, and *r*; and $e^r_h$, $e^r_t$ are the projected representations of  $e^h$,  $e^t$  in the relation *r*’s space.



**Plausibility score**:

![image-20230222193700417](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222193700417.png)

$W_r ∈ R^{k\times d}$ is the transformation matrix of relation *r*, which projects entities from the *d*-dimension entity space into the *k* dimension relation space. 

A lower score suggests that the triplet is more likely to be true.

**Loss**:

![image-20230222195306105](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222195306105.png)

$\{(h,r,t,t^′ )|(h,r,t) \in G, (h,r,t^′ ) \notin G\}$, $(h,r,t^′ )$ is a negative sample constructed by replacing one entity in a valid triplet randomly.

*σ*(·): sigmoid function, ——》将分数映射再0-1区间，归一化

？？？？？？？？？？？why this layer model working as a regularizer

## Attentive Embedding Propagation Layers(upon GCN)

### First-order propagation





和之前模型不同，这个的propagation layer encode了$e_r$.

For entity h, the information propagating from neighbor is :

![image-20230222201217039](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222201217039.png)

$π(h,r,t)$: to controls the decay factor on each propagation on edge (*h*,*r*,*t*), indicating how much information is propagated

from *t* to *h* conditioned to relation *r*.

For $π(h,r,t)$, we use attention mechanism:

![image-20230222204949423](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222204949423.png)

This makes the attention score dependent on the distance between $e^h$ and $e^t$ in the relation *r*’s space.

这里，tanh用于增加非线性因素；但不缺定是否有归一化作用？？？？？归一化就可以把这个function的大小集中在角度上，但是这样$e^h_t$也没有归一化，到时候看看输出参数

and than use softmax to normalize(no need to use as$\frac1{|N_t |}$$\frac1{|N_t ||N_h |}$)

![image-20230222234256082](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222234256082.png)

The final part is aggregation, threre are three choices:

1. GCN aggregator

![image-20230222235616598](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222235616598.png)

2. GraphSage aggregator

![image-20230222235806956](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222235806956.png)

3. Bi-Interaction aggregator

![image-20230223000647293](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223000647293.png)

###  xxxxxxxxxx class UV_Aggregator(nn.Module):    """    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).    """​    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):        ...​    def forward(self, nodes, history_uv, history_r):        # create a container for result, shpe of embed_matrix is (batchsize,embed_dim)        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)​        # deal with each single item nodes' neighbors        for i in range(len(history_uv)):            history = history_uv[i]            num_histroy_item = len(history)            tmp_label = history_r[i]​            # e_uv : turn neighbors(user node) id to embedding            # uv_rep : turn single node(item node) to embedding            if self.uv == True:                # user component                e_uv = self.v2e.weight[history]                uv_rep = self.u2e.weight[nodes[i]]            else:                # item component                e_uv = self.u2e.weight[history]                uv_rep = self.v2e.weight[nodes[i]]​            # get rating score embedding            e_r = self.r2e.weight[tmp_label]            # concatenated rating and neighbor, and than through two layers mlp to get fjt            x = torch.cat((e_uv, e_r), 1)            x = F.relu(self.w_r1(x))​            o_history = F.relu(self.w_r2(x))            # calculate neighbor attention and fjt*weight to finish aggregation            att_w = self.att(o_history, uv_rep, num_histroy_item)            att_history = torch.mm(o_history.t(), att_w)            att_history = att_history.t()​            embed_matrix[i] = att_history        # result (batchsize, embed_dim)        to_feats = embed_matrix        return to_featspython

![image-20230223000834658](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223000834658.png)

![image-20230223000850960](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223000850960.png)

## Model Prediction

multi-layers combination and inner product

![image-20230223001515690](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223001515690.png)

![image-20230223001526127](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223001526127.png)

## Optimizazion

### loss

![image-20230223002144083](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223002144083.png)

$L_{cf}$ is BPR Loss

![image-20230223002439536](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223002439536.png)

$L_{kg}$ is loss forTranR .

![image-20230222195306105](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230222195306105.png)

### Optimizer

Adam

### updata method



we update the embeddings for all nodes; 

hereafter, we sample a batch of (*u*,*i*, *j*) randomly, retrieve their representations after *L* steps of propagation, and then update model parameters by using the gradients of the prediction loss.

在同一个epoch中，先把所以数据扔进tranR训练，得到loss（此时不更新参数）

然后sample算BPR LOSS

# EXPERIMENTS

## RQ1: Performance Comparison 

1. regular dataset

![image-20230223004843914](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223004843914.png)

2. Sparsity Levels

   ![image-20230223005253034](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223005253034.png)

KGAT outperforms the other models in most cases, especially on the two sparsest user groups.

说明KGAT能够缓解稀疏性影响

## RQ2：Study of KGAT

1. study of layer influence and effect of aggregators

![image-20230223010038345](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223010038345.png)

2. cut attention layer and TransR layer

   ![image-20230223010347815](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230223010347815.png)
