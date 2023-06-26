---
title: DHGCN(2HRDR)
date: 2023/6/21 20:46:25
categories:
  - [GNN, hypergraph]
---

propose DHGCN,2HRDR

<!-- more -->

# Background

task: 基于知识图谱的问答系统，在知识图谱中检索与问题相关的多个元组



contribution: propose a convolutional network for directed hypergraph



# DHGCN

## HGCN

given a hypergraph $G=(V,E,W)$, as well as the incidence matrix $H\in R^{|V|\times |E|}$

the edge and vertex degrees:

![image-20230621153922205](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621153922205.png)



while the hypergraph convolutional networks is:

![image-20230621155931211](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621155931211.png)

![IMG_0183(20230621-201655)](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/IMG_0183(20230621-201655).JPG)

## DHGCN

the directed hypergraph can be denoted by two incidence matrices $H^{head}$ and $H^{tail}$ 

the degree:

![image-20230621160733846](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621160733846.png)

the directed hypergraph convolutional networks is:

![image-20230621160751671](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621160751671.png)

<img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/IMG_0184(20230621-202928).JPG" alt="IMG_0184(20230621-202928)" style="zoom:50%;" />

# 2HRDR

## Task Definition

given a knowledge graph $K=(V,E,T)$ and $q=(w_1,w_2,\cdots ,w_{|q|})$.

the task aims to pick the answers from *V*. 

## Method

### **Directed Hypergraph Retrieval and** **Construction**

find subgraph

1. obtain seed entities from the question by entity linking
2. get the entities set within L hops to form a subgraph
3. get  $H^{head}$ and $H^{tail}$ 

### Input Encoder

1. apply a bi-LSTM to encode question and obtain hidden states $H\in R^{|q|\times h}$,we assume h=d

2. employ co-attention to learn query-aware entity representation

   ![image-20230621170223607](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621170223607.png)

### Reasoning over Hypergraph

![image-20230621163315446](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621163315446.png)

![image-20230621172646592](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621172646592.png)

1. Learn Relation Representation Explicitly

   1. combine entity embedding and co-attention 

      ![image-20230621173702999](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621173702999.png)

   2. propagation

      ![image-20230621173728920](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621173728920.png)

   3. aggregation

      ![image-20230621173815550](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621173815550.png)

2. Allocate Relation Weights Dynamically(dynamically allocated hop-by-hop)

   1. use co-attention to cal $R_{co\_attn}$

   2. compute the weight of edge

      ![image-20230621174053983](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621174053983.png)

      ![image-20230621174101797](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621174101797.png)

3. Update Entity Adaptively

   ![image-20230621174229381](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230621174229381.png)
