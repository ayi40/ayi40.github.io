---
title: 王树森推荐系统公开课
date: 2023/3/13 20:46:26
categories:
  - [RecSys, BasicTurtorial]
---

.

<!-- more -->

# 基本概念

## 指标

### 消费指标

点击率=点击次数/曝光次数

点赞量=点赞次数/点击次数

收藏率=收藏次数/点击次数

转发率=转发次数/点击次数

阅读完成率=滑动到底次数/点击次数$\times f(笔记长度)$

### 北极星指标

用户规模：日活用户数（DAU），月活用户数（MAU）

消费：人均使用推荐时长、人均阅读笔记数量

发布： 发布渗透率、人均发布量

## 推荐系统链路



![image-20230313220835272](C:\Users\37523\AppData\Roaming\Typora\typora-user-images\image-20230313220835272.png)

1. 召回：快速从海量数据中取回几千个用户可能感兴趣的物品。
2. 粗排：用小规模的模型的神经网络给召回的物品打分，然后做截断，选出分数最高的几百个物品。
3. 精排： 用大规模神经网络给粗排选中的几百个物品打分，可以做截断，也可以不做截断。
4. 重排： 对精排结果做多样性抽样，得到几十个物品，然后用规则调整物品的排序。



## 实验流程

### 概要03推荐系统的AB测试：没仔细看以后补



# 召回



## 基于物品的协同过滤 ItemCF

基本思想：如果用户喜欢item1,而item1与item2相似，那么用户很可能喜欢item2.

### 基本结构：

![image-20230313223949470](C:\Users\37523\AppData\Roaming\Typora\typora-user-images\image-20230313223949470.png)

我们从用户历史互动知道用户对$item_j$，感兴趣利用下面公式计算对候选物品的兴趣分数
$$
\sum_jlike(user,item_j)\times sim(item_j,item)
$$
在这个例子中，用户对候选item的兴趣是：$2\times 0.1+1\times0.4+4\times0.2+3\times0.6=3.2$,我们计算所有item的分数，然后返回分数最高的若干个item



#### 计算item相似度

可以通过与item交互过的用户重合度计算item相似度（其中一种方法，也可以用KG）

1. 方法1：不考虑用户对物品的喜欢程度

$$
sim(i_1,i_2) = \frac{|W1 \cap W2|}{\sqrt[2]{|W1|\cdot |W2|}}
$$

其中，喜欢物品$i_1$的用户记作$W_1$,喜欢物品$i_2$的用户记作$W_2$.

2. 方法2： 考虑用户对物品的喜欢程度,使用余弦相似度！

   把每个item用向量表示
   $$
   i_1=[like(u_1,i_1),like(u_2,i_1),\cdots ,like(u_n,i_1)] \space u_n\in W
   $$

   $$
   i_2=[like(u_1,i_2),like(u_2,i_2),\cdots ,like(u_n,i_2)] \space u_n\in W
   $$

   $$
   W=W_1\cup W_2
   $$

   我们使用余弦相似度计算：
   $$
   similarity=cos(\theta) = \frac{A\cdot B}{||A||\space ||B||}
   $$
   如果有用户k只喜欢其中一个物品:只喜欢$i_1$不喜欢$i_2$,那么$i_2[k]=0$，所以点乘后第k项为0，所以点乘只与同时喜欢$i_1,i_2$的用户有关系，如下面公式
   $$
   sim(i_1,i_2) = \frac{\sum_{v\in V}like(v,i_i)\cdot like(v,i_2)}{\sqrt[2]{\sum_{u_1\in W_1}like^2(u_1,i_1)}\sqrt[2]{\sum_{u_2\in W_2}like^2(u_2,i_2)}}
   $$
   

### 运作基本流程

1. 实现做离线计算，预先计算两个索引：

   1. “user2item”：记录每个用户最近点击交互过的n个物品ID（lastN）

      ```PYTHON
      # example 不一定是公司真实的保存方式
      user2item={
          'u1':[[i1,like(u1,i1)],[i2,like(u1,i2)],...,[in,like(u1,in)]]
          ...
      }
      ```

      

   2. "item2item":计算物品之间两两相似度，记录每个物品最相似的k个物品。

      ```
      item2item={
      	#target item:[[similar item, similarity score]...]
      	'i1':[[i2,0.9],[i6,0.88]...]
      	'i2':...
      }
      ```

2. 线上做召回
   1. 给定用户ID，通过“user2item”找到用户近期感兴趣的物品列表(last-n)
   2. 对于last-n列表中每个物品，通过“item2item"找到top-k相似物品。现在有1个user，n个互动物品，nxk个候选物品。
   3. 计算候选物品兴趣分数
   4. 返回分数最高的100个物品作为推荐结果







