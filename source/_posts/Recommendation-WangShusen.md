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



## AB测试

完成离线测试后，使用线上小流量AB测试考察指标，或者用AB测试调参（GNN深度）

### 随机分桶

![image-20231119172737630](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20231119172737630.png)

在不同的桶上使用不同的策略或参数实验。

### 分层实验

不同的部门都需要做AB测试，每个部门对应一个层，分层实验满足：

1. **同层互斥**：同一个部门做实验不能使用同一个桶；例：GNN实验占了召回层4个桶，其它召回实验只能用剩下的6个桶。
2. **不同层正交**：每一层独立随机对用户做分桶。每一层都可以独立用100%的用户做实验。



# 召回

## 协同过滤

### 基于物品的协同过滤 ItemCF

基本思想：如果用户喜欢item1,而item1与item2相似，那么用户很可能喜欢item2.

#### 基本结构：

![image-20230313223949470](C:\Users\37523\AppData\Roaming\Typora\typora-user-images\image-20230313223949470.png)

我们从用户历史互动知道用户对$item_j$，感兴趣利用下面公式计算对候选物品的兴趣分数
$$
\sum_jlike(user,item_j)\times sim(item_j,item)
$$
在这个例子中，用户对候选item的兴趣是：$2\times 0.1+1\times0.4+4\times0.2+3\times0.6=3.2$,我们计算所有item的分数，然后返回分数最高的若干个item



##### 计算item相似度

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
   
3. 

3. 皮尔逊系数
   $$
   sim(i,j)=\frac{\sum_{p\in P}(R_{i,p}-\bar R_i)(R_{j,p}-\bar R_j)}{\sqrt{\sum_{p\in P}(R_{i,p}-\bar R_i)^2}\sqrt{\sum_{p\in P}(R_{j,p}-\bar R_j)^2}}
   $$

#### 运作基本流程

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



### Swing召回通道

如果两个Item的重合用户来源于一个小圈子（微信群），一个小圈子用户同时与两个Item交互，不能说明两个Item相似，如果很多不相关的用户交互两个Item，说明Item相似。

#### 基本结构

1. 计算用户重合度

用户$$u_1$$喜欢的物品记作集合$$J_1$$

用户$$u_2$$喜欢的物品记作集合$$J_2$$

定义两个用户的重合度：
$$
overlap(u_1,u_2)=|J_1\cap J_2|
$$
用户$$u_1$$和$$u_2$$的重合度高，则他们可能来自一个小圈子，要降低他们的权重。

2. 计算物品相似度

喜欢物品$$i_1$$的用户记作集合$$W_1$$

喜欢物品$$i_2$$的用户记作集合$$W_2$$
$$
V=W_1\cap W_2
$$

$$
sim(i_1,i_2) = \sum_{u_1\in V}\sum_{u_2\in V}\frac{1}{\alpha+overlap(u_1,u_2)}
$$

u1u2都对物品i1i2感兴趣，这样的用户越多，说明物品越相似

$$\alpha$$是超参数

### 基于用户的协同过滤（UserCF）

假设：u1与u2兴趣十分相似，u1可能会对u2交互的item感兴趣

何为兴趣相似：

1. 点击、点赞、收藏、转发的笔记有很大重合
2. 关注的作者有很大的重合

#### 基本结构

![image-20230525170605162](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230525170605162.png)
$$
\sum_jsim(user,user_j)\times like(user_j,item)
$$

##### 计算User相似度

1. 计算User相似度

   把每个用户表示为一个稀疏向量，向量每个元素对应一个物品。相似度sim就是两个向量夹角的余弦。$$u_1\cdot u_2$$结果就是$$|I|$$

$$
sim(u_1,u_2) = \frac{|I|}{\sqrt{|J_1|\cdot|J_2|}}
$$

$$J_1$$: 用户$$u_1$$喜欢的物品集合

$$J_2$$: 用户$$u_2$$喜欢的物品集合

$$I$$：$$J_1\cap J_2$$

|*|:集合的大小

$$sim(u_1,u_2)\in[0,1]$$,越大代表用户越相似

2. 降低热门物品权重

大家都喜欢哈利波特，哈利波特对用户相似度计算意义小,所以我们降低热门物品权重
$$
sim(u_1,u_2) = \frac{\sum_{l\in I}weight(l)}{\sqrt{|J_1|\cdot|J_2|}}
$$

$$
weight(l) = \frac{1}{log(1+n_l)}
$$

$$n_l$$: 喜欢物品l的用户数量，反应物品的热门程度。$$n_l$$越大，$$log(1+n_l)$$越大，权重越小

#### 运作基本流程

1. 实现做离线计算，预先计算两个索引：

   1. “user2item”：记录每个用户最近点击交互过的n个物品ID（lastN）

      ```PYTHON
      # example 不一定是公司真实的保存方式
      user2item={
          'u1':[[i1,like(u1,i1)],[i2,like(u1,i2)],...,[in,like(u1,in)]]
          ...
      }
      ```

      

   2. "user2user":计算用户之间两两相似度，记录每个用户最相似的k个用户。

      ```
      user2user={
      	#target user:[[similar user, similarity score]...]
      	'u1':[[u2,0.9],[u6,0.88]...]
      	'u2':...
      }
      ```

2. 线上做召回

   1. 给定用户ID，通过“user2user”找到top-k相似用户
   2. 对于top-k列表中每个用户，通过“user2item"找到用户近期感兴趣物品列表(last-n)。
   3. 对于召回的nk个相似物品，用公式预估用户对每个物品的兴趣分数
   4. 返回分数最高的100个物品，作为召回结果

### 协同过滤缺点

。。。

## 向量召回

### 矩阵补充 Matrix Completion

用于填充评分矩阵中无评分的部分，通过求user与item embedding的内积

<img src="C:\Users\37523\AppData\Roaming\Typora\typora-user-images\image-20230626164840701.png" alt="image-20230626164840701" style="zoom:33%;" />

![image-20230626163645000](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626163645000.png)

#### 数据集

1. （用户ID,物品ID，兴趣分数）————》$dataset={(u,i,y)}$

2. 正负例子（0-4分）：
   1. 负例子：曝光没有点击-0分
   2. 正例子：点击、点赞、收藏、转发-各1分

#### 训练

$$
min_{A,B}\sum _{(u,i,y)\in dataset}(y-<a_u,b_i>)^2
$$

#### 缺点

1. 仅用ID embedding，没利用物品、用户的属性。
2. 负样本选取方法不对。
3. 做训练方法不好
   1. 内积效果不如余弦相似度
   2. 用回归方法不如用分类方法。

#### 运作基本流程

1. 离线计算
   1. 训练矩阵A、B（embedding层的参数，A for user, B for item）
   2. 由于矩阵很大，为了快速读取使用hash方法：
      1. 把矩阵A存储到key-value表{user_id: user_embedding}。
      2. （加速最近邻查找）将item分区保存至key-value表

2. 线上服务
   1. 通过用户ID查询用户向量，记作A。
   2. 最近邻查找：查找用户最优可能感兴趣的k个物品作为召回结果。
      1. 第i号物品的embedding向量记作$b_i$
      2. 求$<a,b_i>$
      3. 返回内积最大的k个物品



**加速最近邻查找方法**：

一般item有几亿个，暴力计算内积并排序过慢

方法：

1. 确定衡量最近邻标注：欧氏距离最小（L2距离），向量内积最大（内积相似度），向量夹角余弦最大（cosine相似度）

2. 根据衡量标准将所有item embedding分块，下面为根据余弦相似度分块的例子，每一个区域用一个向量E表示，通过key-value表保存区域向量E与区域中所有向量的embedding。

   ![image-20230626171141459](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626171141459.png)

3. 求区域向量与user的余弦相似度，获取结构最大区域。再将区域中所有的item暴力枚举算相似度。

   ![image-20230626171541263](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626171541263.png)

### 双塔模型

融合除了ID以为的别的特征

![image-20230626173816258](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626173816258.png)

![image-20230626173835292](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626173835292.png)



#### 数据集

1. 正样本

   曝光且有点击的（user，item）组

   问题：少部分物品占据大部分点击，导致正样品大多是热门物品，对冷门物品不公平。

   解决：过采样冷门物品，或降采样热门物品

   ​	过采样：一个样品出现多次

   ​	降采样：一些样本被抛弃

2. 负样本

   混合几种负样本：50%的简单负样本，50%的困难负样本

   我们分别讨论下面三种可以作为负样本的数据。

   ![image-20230626202449765](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626202449765.png)

   1. 简单负样本

      没有被召回的数据: **全体物品**

      没被召回的数据，大概率是用户不感兴趣的，未被召回的样本约等于全体物品，所以在全体物品中做抽样作为负样本。

      **均匀抽样**：正样本大多是热门物品，负样本大多是冷门物品。（因为热门物品比例小），所以我们需要利用非均匀抽样打压热门物品。

      **非均抽采样**：负样本抽样概率与热门程度（点击次数）正相关，$抽样概率\propto (点击次数)^{0.75}$

      

      **Batch内负采样**

      ![image-20230626203300364](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626203300364.png)

      一个batch有n个正样本对，一个用户和n-1个物品组成负样本，batch中一共有n（n-1）个负样本对。

      此时，热门物品成为负样本的概率过大（热门物品成为正样本概率大）：$抽样概率\propto (点击次数)$

      所以做训练时，兴趣分数调整为：$cos(a,b_i)-logp_i$降低热门物品作为负样本的惩罚

   2. **困难负样本**：用户有一点兴趣，但兴趣不够，特别容易分错

      被粗排淘汰的物品（比较困难）

      精排分数靠后的物品（非常困难）

   **注意**：不能用曝光但没有点击的样本，因为能通过精排（更复杂的模型）的样本已经是用户比较感兴趣的样本，可能只是机缘巧合没有点击，训练召回不能用这一类样本，但是训练排序可以

#### 训练

##### Pointwise

当做二分类任务，对于正样本，鼓励cos(a,b)接近+1；对于负样本，鼓励cos(a,b)接近-1

##### Pairwise

鼓励$cos(a,b^+)$大于$cos(a,b^-)$

Triplet hinge loss:
$$
L(a,b^+,b^-) = max\{0,cos(a,b^-)+m-cos(a,b^+)\}
$$
m为超参数

Triplet logistic loss:
$$
L(a,b^+,b^-) = log(1+exp[\sigma(cos(a,b^-)-cos(a,b^+))])
$$


##### Listwise

![image-20230626175637609](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626175637609.png)

#### 运作基本流程

1. 离线存储：把物品向量b存入向量数据库。
2. 线上召回：查找用户最感兴趣的k个物品。
   1. 给定用户ID和画像，线上用升级网络算用户向量A。
   2. 最近邻查找

为什么用户向量要在线计算：

1. 没做一次召回只用到一个用户向量A，计算成本较小。
2. 用户兴趣动态变化，物品较稳定。

#### 模型更新

![image-20230626205649864](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626205649864.png)

**全量更新**：

​	在昨天模型参数基础上做训练（不是随机初始化），用昨天的数据，shuffle后训练一个epoch后发布新的用户塔神经网络和物品向量，供线上召回使用。

**增量更新**：

​	用户兴趣随时发生变化，实时收集线上数据，对模型做online learning，增量更新ID Embedding参数(不更新神经网络其他部分参数)，发布用户ID Embedding，供用户塔线上计算用户向量。

**不能只做增量更新，不做全量更新**

1. 小时级数据有偏差，分钟级偏差更大。
2. 全量更新：random shuffle一天数据，做 1epoch训练；增量更新按照数据从早到晚顺序做1epoch训练，全量更新效果更好。

#### 自监督学习

##### 背景

推荐系统头部效应严重：少部分物品占据大部分点击，大部分物品曝光、点击次数不高，导致高点击物品的表征学习的好，长尾物品的表征学的不好，用自监督学习做data augmentation，更好的学习长尾物品的向量表征。

##### Method

![image-20230626212328043](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626212328043.png)

###### 特征变换方法

1. Random Mask

   随机选一些离散特征(例如类目特征)，把它们遮住

   例子：$U=\{数码，摄影\}$->$U'-\{default\}$

2. Dropout

   一个物品可以有多个类目，那么类目是一个多值离散特征。Dropout会随机丢弃特征中50%的值。

   例子：$U=\{数码，摄影\}$->$U'-\{数码\}$

3. complementary互补特征

   假设物品一共有四种特征：ID,类目，关键词，城市

   随机分成两组：{ID,关键词}，{类目，城市}

   {ID,default，关键词，default}作为表征i‘

   {default，类目，default，城市}作为表征i‘’

4. Mask一组关联的特征

   p(u): 某特征取值为u的概率

   p(u,v):某特征取值为u，另一个特征取值为v同时发生的概率

   离线计算特征的两两关系，用户信息(mutual information):
   $$
   MI(U,V)=\sum_{u\in U}\sum_{v\in V}p(u,v)\cdot log\frac{p(u,v)}{p(u)\cdot p(v)}
   $$
   假设一共有k种特征。离线计算两两MI，得到kxk的矩阵，随机选一个特征为种子，找到种子最相关的k/2中特征Mask掉，保留其余的k/2中特征。

   比random mask、dropout、互补特征等方法效果更好，但方法复杂实现难度大不容易维护。

###### 自监督训练

从全体物品中均匀抽样得到m个物品，作为一个batch。

做两类特征变换，物品他输出两组向量。

![image-20230626214137364](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626214137364.png)

###### 自监督训练+正常训练

![image-20230626214246267](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626214246267.png)

## 不适合召回的模型

![image-20230626175824199](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230626175824199.png)

召回需要计算的item量很大，所以我们一般只做后期融合（计算相似度的时候再融合user和item的embedding），因为融合步骤一般要在线计算不能离线计算完保存（需求内存量太大）。如果我们在召回阶段就要让user embedding与上亿个item embedding过神经网络模型，这样时间复杂度太高了。

## 其他方式召回

地理召回

作者召回

缓存召回：复用前n次推荐精排的结果

## 曝光过滤与链路

。。





# 排序



