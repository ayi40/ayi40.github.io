---
title: 常见优化函数
date: 2023/3/7 20:46:25
categories:
  - [ML, Basic]
---

.

<!-- more -->

# 常见优化函数

## 梯度下降GD

**梯度下降的核心思想：负梯度方向是使函数值下降最快的方向**

### 批次梯度下降BGD

![image-20230307202943629](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307202943629.png)

**优点**：在梯度下降法中，因为每次都遍历了完整的训练集，**其能保证结果为全局最优**

![image-20230307204257470](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307204257470.png)

**缺点**：我们需要对于每个参数求偏导，且在对每个参数求偏导的过程中还需要对训练集遍历一次，当训练集（m）很大时，计算费时

**解决方法**：使用minibatch去更新

### 随机梯度下降

为了解决BGD耗时过长，它是利用单个样本的损失函数对θ求偏导得到对应的梯度，来更新θ，更新过程如下：

![image-20230307204324179](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307204324179.png)

速度快，但受抽样影响大，**噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。**

![image-20230307204553630](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307204553630.png)

因为每一次迭代的梯度受抽样的影响比较大，学习率需要逐渐减少，否则模型很难收敛。在实际操作中，一般采用线性衰减：
$$
\eta_k=(1-\alpha)\eta_0+\alpha\eta_{\tau}
$$

$$
\alpha=\frac{k}{\tau}
$$

$\eta_0$:初始学习率

$\eta_{\tau}$： 最后一次迭代的学习率

$\tau$：自然迭代次数

$\eta_{\tau}$设为$\eta_0$的1%，k一般设为100的倍数。

**优点**：收敛速度快

**缺点**：

1. 训练不稳定：噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。

2. 选择适当的学习率可能很困难。 太小的学习率会导致收敛性缓慢，而学习速度太大可能会妨碍收敛，并导致损失函数在最小点波动。
3. 无法逃脱鞍点

<details>
    在数学中，鞍点或极小值点是函数图形表面上的一个点，其正交方向上的斜率(导数)均为零(临界点)，但不是函数的局	部极值。一句话概括就是：一个不是局部极值点的驻点称为鞍点。
 	*驻点：函数在一点处的一阶导数为零。
    <img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307205942585.png">

    ### min-batch 小批量梯度下降MBGD

**算法的训练过程比较快，而且也要保证最终参数训练的准确率**

m表示一个批次的数据个数

## 动量方法

### Momentum随机梯度下降

核心思想：Momentum借用了物理中的**动量**概念,即前一次的梯度也会参与运算。为了表示动量，引入了**一阶动量**m。![img](https://www.nowcoder.com/equation?tex=m&preview=true)是之前的梯度的累加,但是每回合都有一定的衰减。公式如下：
$$
m_t=\beta m_{t-1}+(1-\beta)\cdot g_t
$$

$$
w_{t+1}=w_t-\eta \cdot m_t
$$

$g_t$： 为第t次计算的梯度（就是现在要算这次）

$m_{t-1}$: 为之前梯度的累加

$\beta$: 动量因子

所以当前权值的改变受上一次改变的影响，类似加上了**惯性**。

优点：momentum能够加速SGD收敛，抑制震荡。并且动量有机会逃脱局部极小值(鞍点)。

1. 在梯度方向改变时，momentum能够降低参数更新速度，从而减少震荡；
2. 在梯度方向相同时，momentum可以加速参数更新， 从而加速收敛。

### Nesterov动量随机梯度下降法

Nesterov是Momentum的变种。与Momentum唯一区别就是，计算梯度的不同。Nesterov动量中，先用当前的速度临时更新一遍参数，在用更新的临时参数计算梯度。

在momentum更新梯度时加入对当前梯度的校正，让梯度“多走一步”，可能跳出局部最优解：
$$
w_t^*=\beta m_{t-1}+w_t
$$

$$
m_t=\beta m_{t-1}+(1-\beta)\cdot g_t
$$

$$
w_{t+1}=w_t-\eta \cdot m_t
$$

这里的$g_t$用临时点$w_t^*$计算的

## 更新学习率方法

### Adagrad

引入**二阶动量**，根据训练轮数的不同，对学习率进行了动态调整：

![image-20230307213914026](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307213914026.png)

**缺点**：仍然需要人为指定一个合适的全局学习率，同时网络训练到一定轮次后，分母上梯度累加过大使得学习率为0而导致训练提前结束。

### Adadelta(不是很懂)

![image-20230307215135905](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307215135905.png)

### RMSProp

AdaGrad算法在迭代后期由于学习率过小，可能较难找到一个有用的解。为了解决这一问题，RMSprop算法对Adagrad算法做了一点小小的修改，RMSprop使用指数衰减只保留过去给定窗口大小的梯度，使其能够在找到凸碗状结构后快速收敛。RMSProp法可以视为Adadelta法的一个特例，即依然使用全局学习率替换掉Adadelta法中的$s_t$:

![image-20230307215341546](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307215341546.png)

推荐$\eta_{global}=1,\rho=0.9,\epsilon=10^{-6}$

缺点：依然使用了全局学习率，需要根据实际情况来设定
优点：

1. 分母不再是一味的增加，它会重点考虑距离它较近的梯度（指数衰减的效果）
2. 只用了部分梯度加和而不是所有，这样避免了梯度累加过大使得学习率为0而导致训练提前结束。

### Adam

https://zhuanlan.zhihu.com/p/377968342

