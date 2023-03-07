---
title: 常见激活函数
date: 2023/2/28 20:46:25
categories:
  - [ML, Basic]
---

.

<!-- more -->

# 常见激活函数

激活函数作用：加入非线性因素

## Sigmoid

$$
\sigma(x) = \frac{1}{1+exp(-x)}
$$

<img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230228224935533.png" alt="image-20230228224935533" style="zoom:40%;" />

输出的值范围在[0,1]之间。但是`sigmoid`型函数的输出存在**均值不为0**的情况，并且存在**梯度消失的问题**，在深层网络中被其他激活函数替代。在**逻辑回归**中使用的该激活函数用于输出**分类**。

### 求导公式

链式法则

### 梯度消失原因：

$$
\sigma'(x) = \sigma\space \cdot (1-\sigma)
$$

1. sigmoid函数两边的斜率趋向0，很难继续学习
2. sigmoid导数两个部分都小于1，在深层神经网络中，靠前layer参数会因为后面多层sigmoid导数叠加（链式法则）导致更新的特别慢。

### 缺点解决办法

1. 在深层网络中被其他激活函数替代。如`ReLU(x)`、`Leaky ReLU(x)`等
2. 在分类问题中，sigmoid做激活函数时，使用交叉熵损失函数替代均方误差损失函数。
3. 采用正确的权重初始化方法（让初始化的数据尽量不要落在梯度消失区域）
4. 加入BN层（同上，避免数据落入梯度消失区）
5. 分层训练权重

## tanh

$$
tanh(x) = \frac{e^x-e^{(-x)}}{e^x+e^{(-x)}} =\frac{e^{2x}-1}{e^{2x}+1}= 2 \cdot sigmoid(2x)-1
$$

<img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307194241514.png" alt="image-20230307194241514" style="zoom:67%;" />

`tanh(x)`型函数可以解决`sigmoid`型函数的**期望（均值）不为0**的情况。函数输出范围为(-1,+1)。但`tanh(x)`型函数依然存在**梯度消失的问题**。

在LSTM中使用了`tanh(x)`型函数。

## Relu

`ReLU(x)`型函数可以有效避免**梯度消失的问题**，公式如下：

![image-20230228222815687](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230228222815687.png)

<img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307194352199.png" alt="image-20230307194352199" style="zoom:67%;" />

`ReLU(x)`型函数的缺点是**负值成为“死区”**，神经网络无法再对其进行响应。Alex-Net使用了`ReLU(x)`型函数。当我们训练深层神经网络时，最好使用`ReLU(x)`型函数而不是`sigmoid(x)`型函数。

ReLU梯度稳定，值还比sigmoid大，所以**可以加快网络训练**。

但是要注意，我们在输入图像时就要注意，应该使用Min-Max归一化，而不能使用Z-score归一化。（避免进入死区）

### 在0点不可导

人为将梯度规定为0（源码就是这么写的）

## Relu6

Relu的正值输出是[0，无穷大]，但计算机内存优先，所以限定relu最大值为6

<img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307194457525.png" alt="image-20230307194457525" style="zoom:67%;" />

## LeakyRelu

为**负值增加了一个斜率**，缓解了“死区”现象，公式如下：

<img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307194659126.png" alt="image-20230307194659126" style="zoom:67%;" />

![image-20230228222900735](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230228222900735.png)

`Leaky ReLU(x)`型函数缺点是，**超参数a（阿尔法）合适的值不好设定**。当我们想让神经网络能够学到负值信息，那么使用该激活函数。

## P-Relu 参数化Relu

数化ReLU（P-ReLU）。参数化ReLU为了解决超参数a（阿尔法）合适的值不好设定的问题，干脆将这个参数也融入模型的整体训练过程中。也使用误差反向传播和随机梯度下降的方法更新参数。

## R-Relu 随机化Relu

就是超参数a（阿尔法）随机化，**让不同的层自己学习不同的超参数**，但随机化的超参数的分布符合均值分布或高斯分布。

## Mish激活函数

<img src="https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307194824945.png" alt="image-20230307194824945" style="zoom:67%;" />
$$
Mish(x) = x\cdot tanh(log(1+e^x))
$$

在负值中，允许有一定的梯度流入。

## ELU指数化线性单元

也是为了解决死区问题，公式如下：

![image-20230307194918301](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307194918301.png)

![image-20230228224119801](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230228224119801.png)

缺点是**指数计算量大**。

## Maxout

就是用一个MLP层作为激活函数。

![image-20230307195003327](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307195003327.png)

![image-20230307195013184](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307195013184.png)

与常规的激活函数不同，**Maxout**是一个可以学习的**分段线性函数**。其原理是，任何ReLU及其变体等激活函数都可以看成分段的线性函数，而Maxout加入的一层神经元正是一个可以学习参数的分段线性函数。

优点是其拟合能力很强，理论上可以拟合任意的凸函数。缺点是参数量激增！在Network-in-Network中使用的该激活函数。



# Softmax求导

![image-20230307195822249](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307195822249.png)

要结合交叉熵loss函数考虑

$\frac{dL}{dz}=\frac{dL}{da}\cdot \frac{da}{dz}$

假设第j个类别是正确的，$y_j=1$,其它为0

$L = -\sum_{i=1}^ny_iln(a_i)$

$\frac{dL}{da} = -y_iln(a_j)=-ln(a_j)$

![image-20230307200559541](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307200559541.png)

所以最终Loss只跟label类别有关

所以当i=j：

![image-20230307201705230](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307201705230.png)

当i!=j:

![image-20230307201737792](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230307201737792.png)

