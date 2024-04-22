---
title: LLM Basic Knowledge
date: 2024/4/18 20:46:25
categories:
  - [ML, LLM]
---



参考博客：

https://dongnian.icu/note/llm

[Transformer学习笔记一：Positional Encoding（位置编码） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/454482273)





<!-- more -->

# LLM 模型

## 一些概念

### prefix LM 和 causal LM

+ prefix LM：可以看到输入序列的上下文作为条件信息。
+ causal LM：自回归语言模型，只能看到当前和历史输入token序列。





## 主流预训练框架

### 自回归模型

根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词，这种类型的LM被称为自回归语言模型，常用于生成任务。代表作有GPT。

将每个单词$x_n$当作token，计算一个句子存在概率
$$
p(x_1:x_L) = p(x_1)p(x_2|x_1)p(x_3|x_1:x_2)\cdots p(x_L|x_1：x_{L-1})
$$
e.g.
$$
\begin{align*} p({the}, {mouse}, {ate}, {the}, {cheese}) = \, & p({the}) \\ & p({mouse} \mid {the}) \\ & p({ate} \mid {the}, {mouse}) \\ & p({the} \mid {the}, {mouse}, {ate}) \\ & p({cheese} \mid {the}, {mouse}, {ate}, {the}). \end{align*}
$$

自回归语言模型的特点是**它可以利用例如前馈神经网络等方法有效计算出每个条件概率分布**。

### Autoencoding自编码模型

通过某个降噪目标（MLM）训练的双向文本编码器，即是mask掉文本中间某个token，让模型去预测，例如BERT。编码器会产出适用于NLU任务的上下文表示，但无法直接用于文本生成。

### encoder-decoder模型

源自Seq2seq模型，代表作有T5.采用双向注意力机制，常用于条件生成任务，例如文本摘要或机械翻译。



## 经典NLP模型

### N-gram

N-gram假设第N个词的出现只与前面N-1个词相关，而与其它任何词都不相关；形成一个长度为N的滑动窗口计算第N个单词出现概率。
$$
p(x_i|x_{1:i-1}) = p(x_i|x_{i-(n-1):i-1})
$$
对于bigram model：
$$
p(x_i|x_{i-1}) = \frac{C(x_{i-1}x_i)}{C(x_{i-1})}
$$
对于n-gram model：
$$
p(x_i|x_{i-n-1},\cdots x_{i-1}) = \frac{C(x_{i-n-1}\cdots x_{i})}{C(x_{i-n-1}\cdots x_{i-1})}
$$
然后在给定的训练语料中，将上述的条件概率值都统计计算出来即可。

如果n太小，那么模型将无法捕获长距离的依赖关系。然而，如果n太大，统计上将无法得到概率的好估计.

### 神经语言模型

#### RNN

#### LSTM

#### Seq2seq

#### Transformer

##### 基本架构

![image-20240418155205656](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418155205656.png)

##### Attention

核心思想：网络应该更关注输入中的重要部分，而忽略不重要的部分，它通过学习不同部分的权重，将输入的序列中的重要部分显式地加权，从而使得模型可以更好地关注与输出有关的信息。

相较于传统的Seq2Seq模型只使用编码器来捕捉输入序列的信息，而解码器只从编码器的最后状态中获取信息，并将其用于生成输出序列。
Attention机制允许解码器在生成每个输出时，根据输入序列的不同部分给予不同的注意力，从而使得模型更好地关注到输入序列中的重要信息。




Transformer的Attention公式如下：

![image-20240418155227786](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418155227786.png)

+ Q,K,V通过输入经过三个不同的MLP层计算获得，使用三个不同的vector（QKV）可以增强网络表达能力

+ $\sqrt{d_k}$作用：$QK^t$相乘会让数值变大，scaling 后进行softmax可以使得输入的数据的分布变得更好，避免数据进入softmax敏感区间，防止梯度消失，让模型能够更容易训练。

  Attention的计算是在内积之后进行softmax，可以大致认为内积之后、softmax之前的数值在$−3\sqrt{d}$−到$3\sqrt{d}$这个范围内，由于d通常都至少是64，所以$e^{3\sqrt{d}}$比较大而$e^{-3\sqrt{d}}$比较小，因此经过softmax之后，Attention的分布非常接近一个one hot分布了，这带来严重的梯度消失问题，导致训练效果差。

  可以不除以$\sqrt{d_k}$，只要有别的方法可以缓解梯度消失即可。例如T5，初始化q、k全连接层的时候，其初始化方差要多除以一个d。

+ Transformer使用多头注意力机制：增强网络的表达能力。不同的头关注不同信息。

  假设有一个句子"the cat, which is black, sat on the mat"。在处理"sat"这个词时，一个头可能会更注"cat"，因为"cat"是"sat"的主语；另一个头可能会更关注"on the mat"，因为这是"sat"的宾语；还有一个头可能会关注"which is black"，因为这是对"cat"的修饰。

  经过多头之后，我们还需要att_out线性层来做线性变换，以自动决定（通过训练）对每个头的输出赋予多大的权重，从而在最终的输出中强调一些头的信息，而忽视其他头的信息。

+ self-attention中，Q和K在点积之后，需要先经过mask再进行softmax，因此，对于要屏蔽的部分，mask之后的输出需要为负无穷，这样softmax之后输出才为0。

+ transformer使用了权重共享：

  在Transformer中，Encoder和Decoder是由多层的Self-Attention Layer和前馈神经网络层交叉堆叠而成。

  在Encoder中，所有的自注意力层和前馈神经网络层都共享相同的参数。这种共享保证了每一层都执行相同的计算过程，使得模型能够更好地捕捉输入序列的不同位置之间的关联性。

  在Decoder中，除了和Encoder相同的权重共享方式外，还存在另一种特殊的权重共享：Decoder的自注意力层和Encoder的自注意力层之间也进行了共享。通过这种共享方式，Decoder可以利用Encoder的表示来理解输入序列并生成输出序列。

  权重共享的好处是大大减少了模型的参数数量，使得Transformer可以更有效地训练，并且更容易进行推理。此外，共享参数还有助于加快训练速度和提高模型的泛化能力，因为模型可以在不同位置共享并学习通用的特征表示。

  

  

  

##### 位置编码

发展历史：[Transformer学习笔记一：Positional Encoding（位置编码） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/454482273)

![image-20240418215532333](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418215532333.png)

为什么用三角函数：希望位置编码是连续有界

为什么$w_k$很小，避免序列前后端的编码重合

为什么sin和cos交替使用：希望不同位置编码之间能通过线性转换得到

![image-20240418215748852](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418215748852.png)

## 常见大语言模型

常有encoder-only, decoder-only以及encoder-decoder模型

- Encoder：Transformer中的Encoder是用于将输入序列转换成隐藏表示的模块。它将输入序列中的每一个位置的词嵌入向量作为初始输入，然后通过多层的自注意力机制和全连接层，将每个位置的信息编码成一个定长的隐藏向量表示。Encoder的输出可以被送入Decoder中进行下一步处理。
- Decoder：Transformer中的Decoder是用于生成输出序列的模块。它接受Encoder的输出，以及前面已经生成的部分输出序列作为输入。Decoder的主要任务是生成下一个位置的词，直到整个序列生成完成。Decoder同样也是由多层的自注意力机制和全连接层组成，但相比于Encoder还加入了一个额外的注意力机制，用于将Encoder输出的信息融合到生成过程中。Decoder还包括一个线性变换层，用于将Decoder的输出映射成输出词的概率分布。

Encoder和Decoder的区别在于它们的输入和输出以及它们的功能。Encoder的输入是输入序列，输出是每个位置的隐藏向量表示；Decoder的输入是Encoder的输出和前面生成的部分输出序列，输出是生成的下一个位置的词。Encoder用于编码输入信息，Decoder用于生成输出信息。

### GPT系列

![image-20240419170521376](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240419170521376.png)

+ 使用了一个仅有decoder的 Transformer 结构，每一个作为一个Layer，共有12层。
+ 激活函数是 GELU（更平滑减少梯度爆炸的风险 处处可导，不会出现神经元死亡的状态）。

+ 首先以无监督的方式预训练模型，让它接触大量的原始文本数据。这个预训练阶段使模型能够理解自然语言中存在的统计模式和结构。

+ 模型经历了一个监督微调阶段，其中它在具有标签数据的特定任务上得到了进一步的改进。

### BERT系列

![image-20240419172830240](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240419172830240.png)

+ 采用MLM对双向的Transformers进行预训练，以生成深层的双向语言表征。
+ 预训练后，只需要添加一个额外的输出层进行fine-tune，就可以在各种各样的下游任务中取得state-of-the-art的表现。在这过程中并不需要对BERT进行任务特定的结构修改。

#### 结构

BERT是基于transformer的，但是它只使用了transformer的encoder部分，它的整体框架是由多层transformer的encoder堆叠而成的。每一层的encoder则是由一层muti-head-attention和一层feed-forword组成

#### 预训练

+ Masked Language Model（MLM）

  以15%的概率用mask token （[MASK]）随机地对每一个训练序列中的token进行替换，然后预测出[MASK]位置原有的单词。然而，由于[MASK]并不会出现在下游任务的微调（fine-tuning）阶段，因此预训练阶段和微调阶段之间产生了**不匹配***（这里很好解释，就是预训练的目标会令产生的语言表征对[MASK]敏感，但是却对其他token不敏感）*。因此BERT采用了以下策略来解决这个问题：

  首先在每一个训练序列中以15%的概率随机地选中某个token位置用于预测，假如是第i个token被选中，则会被替换成以下三个token之一

  1）80%的时候是[MASK]。如，my dog is **hairy**——>my dog is **[MASK]**

  2）10%的时候是随机的其他token。如，my dog is **hairy**——>my dog is **apple**

  3）10%的时候是原来的token*（保持不变，个人认为是作为2）所对应的负类）*。如，my dog is **hairy**——>my dog is **hairy**

  再用该位置对应的输出向量 去预测出原来的token（*输入到全连接，然后用softmax输出每个token的概率，最后用交叉熵计算loss）*。

  该策略令到BERT不再只对[MASK]敏感，而是对所有的token都敏感，以致能抽取出任何token的表征信息。

+ Next Sentence Prediction（NSP）

  预测两个句子是否连在一起。具体的做法是：对于每一个训练样例，我们在语料库中挑选出句子A和句子B来组成，50%的时候句子B就是句子A的下一句*（标注为IsNext）*，剩下50%的时候句子B是语料库中的随机句子*（标注为NotNext）*。接下来把训练样例输入到BERT模型中，用[CLS]对应的C信息去进行二分类的预测。

+ Masked Language Model（MLM）

  以15%的概率用mask token （[MASK]）随机地对每一个训练序列中的token进行替换，然后预测出[MASK]位置原有的单词。然而，由于[MASK]并不会出现在下游任务的微调（fine-tuning）阶段，因此预训练阶段和微调阶段之间产生了**不匹配***（这里很好解释，就是预训练的目标会令产生的语言表征对[MASK]敏感，但是却对其他token不敏感）*。因此BERT采用了以下策略来解决这个问题：

  首先在每一个训练序列中以15%的概率随机地选中某个token位置用于预测，假如是第i个token被选中，则会被替换成以下三个token之一

  1）80%的时候是[MASK]。如，my dog is **hairy**——>my dog is **[MASK]**

  2）10%的时候是随机的其他token。如，my dog is **hairy**——>my dog is **apple**

  3）10%的时候是原来的token*（保持不变，个人认为是作为2）所对应的负类）*。如，my dog is **hairy**——>my dog is **hairy**

  再用该位置对应的 �� 去预测出原来的token（*输入到全连接，然后用softmax输出每个token的概率，最后用交叉熵计算loss）*。

  该策略令到BERT不再只对[MASK]敏感，而是对所有的token都敏感，以致能抽取出任何token的表征信

![image-20240419181519909](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240419181519909.png)

+ 最终输入
  + Token Embeddings是词向量，第一个单词是CLS（Classification）标志，可以用于之后的分类任务；第一个SEP表示第一个句子的结束，同时标志第二个句子开始。
  + Segment Embeddings用来区别两种句子，因为预训练不光做LM（语言模型）还要做以两个句子为输入的分类任务
  + Position Embeddings表示位置信息，这里的位置embedding是通过学习的方式得到的。

最后训练样例长这样：

Input1=[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]

Label1=IsNext

Input2=[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]

Label2=NotNext

把每一个训练样例输入到BERT中可以相应获得两个任务对应的loss，再把这两个loss加在一起就是整体的预训练loss。*（也就是两个任务**同时**进行训练）*

#### Fine-tuning

![image-20240419182742596](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240419182742596.png)

+ sequence的分类任务：(a)图表示两个句子的分类，如比如判断两句话是否表示相同的含义；(b)是单句子分类，如：比如判断电影评论是喜欢还是讨厌。预训练中的NSP任务使得BERT中的“[CLS]”位置的输出包含了整个句子对（句子）的信息，我们利用其在有标注的数据上微调（Fine-tuning）模型，给出预测结果。所以，这两种情况，只需要在 Transformer 的输出之上加一个分类层。

+ 问答任务：输入部分由问题和包含答案的文本组成，并有特殊分隔符“【SEP】”分隔。因为答案由文本中连续的token组成，所以**预测答案的过程本质上是确定答案开头和结尾token所在的位置的过程**。e.g.：输入：身体内哪些元素缺失容易导致抽筋[sep]小腿肚抽筋通常是由于钙流失导致骨质疏松引起的；输出：钙（输出token位置）

+ NER（实体命名识别）：给出一句话，对每个词进行标注，判断属于人名，地名，机构名，还是其他。

  + BERT模型+FC layer（全连接层）：

    BERT的output 是每个token的encoding vector。只需要在BERT的基础上增加一层全连接层，一般情况下，在NER任务中，全连接层(经过softmax)的输出为4个维度，分别作为每一类的概率。（在NER任务中一般有4类：B表示实体的开始，I表示实体的中间，E表示实体的结束，O表示不是实体）。

  + BERT+CRF 模型

    在BERT后连接一个CRF层，CRF是一种经典的概率图模型，CRF层可以加入一些约束来保证最终的预测结果是有效的。这些约束可以在训练数据时被CRF层自动学习得到。

#### 小知识点

+ Bert在原输入前加入[cls]用于学习整个句子的表示

+ warm-up：将学习率逐渐从一个较小的初始值增加到预定的最大学习率，解决两个问题：

  **不稳定性：**在训练初期，由于模型参数的随机初始化以及模型的复杂性，模型可能处于一个较不稳定的状态。此时使用较大的学习率可能导致模型的参数变动太大，使得模型很难收敛，学习率warm-up可以在这个阶段将学习率保持较小，提高模型训练的稳定性。
  **避免过拟合：**BERT模型往往需要较长的训练时间来获得高质量的表示。如果在训练的早期阶段就使用较大的学习率，可能会导致模型在训练初期就过度拟合训练数据，降低模型的泛化能力。通过学习率warm-up，在训练初期使用较小的学习率，可以避免过度拟合，等模型逐渐稳定后再使用较大的学习率进行更快的收敛。

+ 

### XLNet

To be continue...

### RoBERTa

To be continue...

### T5

To be continue...

### LLama

#### LLama

LLaMA 所采用的 Transformer 结构和细节，与标准的 Transformer 架构不同的地方包括采用了**前置层归一化**（Pre-normalization）并使用 **RMSNorm 归一化函数** （Normalizing Function）、激活函数更换为 **SwiGLU**，并使用了**旋转位置嵌入**（RoP），整体 Transformer 架构与 GPT-2 类似。

##### RMSNorm归一化

![image-20240418160600375](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418160600375.png)

为了使得模型训练过程更加稳定，GPT-2 相较于 GPT 就引入了前置层归一化方法，将第一个层归一化移动到多头自注意力层之前，第二个层归一化也移动到了全连接层之前，同时残差连接的位置也调整到了多头自注意力层与全连接层之后。层归一化中也采用了 RMSNorm 归一化函数。 针对输入向量 aRMSNorm 函数计算公式如下

$$
RMS(a)=\sqrt{\frac{1}{n}\sum_{i=1}^na_i^2}
$$

$$
\overline{a_i}=\frac{a_i}{RMS(a)}
$$

此外，RMSNorm 还可以引入可学习的缩放因子$g_i$和偏移参数$b_i$，从而得到
$$
\overline{a_i}=\frac{a_i}{RMS(a)}g_i + b_i
$$

##### SwiGLU激活函数

$$
Swish_{\beta}(x) = x\sigma(\beta x)
$$

![image-20240418200505781](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418200505781.png)



Transformer中FFN(Feed Forwar d)层包括两层全连接，第一层升维，第二层降维回归到输入维度，中间插入一个非线性激活函数ReLU。
$$
FFN_{ReLU}(x,W_1,W_2)=ReLU(xW_1)W_2
$$
![image-20240418200241972](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418200241972.png)

SwiGLU激活函数是相较于 ReLU 函数在大部分评测中都有不少提升。在 LLaMA 中全连接层 使用带有 SwiGLU 激活函数的 FFN（Position-wise Feed-Forward Network）的计算公式如下：
$$
FFN_{SwiGLU}(x,W_1,V,W_2)=SwiGLU(x,W,V)W_2
$$

$$
SwiGLU(x,W,V) = Swish_{\beta}(xW)\otimes xV
$$

$$
Swish_{\beta}(x) = x\sigma(\beta x)
$$

![image-20240418200302617](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418200302617.png)

llama是把SwiGLU中的W，V，W2的矩阵维度从(dim， dim)变成(dim, 2/3dim)，从而打平参数量和计算量。

![image-20240418200358968](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418200358968.png)

##### 旋转位置嵌入（RoPE）

[十分钟读懂旋转编码（RoPE） (zhihu.com)](https://www.zhihu.com/tardis/zm/art/647109286?source_id=1005#:~:text=旋转位置编码（Rotary Position Embedding，RoPE）是论文 Roformer%3A Enhanced Transformer With Rotray,提出的一种能够将相对位置信息依赖集成到 self-attention 中并提升 transformer 架构性能的位置编码方式。 而目前很火的 LLaMA、GLM 模型也是采用该位置编码方式。)

对于位置编码，常规的做法是在计算query,key和value向量之前，会计算一个位置编码向量$p_i\in R_d$加到词嵌入$x_i\in R_d$上，然后再乘对应的变换矩阵转换为$q,k,v$

经典的$p_i$计算方法是Sinusoidal 函数，k是第k个token,维度是d，i=2tor2t+1是位置向量里第i个元素：
$$
p_{i,2t} = sin(\frac{k}{10000^{2t/d}})
$$

$$
p_{i,2t+1} = cos(\frac{k}{10000^{2t/d}})
$$

RoPE为了能利用上 token 之间的相对位置信息，假定 query 向量$q_m$ 和 key 向量$k_n$间的内积操作可以被一个函数$g$表示，该函数$g$的输入是词嵌入向量$x_m,x_n$ 和它们之间的相对位置m-n ：
$$
<f_q(x_m,m),f_k(x_n,n)>=g(x_m,x_n,m-n)
$$
接下来的目标就是找到一个等价的位置编码方式，从而使得上述关系成立,在二维：

![image-20240418221338409](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418221338409.png)

![image-20240418221353501](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418221353501.png)

扩展到任意维度

![image-20240418221653937](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418221653937.png)

![image-20240418221835955](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418221835955.png)

#### Llama-2

模型结构的变动主要是体现在GQA和FFN缩放上

+ MHA改成GQA：整体参数量会有减少
+ FFN模块矩阵维度有扩充：增强泛化能力，整体参数量增加
+ 上下文长度是llama两倍(长度从2048->4096) 训练语料增加约 40%，体现在1.4T->2.0T的Tokens llama2-34B和llama2-70B使用了GQA，加速模型训练和推理速度

##### MQA和GQA

+ Mutil-Head Attention 因为自回归模型生成回答时，需要前面生成的KV缓存起来，来加速计算。
+ Multi-Query Attention 多个头之间可以共享KV对，因此速度上非常有优势，实验验证大约减少30-40%吞吐。
+ Group Query Attention 没有像MQA那么极端，将query分组，组内共享KV，效果接近MQA，速度上与MQA可比较。

![image-20240418222254764](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240418222254764.png)

### ChatGLM系列

#### ChatGLM

GLM希望通过多任务学习将不同框架目标结合

GLM模型基于autoregressive blank infilling方法，结合了上述三种预训练模型的思想。

+ 自编码思想：在输入文本中，随机删除连续的tokens。
+ 自回归思想：顺序重建连续tokens。在使用自回归方式预测缺失tokens时，模型既可以访问corrupted文本，又可以访问之前已经被预测的spans。
+ span shuffling + 二维位置编码技术。
+ 通过改变缺失spans的数量和长度，自回归空格填充目标可以为条件生成以及无条件生成任务预训练语言模型。

##### 自回归空格填充任务

[清华大学通用预训练模型：GLM (zhihu.com)](https://www.zhihu.com/tardis/zm/art/637382548?source_id=1005)

给定一个输入文本$x=[x_1,\cdots , x_n]$,从中随机取样多个文本片段$\{s_1,s_2,\cdots, s_m\}$构成span，对应x中一系列连续的词，每个片段用一个单独的[mask]替换，这样原文本x将变成一个损坏文本,e.g.$x_{corrup}=[x_1,[mask], x_4,\cdots, [mask]]$。模型以自回归的方式从损坏的文本中预测缺失的词,这意味着在预测一个片段中的缺失词时，模型可以访问损坏的文本和**之前预测的片段**。

![image-20240419145518680](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240419145518680.png)

+ 原始文本$x$被分成两部分：Part A是损坏文本$x_{corrup}$,Part B是被mask的span，这里假设mask掉$[x_3]$和$[x_5,x_6]$，跨度长度付出泊松分布($\lambda =3$)。Part A 的词可以相互看到，但不能看到 Part B 中的任何词。Part B 的词可以看到 Part A 和 Part B 中的前置词（前面预测的词），但不能看到 Part B 中的后续词。
+ mask掉$[x_3]$和$[x_5,x_6]$， 并打乱 Part B 的顺序。为了捕捉span之间的内在联系，随机交换span的顺序。
+ GLM 自回归地生成 Part B。 每个片段在输入时前面加上 [Start]，在输出时后面加上 [End]。 二维位置编码表示不同片段之间和片段内部的位置关系。
+ **自注意力掩码**。 **灰色区域被掩盖**。 **Part A 的词语可以自我看到（图2(d)蓝色框），但不能看到 Part B。 Part B 的词语可以看到 Part A 和 Part B 中的前面的词语（图2(d)黄色和绿色框对应两个片段）**。 [M] := [MASK]，[S] := [START]，[E] := [END]。

##### 多目标预训练

GLM 遮盖了短的文本区域，适合于 NLU 任务。我们更感兴趣的是预训练一个能够同时处理 NLU 和文本生成的单一模型。因此，清华大学研究了一种多任务预训练设置，其中一个生成更长文本的目标与空白填充目标共同优化。GLM 考虑了以下两种目标：

- 文档级别。采样一个单一的区域，其长度从原始长度的 50% 到1 00% 之间的均匀分布中采样。该目标旨在进行长文本生成。
- 句子级别。限制遮盖的区域必须是完整的句子。多个区域（句子）被采样，覆盖原始文本的 15% 的词数。该目标旨在进行 seq2seq 任务，其预测结果通常是完整的句子或段落。

这两种新的目标都是按照原始目标的相同方式定义的，即公式1。唯一的区别是区域的数量和区域的长度。

##### 模型架构

GLM 使用了一个单一的 Transformer，并对架构做了一些修改：

（1）重新排列了层归一化和残差连接的顺序，这对于大规模的语言模型来避免数值错误是非常关键。

（2）使用了一个单一的线性层来进行输出词的预测。

（3）用 GeLUs 替换了 ReLU 激活函数。

##### 二维位置编码

图中 Position1 = [1, 2, 3, 4, 5, 5, 5, 5, 3, 3]，Position2 = [0, 0, 0, 0, 0, 1, 2, 3, 1, 2] 是怎么得到的。Position1 和 Position2 是输入的二维编码，第一个维度表示片段在原始文本中的相对位置，第二个维度表示片段内部的相对位置。

GLM 的编码方法确保了模型在重建被遮盖的跨度时不知道它们的长度。这与其他模型相比是一个重要的区别。

注意Position2耶不会知道跨度长度，因为token是一个一个预测的，position encoding不断加1直到预测到[end]。

##### 多任务

+ NLU

GLM 将 NLU 分类任务重新制定为填空生成任务，例如，情感分类任务可以表述为 “{SENTENCE}。这真的是 [MASK]”。输出label y也同样会被映射到完形填空的答案中。“positive” 和 “negative” 对应的标签就是“good” 和 “bad。因此，句子是正面或负面的概率与在空白处预测“好”或“坏”成正比。然后我们用交叉熵损失来微调 GLM。

![image-20240419153829264](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240419153829264.png)

+ NLG

给定的上下文构成了输入的 Part A，末尾附加了一个 mask 符号。模型自回归地生成 Part B 的文本。可以直接应用预训练的 GLM 进行无条件的生成，或者在下游的条件生成任务上对其进行微调。

![image-20240419153920299](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240419153920299.png)

#### ChatGLM-2

+ 更长的上下文：基于 FlashAttention 技术，将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练。对于更长的上下文，发布了 ChatGLM2-6B-32K 模型。LongBench 的测评结果表明，在等量级的开源模型中，ChatGLM2-6B-32K 有着较为明显的竞争优势。
+ 更强大的性能：基于 ChatGLM 初代模型的开发经验，全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。 
+ 更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。 
  更开放的协议：ChatGLM2-6B 权重对学术研究完全开放，在填写问卷进行登记后亦允许免费商业使用。

#####  与ChatGLM的变化

+ 使用了RoPE替换二维位置编码。目前大部分主流的LLMs都在使用RoPE，
+ Multi-Query Attention：这是一种共享机制的Attention，相比Multi-Head Attention，其Query部分没有区别，Key和Value可以只用一个Head。计算时，对Key和Value进行expand或者repeat操作，使它们填充到与Query一样的维度，后续计算就与Multi-Head Attention没区别。
+ Attention Mask: V1的attention mask分了2部分，Part A和Part B，Part A部分是双向Attention，Part B部分是Causal Attention。在V2版本，全部换成了Causal Attention，不再区分是Part A还是Part B，完全变成了decoder-only的架构。
+ 多目标任务：Chat版本主要还是用的gMask生成式任务，但是在V1版本的代码还能看到mask、gMask等字样，V2已经摒弃了这些特殊token，原因与Attention Mask一致，均因为变成了decoder-only的架构，不再需要区分Part A和Part B。
  3.ChatGLM-3

#### ChatGLM-3

ChatGLM2与ChatGLM3模型架构是完全一致的。
词表的大小从ChatGLM的150528缩小为65024
位置编码从每个GLMBlock一份提升为全局一份
SelfAttention之后的前馈网络有不同。ChatGLM用GELU（Gaussian Error Linear Unit）做激活；ChatGLM用Swish-1做激活。而且ChatGLM2、3应该是修正了之前的一个bug，因为GLU（Gated Linear Unit）本质上一半的入参是用来做门控制的，不需要输出到下层，所以ChatGLM2、3看起来前后维度不一致（27392->13696)反而是正确的。

# 有监督微调SFT

流程：

+ 预训练模型选择：选择一个在大规模数据上进行预训练的模型作为基础模型。例如，可以选择一种预训练的语言模型，如BERT、GPT等。
+ 数据准备：准备用于微调的特定任务数据集。这些数据集应包含任务相关的样本和相应的标签或目标。确保数据集与任务的特定领域或问题相关。
+ 构建任务特定的模型头：根据任务的要求，构建一个特定的模型头（task-specific head）。模型头是添加到预训练模型之上的额外层或结构，用于根据任务要求进行输出预测或分类。例如，对于文本分类任务，可以添加一个全连接层和softmax激活函数。
+ 参数初始化：将预训练模型的参数作为初始参数加载到微调模型中。这些参数可以被视为模型已经学习到的通用语言表示。
+ 微调训练：使用特定任务的数据集对模型进行有监督训练。这包括将任务数据输入到模型中，计算损失函数，并通过反向传播和优化算法（如梯度下降）更新模型参数。在微调过程中，只有模型头的参数会被更新，而预训练模型的参数会保持不变。
+ 调整超参数：微调过程中，可以根据需要调整学习率、批量大小、训练迭代次数等超参数，以达到更好的性能。
+ 评估和验证：在微调完成后，使用验证集或测试集对微调模型进行评估，以评估其在特定任务上的性能。可以使用各种指标，如准确率、精确率、召回率等。
  可选的后续微调：根据实际情况，可以选择在特定任务的数据上进行进一步的微调迭代，以进一步提高模型性能。

## PEFT（Parameter-Efficient Fine-Tuning）

PEFT旨在仅训练少量参数使模型适应到下游任务，通过冻结预训练模型的某些层，并仅微调特定于下游任务的最后几层来实现这种效率。即可节省计算资源，又只修改模型参数的一小部分，并且不容易过度拟合。高效微调技术可以粗略分为以下三大类：

+ 增加额外参数：

  + 适配器（Adapters）：适配器层是插入预训练模型层之间的小型神经网络。在微调过程中，只训练这些适配器层，保持预先训练的参数冻结

  + 软提示：固定模型权重并更新提示的参数，生成的提示被称为“软提示”，e.g.

    对于给定的: `What's 2+2?`.

    1. 它可能被标记为 `What, 's, 2, +, 2, ?`.
    2. 然后，每个标记将被转换为一组值的向量。
    3. 这些向量可以视为模型参数。模型可以进一步训练，仅调整这些提示的权重。一旦我们开始更新这些权重，标记的向量就不再对应于词汇表中实际的嵌入。

+ 选取一部分参数更新

  + 选择性层调整（Selective Layer Tuning）：可以只微调层的一个子集，而不是微调模型的所有层。
  + 稀疏微调（Sparse Fine-Tuning）：传统的微调会略微调整所有参数，但稀疏微调只涉及更改模型参数的一个子集。

+ 引入重参数化

## Prefix Tuning

Prefix Tuning提出固定预训练LM，为LM添加可训练，任务特定的前缀，这样就可以为不同任务保存不同的前缀，微调成本也小；同时，这种Prefix实际就是连续可微的Virtual Token（Soft Prompt/Continuous Prompt），相比离散的Token，更好优化，效果更好。

![image-20240421142012294](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421142012294.png)



在输入token之前构造一段任务相关的virtual tokens作为Prefix，然后训练的时候只更新Prefix部分的参数，而PLM中的其他部分参数固定。针对不同的模型结构，需要构造不同的Prefix。

+ 针对自回归架构模型：在句子前面添加前缀，得到 z = [PREFIX; x; y]，合适的上文能够在固定 LM 的情况下去引导生成下文（比如：GPT3的上下文学习）。
+ 针对编码器-解码器架构模型：Encoder和Decoder都增加了前缀，得到 z = [PREFIX; x; PREFIX0; y]。Encoder端增加前缀是为了引导输入部分的编码，Decoder 端增加前缀是为了引导后续token的生成。

![image-20240421142326432](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421142326432.png)

该方法其实和构造Prompt类似，只是Prompt是人为构造的“显式”的提示，并且无法更新参数，而Prefix则是可以学习的“隐式”的提示。


![image-20240421142410640](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421142410640.png)

同时，为了防止直接更新Prefix的参数导致训练不稳定和性能下降的情况，在Prefix层前面加了MLP结构，训练完成后，只保留Prefix的参数。

![image-20240421142649411](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421142649411.png)

除此之外，通过消融实验证实，只调整embedding层的表现力不够，将导致性能显著下降，因此，在每层都加了prompt的参数，改动较大。

![image-20240421142704694](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421142704694.png)

另外，实验还对比了位置对于生成效果的影响，Prefix-tuning也是要略优于Infix-tuning的。其中，Prefix-tuning形式为 [PREFIX; x; y]，Infix-tuning形式为 [x; INFIX; y]。

## Prompt Tuning

作者提出了Prompt Tuning，通过反向传播更新参数来学习prompts，而不是人工设计prompts；同时冻结模型原始权重，只训练prompts参数。

![image-20240421143331005](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421143331005.png)

该方法可以看作是Prefix Tuning的简化版本，它给每个任务定义了自己的Prompt，然后拼接到数据上作为输入，但只在输入层加入prompt tokens，并且不需要加入 MLP 进行调整来解决难训练的问题。

Prompt Tuning 还提出了 Prompt Ensembling，也就是在一个批次（Batch）里同时训练同一个任务的不同 prompt（即采用多种不同方式询问同一个问题），这样相当于训练了不同模型，比模型集成的成本小多了。

## P-Tuning

该方法将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。

相比Prefix Tuning，P-Tuning加入的可微的virtual token，但仅限于输入层，没有在每一层都加；另外，virtual token的位置也不一定是前缀，插入的位置是可选的。这里的出发点实际是把传统人工设计模版中的真实token替换成可微的virtual token。


![image-20240421143637201](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421143637201.png)

经过预训练的LM的词嵌入已经变得高度离散，如果随机初始化virtual token，容易优化到局部最优值，而这些virtual token理论是应该有相关关联的。因此，作者通过实验发现用一个prompt encoder来编码会收敛更快，效果更好。即用一个LSTM+MLP去编码这些virtual token以后，再输入到模型。

## P-Tuning v2

之前的Prompt Tuning和P-Tuning等方法存在两个主要的问题：

+ 缺乏模型参数规模和任务通用性。
  + 缺乏规模通用性：Prompt Tuning论文中表明当模型规模超过100亿个参数时，提示优化可以与全量微调相媲美。但是对于那些较小的模型（从100M到1B），提示优化和全量微调的表现有很大差异，这大大限制了提示优化的适用性。
  + 缺乏任务普遍性：尽管Prompt Tuning和P-tuning在一些 NLU 基准测试中表现出优势，但提示调优对硬序列标记任务（即序列标注）的有效性尚未得到验证。
+ 缺少深度提示优化，在Prompt Tuning和P-tuning中，连续提示只被插入transformer第一层的输入embedding序列中，在接下来的transformer层中，插入连续提示的位置的embedding是由之前的transformer层计算出来的，这可能导致两个可能的优化挑战。
  + 由于序列长度的限制，可调参数的数量是有限的。
  + 输入embedding对模型预测只有相对间接的影响。

考虑到这些问题，作者提出了Ptuning v2，它利用深度提示优化（如：Prefix Tuning），对Prompt Tuning和P-Tuning进行改进，作为一个跨规模和NLU任务的通用解决方案。

该方法在每一层都加入了Prompts tokens作为输入，而不是仅仅加在输入层。

![image-20240421144227395](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421144227395.png)

具体做法基本同Prefix Tuning，可以看作是将文本生成的Prefix Tuning技术适配到NLU任务中，然后做了一些改进：

+ 移除重参数化的编码器。以前的方法利用重参数化功能来提高训练速度和鲁棒性（如：Prefix Tuning中的MLP、P-Tuning中的LSTM））。在 P-tuning v2 中，作者发现重参数化的改进很小，尤其是对于较小的模型，同时还会影响模型的表现。
+ 针对不同任务采用不同的提示长度。提示长度在提示优化方法的超参数搜索中起着核心作用。在实验中，我们发现不同的理解任务通常用不同的提示长度来实现其最佳性能，这与Prefix-Tuning中的发现一致，不同的文本生成任务可能有不同的最佳提示长度。
+ 引入多任务学习（一种通过共享模型参数来学习多个闲逛任务的方法，利用不同任务之间的知识提高泛化能力。多任务学习的损失函数是每个人物的损失函数加权求和）。先在多任务的Prompt上进行预训练，然后再适配下游任务。多任务学习对我们的方法来说是可选的，但可能是相当有帮助的。一方面，连续提示的随机惯性给优化带来了困难，这可以通过更多的训练数据或与任务相关的无监督预训练来缓解；另一方面，连续提示是跨任务和数据集的特定任务知识的完美载体。我们的实验表明，在一些困难的序列任务中，多任务学习可以作为P-tuning v2的有益补充。
+ 回归传统的分类标签范式，而不是映射器。标签词映射器（Label Word Verbalizer）一直是提示优化的核心组成部分，它将one-hot类标签变成有意义的词（例如positive或者negative），以利用预训练语言模型头。尽管它在few-shot设置中具有潜在的必要性，但在全数据监督设置中，Verbalizer并不是必须的。它阻碍了提示调优在我们需要无实际意义的标签和句子嵌入的场景中的应用。因此，P-Tuning v2回归传统的CLS标签分类范式，采用随机初始化的分类头（Classification Head）应用于tokens之上，以增强通用性，可以适配到序列标注任务。

## LoRA

文的作者认为权重更新的那部分参数矩阵尽管随机投影到较小的子空间，仍然可以有效的学习，可以理解为针对特定的下游任务这些权重矩阵就不要求满秩。

该方法的核心思想就是通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练。

在涉及到矩阵相乘的模块，在原始的PLM旁边增加一个新的通路，通过前后两个矩阵A,B相乘，第一个矩阵A负责降维，第二个矩阵B负责升维，中间层维度为r，从而来模拟所谓的本征秩（intrinsic rank）。

可训练层维度和预训练模型层维度一致为d，先将维度d通过全连接层降维至r，再从r通过全连接层映射回d维度，其中，r<<d，r是矩阵的秩，这样矩阵计算就从d x d变为d x r + r x d，参数量减少很多。

![image-20240421145635079](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20240421145635079.png)
