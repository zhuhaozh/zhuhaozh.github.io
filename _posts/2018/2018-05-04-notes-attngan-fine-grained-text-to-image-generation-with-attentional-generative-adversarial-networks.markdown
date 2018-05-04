---
layout: post
title: >-
  论文阅读笔记:AttnGAN Fine-Grained Text to Image Generation with Attentional
  Generative Adversarial Networks
date: '2018-05-04 18:27'
categories: paper-read
tags: cvpr2018 gans
mathjax: true
author: zh
---
* contents
{:toc}

## 研究背景

从自然语言来生成图片在很多应用领域的一个基础性问题，这也是这些年来的一个研究热点。目前也有不少文字到图片的方法，它们大多数都是基于GANs的。一个最普遍的方法是将整个文本编码成全局句子向量(global sentence vector)，并将它作为GANs的条件来生成图片。但是由于GANs仅仅依靠全局句子向量的就很难在单词的级别上生成更精细的图片，即图片很难反应具体单词的含义且清晰度不够。





![demo](/images/2018/05/demo.png)
## 解决思路
为了解决上述提到的问题，作者使用注意力机制，来加强生成器对整个句子的具体单词的理解，使用多级加强来生成更加细粒度、清晰的图片。作者将这个模型称之为AttnGAN。具体的结构如图。

![attn-gan-architecture](/images/2018/05/attn-gan-architecture.png)

**1. Attention Generation Network**
注意力机制是用来根据单词生成不同的最相关子区域。也就是说，我们不仅仅要将自然语言编码成一个全局句子向量，还需要将其编码一单词向量(word vector)。生成网络**第一步**使用全局句子向量来生成一个低分辨率的图片。**之后的阶段** ，图片向量在每一个子区域使用注意力层查询单词向量，以此生成单词语境向量(word context vector)。之后将区域图片和它对应的单词语境向量结合起来，生成一个多模态语境向量(multimodal context vector)。这样的网络将能够高效的生成细节丰富的高分辨率图片。

**2. Deep Attentional
Multimodal Similarity Model (DAMSM)**
DAMSM是为了计算生成图片与使用全局句子向量和细粒度的单词信息之间的相似性，使用该方法来作为图片-文本的匹配损失(matching loss)。具体的细节见下

## 实现细节
### Attention Generation Network
依据前面的说明，假设该生成网络有m个生成器($G_0,G_1,...,G_{m-1}$)，分别使用不同的隐藏状态($h_0,h_1,...,h_{m-1}$)作为输入，且生成的图片规模从小到大($\hat{x}_0,\hat{x}_1,...,\hat{x}_{m-1}$)。
具体的公式为：

> $$h_0 =F_0(z,F^{ca}(\overline e))$$
$$h_i =F_i(h_{i-1},F^{attn}_i(e,h_{i-1}))     ,i = 1,2,...,m-1$$
$$\hat x_{i} = G_i(h_i)$$

 其中：

>- $z$为噪声向量（通常从正态分布中采集）
- $\overline e$为全局句子向量
- $e$为词向量的矩阵
- $F^{ca}$作为将语句向量$\overline e$转换为条件向量的 _条件增强器_
- $F_i^{attn}$是之前提到的第$i$阶段的注意力模型

$F^{attn}(e, h)$有两个输入：
1. 词特征$e \in \mathbb{R}^{D×N}$(word features)
2. 前一层传来的图片特征$h \in \mathbb{R}^{\hat{D}×N}$

首先通过增加一层感知层将词特征转换为图片特征的 _普通语义空间_。i.e. $e^{\prime}=Ue$其中$U \in \mathbb{R}^{\hat D×D}$
。

接着再基于输入的$h$，从单词-上下文向量(word-context vector)来计算每一个子区域的图片。$h$的每一列是一个子区域图片的特征向量。对$j^{th}$的子区域来说，它的单词上下文向量是有关$h_j$的**单词向量**的动态代表。这个单词向量可以通过下式计算：

$$c_j = \sum_{i=0}^{T-1}\beta_{j,i} e^\prime, $$
其中$$\beta_{j,i} = \frac{e^{s_{j,i}^\prime}}{\sum_{k=0}^{T-1}e^{s_{j,k}^\prime}}$$
$$s^\prime_{j,i} = h_j^Te_i^\prime$$

$\beta{j,i}$代表当生成第j个子区域的图片时，第i个单词对其的权重。$F^{attn}(e,h)=(c_0,c_1,...,c_{N-1}) \in \mathbb R^{\hat D × N}$

最后，图片特征和与之对应的单词-上下文向量被结合在一起在下一阶段来生成图片。最终的目标方程定义为：$$\mathcal{L} = \mathcal{L}_G + \lambda \mathcal{L}_{DAMSM}$$
$$\mathcal{L}_{G} = \sum_{i=0}^{m-1}\mathcal{L}_{G_{i}}$$
{% raw %}

$\lambda$是一个超参数，用来平衡上述方程的左右两项。
第一项是GAN loss，它包括了条件和无条件分布，AttnGAN的第i阶段，生成器$G_i$有着一个与之对应的$D_i$，无条件损失用来决定一个图片是真实的还是生成的，条件损失用来判断一个图片与对应的句子是否匹配
这部分的对抗损失定义为：

$$\mathcal{L}_{G_{i}} = \underbrace {-\frac{1}{2}\mathbb{E}_{\hat{x}_i \in P_{G_i}}[log(D_i(\hat{x}_i))]}_{无条件损失} - \underbrace{\frac{1}{2}\mathbb{E}_{\hat{x}_i \in P_{G_i}}[log(D_i(\hat{x}_i,\overline e))]}_{条件损失}$$

同时$D_i$也有对应的交叉熵损失：

$$\mathcal{L}_{G_{i}} = \underbrace {-\frac{1}{2}\mathbb{E}_{{x}_i \in P_{G_i}}[log(D_i({x}_i))] - \frac{1}{2}[log(1 - D_i(\hat{x}_i))]}_{无条件损失} \\- \underbrace{\frac{1}{2}\mathbb{E}_{{x}_i \in P_{G_i}}[log(D_i({x}_i,\overline e))]- \frac{1}{2}[1 - log(D_i(\hat{x}_i,\overline e))] }_{条件损失}$$

$x_i$来自真实图片分布$p_{data_i}$，$\hat{x_i}$来自模型分布$p_{G_i}$
{% endraw %}


### Deep Attentional Multimadal Similarity Model

DAMSM学习两个神经网络，用于将子区域的图片和句子中的单词都映射到一个普通语义空间。这样就可以来判别图片和文本之间的相似性了

- **文本编码器(Text encoder)**

这是用一个双向的LSTM实现，它从文本的描述中抽取语义向量。在双向LSTM中，每个单词都对应两个隐藏状态(hidden state)，分别用于对应两个不同的方向，因此需要将这两个隐藏状态连接来代表一个单词的语义。所有单词的特征矩阵由$e \in \mathbb R ^{D×T}$表示，$e$的第i列$e_i$代表第i个单词的特征矩阵。其中D是单词矩阵的维度，T是单词的数量。最后的隐藏状态连接起来作为全局句子向量，由$\overline e \in \mathbb R^{D}$表示。

- **图像编码器(Image encoder)**

这由CNN实现，用CNN将图片映射成一个语义向量。CNN的中间层学习一个图像不同子区域的局部特征，后面的层次学习的是图片全局特征。作者使用的是在ImageNet上预训练的Inception-v3模型。首先将图片转为299*299像素，然后从Inception-v3的"$mixed\_6e$"抽取局部特征矩阵$f \in \mathbb R^{768×289}$(从 768*17*17转换而来)。$f$的的每一列是一个图片子区域的特征向量。768是局部特征向量的维度，而289是一个图片子区域的大小。全局特征向量$f \in \mathbb R^{2048}$是从最后的Inception-v3 的average pooling layer抽取而来。最后再通过增加一个感知层(公式见下)，将图片特征转换为一个普通语义空间。
  $$v = Wf，\overline v = \overline W \overline f$$
其中$v \in \mathbb R ^{D×289}$。它的第i列$v_i$是图片的第i个子区域的特征向量，$\overline v \in \mathbb R^D$是整个图像的全局向量。$D$是多模态特征空间的维度(i.e., 图片和文本的模式(modalities))。

为了效率，作者直接使用了Inception-v3的所有参数。新添加层的参数和剩余其他的网络一起训练。

- **注意力驱动的图片-文本匹配分数(Attention-driven image-text matching score)**

这个设计是用来测量基于注意力模型的图片-句子对之间的匹配程度。
首先计算的是所有可能的单词/图片子区域的相似矩阵：
  $$s=e^Tv$$
其中$s \in \mathbb R^{T×289}$，$s_{i,j}$是第i个单词和第j个图片子区域的点乘，并可以将其规则化：
  $$\overline s_{i,j} = \frac{exp(s_{i,j})}{\sum_{k=0}^{T-1}exp(s_{k,j})}$$

然后给每一个单词(query)构建一个注意力模型来区域-上下文向量(region-context vector)。区域-上下文向量$c_i$动态的代表和第i个单词相联系的图片的子区域。它带权重的将所有区域可视化向量相加。
  $$c_i = \sum_{j=0}^{288}\alpha_jv_j$$
  $$\alpha_j = \frac{exp(\gamma_1\overline{s}_{i,j})}{\sum{k=0}{288}exp(\gamma_1\overline{s}_{i,k})}$$

其中，$\gamma_1$是当计算一个单词的区域-上下文向量时，用来决定要给一个区域的特征多大的注意力

最后，作者定义了第i个单词和图片之间的相关性为$c_i$和$e_i$之间的余弦相似度。$R(c_i,e_i) = \frac {(c_i^T e_i)}{||c_i|| * ||e_i||}$。
整个图片上的图片/文字匹配得分定义为：

  $$R(Q,D) = log(\sum_{i=1}^{T-1}exp(\lambda_2R(c_i,e_i)))^{\frac{1}{\lambda_2}} \tag{10}$$
其中，$\lambda_2$决定要给最相关的那个word-to-region-context对增加多大的重要性。当$\lambda_2\rightarrow \infty$时，R(Q,D)大约为$max_{i=1}^{T-1}R(c_i,e_i)$

- **DAMSM loss**

这个设计的是用来在半监督的方式下学习注意力模型，在这种情况下监督在整个图片和整个句子的匹配。一批图片/句子对$\{(Q_i,D_i)\}_{i=1}^M$，句子${D_i}$与对应匹配的$Q_i$的后验概率的计算为：
$$P(D_i|Q_i) = \frac{exp(\lambda_3R(Q_i,D_i))}{\sum_{j=1}^Mexp(\lambda_3R(Q_i,D_j))} \tag{11}$$

其中$\lambda_3$是一个由实验来决定的平衡因素

并定义一个损失方程作为负log后验概率，表示描述与相匹配的图片之间的损失。
$$\mathcal{L}_1^w = -\sum_{i=1}^MlogP(D_i|Q_i) \tag{12}$$
对称的，也同样要最小化：
$$\mathcal{L}_2^w = -\sum_{i=1}^MlogP(Q_i|D_i) \tag{13}$$
$$P(Q_i|D_i) = \frac{exp(\lambda_3R(Q_i,D_i))}{\sum_{j=1}^Mexp(\lambda_3R(Q_j,D_i))} $$
如果用$R(Q,D) = (\frac{\overline{v}^T \overline{e}}{||\overline v|| * ||\overline e||})$来代替eq(11),(12),(13)中的式子，那么将能够得到损失方程$\mathcal{L}_1^s$，$\mathcal{L}_2^s$ (s代表"sentence"，$\overline{e}代表句子向量，$\overline v$代表全局图片向量)

最后的DAMSM loss就被定义为：
$$\mathcal{L}_{DAMSM} = \mathcal{L}_{1}^{w} + \mathcal{L}_{2}^{w}+\mathcal{L}_{1}^{s}+\mathcal{L}_{2}^{s}$$


基于实验的测试，作者将超参数设置为$\lambda_1=5,\lambda_2=5,\lambda_3=10,M=50$


## 实验
![experiment_1](/images/2018/05/experiment-1.png)
![experiment_2](/images/2018/05/experiment-2.png)
![experiment-3](/images/2018/05/experiment-3.png)
![experiment-4](/images/2018/05/experiment-4.png)
