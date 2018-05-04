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

## 研究背景

从自然语言来生成图片在很多应用领域的一个基础性问题，这也是这些年来的一个研究热点。目前也有不少文字到图片的方法，它们大多数都是基于GANs的。一个最普遍的方法是将整个文本编码成全局句子向量(global sentence vector)，并将它作为GANs的条件来生成图片。但是由于GANs仅仅依靠全局句子向量的就很难在单词的级别上生成更精细的图片，即图片很难反应具体单词的含义且清晰度不够。

## 解决思路
为了解决上述提到的问题，作者使用注意力机制，来加强生成器对整个句子的具体单词的理解，使用多级加强来生成更加细粒度、清晰的图片。作者将这个模型称之为AttnGAN。具体的结构如图。

![attn-gan-architecture](/images/2018/05/attn-gan-architecture.png)

**1. Attention Generation Network**
注意力机制是用来根据单词生成不同的最相关子区域。也就是说，我们不仅仅要将自然语言编码成一个全局句子向量，还需要将其编码一单词向量(word vector)。生成网络**第一步**使用全局句子向量来生成一个低分辨率的图片。**之后的阶段** ，图片向量在每一个子区域使用注意力层查询单词向量，以此生成单词语境向量(word context vector)。之后将区域图片和它对应的单词语境向量结合起来，生成一个多模态语境向量(multimodal context vector)。这样的网络将能够高效的生成细节丰富的高分辨率图片。

**2. Deep Attentional
Multimodal Similarity Model (DAMSM)**
DAMSM是为了计算生成图片与使用全局句子向量和细粒度的单词信息之间的相似性，使用该方法来作为图片-文本的匹配损失(matching loss)。具体的细节见下

## 实现细节
**Attention Generation Network**
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
{% endraw %}

$x_i$来自真实图片分布$p_{data_i}$，$\hat{x}_i$来自模型分布$p_{G_i}$


**Deep Attentional Multimadal Similarity Model**
