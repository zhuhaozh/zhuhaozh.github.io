---
layout: post
title: '论文阅读笔记：Learning to Compare: Relation Network for Few-Shot Learning'
date: '2018-04-29 00:49'
categories: paper-read
tags: cvpr2018 gans
mathjax: true
author: zh
---

* contents
{:toc}


## 背景与现状
阻碍Person ReId的两大原因：

1. 目前所使用的数据集与现实世界面临的差距很大，且数据量很少
2. 不同数据集之间的domain gap会导致严重的性能下降，如不同的灯光条件、分辨率、季节、背景等原因





## 解决方案
为了解决这两个问题，作者做了分别两方面的努力：

1. 采集了一个新的多场景、多时间的person ReID数据集(MSMT17)，它与目前的类似数据集不同的是，MSMT17有如下特征：1)原始的视频是由15个部署在室外或室内场景的摄像机，因此它有着复杂的场景转换和背景；2)这些视频覆盖了很长一段时间，有着复杂的光照条件；3)它有着目前最大的标注实体和bounding box
2. 为了弥补domain gap，通过将A数据集中的人物转换为B的数据集的形式，即，从A转换出的人物保留自己的实体特征，同时有着和B中数据集类似的风格，如背景、光照等，受Cycle-GAN的启发实现该模型，将其称为PTGAN。

## 相关工作
**1. Person ReID中的解释性学习(Descriptor Learning)**

**2. 利用GAN的Image2Image转换**

## MSMT17

## Person Transfer GAN
正如前面所说，不同的数据集之间的视频风格可能完全不同，我们训练的网络也与真实应用的场景拍摄下来的照片可能也会完全不同，为了弥补训练集与测试集之间的domain gap，所以就提出了Person Transfer GAN。即将数据集A中的人转换为B数据集的风格。

PTGAN的想法是受到了cycle GAN的启发。PTGAN需要1)转换图片的风格，2)保持图片中人物的特征。所以转换后与转换前的两个图片要被认做有着一样的人物ID。
将PTGAN的损失函数定义为：
$$\mathcal{L}_{PTGAN} = \mathcal{L}_{Style} + \lambda_1\mathcal{L}_{ID} $$
其中$\lambda_1$是用来平衡风格损失和特征损失。

同时因为ReID的数据集并不包含某个人成对的图片(同样的人来自不同的数据集)，所以就可以将风格转换任务看做是无配对的image to image 转换任务。因为Cycle-GAN在这项人物中有着很好的表现，所以就使用Cycle-GAN来学习两个数据集中间的风格映射函数。$G$被认做是$A\rightarrow B$的风格映射函数，$\overline{G}$则是$B\rightarrow A$的风格映射函数。$D_A$和$D_B$分别是A和B的风格判别器。风格转换的损失函数可以定义为：
$$\mathcal{L}_{Style} = \mathcal{L}_{GAN}(G,D_B,A,B)+\mathcal{L}_{GAN}(\overline{G},D_A,B,A)+\lambda_2 \mathcal{L}_{cyc}(G,\overline{G})$$
其中，$\mathcal{L}_{GAN}$为标准的对抗损失，$\mathcal{L}_{cyc}$为标准的循环一致性损失。

人物特征损失的计算过程为：1)获取人物的前景掩码，2)计算转换前后人物前景的变化。

给定A的数据分布为 $a \sim \mathcal{p}_{data}(a)$，B的数据分布为$b \sim \mathcal{p}_{data}(b)$
所以特征损失函数为
$$\mathcal{L}_{ID} = \mathbb{E}_{a \sim \mathcal{p}_{data}(a)}[||(G(a) -a) \odot M(a)||{_2}] + \mathbb{E}_{b \sim \mathcal{p}_{data}(b)}[||(\overline G(b) -b) \odot M(b)||{_2}]$$

G(a)代表从a转换后的图片，M(a)代表a的前景掩码
使用PSPNet来抽取图片关于人物的前景掩码
![examples](/images/2018/04/examples.png)

## 实验
