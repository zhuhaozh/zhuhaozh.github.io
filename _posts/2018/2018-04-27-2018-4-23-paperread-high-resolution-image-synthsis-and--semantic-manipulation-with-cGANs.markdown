---
layout: post
title: '论文阅读笔记：High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs'
date: '2018-04-27 14:02'
categories: paper-read
tags: cvpr2018 gans
---

* content
{:toc}

High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

## Abstract
这篇文章里提出来使用CGANs，通过语义标签图(semantic label maps)来一个合成高分辨率的具有仿现实的图片，

使用一个新颖的 adversarial loss/mutil-scale generator/mutil-scale discriminator





两个可视化的操纵功能：
1. we incorporate object instance segmentation information, which enables object manipulations
such as removing/adding objects and changing the object category.
2. we propose a method to generate diverse results given the same input, allowing users to edit
the object appearance interactively. Human opinion studies demonstrate that our method significantly outperforms
existing methods, advancing both the quality and the resolution of deep image synthesis and editing

And This method advancing both the quality and the resolution of deep image synthesis and editing.

## Introduction
目前的状况：
Although existing graphics algorithms excel at the task, building and editing virtual environments is expensive and time-consuming

需要解决的问题：
If we were able to render photo-realistic images
using a model learned from data, we could turn the process
of graphics rendering into a model learning and inference
problem. Then, we could simplify the process of creating
new virtual worlds by training models on new datasets. We
could even make it easier to customize environments by allowing users to simply specify semantic information rather
than modeling geometry, materials, or lighting.

该方法的应用场景：
we can use it to create synthetic training data for training visual recognition algorithms, since it is much easier to create
semantic labels for desired scenarios than to generate training images. Using semantic segmentation methods, we can
transform images into a semantic label domain, edit the objects in the label domain, and then transform them back to
the image domain. This method also gives us new tools for
higher-level image editing, e.g., adding objects to images or
changing the appearance of existing objects

目前的从Senmantic labels合成图片的方法：
can use the pix2pix method, an image-to-image translation framework [21] which leverages generative adversarial networks
(GANs) [16] in a conditional setting.

Recently, Chen and Koltun [5] suggest that adversarial training might be unstable and prone to failure for high-resolution image generation tasks. Instead, they adopt a modified perceptual
loss [11, 13, 22] to synthesize images, which are highresolution but often lack fine details and realistic textures

该文章解决的当今问题：
1. the difficulty of generating highresolution images with GANs [21]
2. the lack of details and realistic textures in the previous high-resolution
results

解决的途径：
through a new, robust adversarial learning objective together with new multi-scale generator
and discriminator architectures, we can synthesize photorealistic images at 2048 × 1024 resolution, which are more
visually appealing than those computed by previous methods

to support interactive semantic manipulation：
we extend our method in two directions.
First, we use instance-level object segmentation information, which can separate different object instances within the same category. This enables flexible object manipulations, such as
adding/removing objects and changing object types.
Second, we propose a method to generate diverse results given the same input label map, allowing the user to edit the appearance of the same object interactively.


## Related Network
1. GANs
  - GANs enable a wide variety of applications.
  - Inspired by their successes, we propose a new coarse-to-fine generator and multi-scale discriminator architectures suitable for conditional image generation at a much higher resolution.

2. Image2Image translation
  - Many researchers have leveraged adversarial learning for image-to-image translation [21], whose goal is to translate an input image from one domain to another domain given input-output image pairs as training data.
  - Recently, Chen and Koltun [5] suggest that it might be hard for conditional GANs to generate high-resolution images due to the training instability and optimization issues. To avoid this difficulty, they use a direct regression objective based on a perceptual loss [11, 13, 22] and produce the first model that can synthesize 2048 × 1024 images.he generated results are high-resolution but often lack fine details and realistic textures.
  - Our method is motivated by their success. We show that using our new objective function as well as novel multi-scale generators and discriminators, we not only largely stabilize the training of conditional GANs on high-resolution images, but also achieve significantly better results compared to Chen and Koltun

3. Deep visual manipulation
  - deep neural networks have obtained promising results in various image processing tasks, However, most of these works lack an interface for users to adjust the current result or explore the output space.
  - To address this issue, Zhu et al. [64] developed an optimization method for editing the object appearance based on the priors learned by GANs. Recent works [21, 46, 59] also provide user interfaces for creating novel imagery from low-level cues such as color and sketch. All of the prior works report results on low-resolution images.
  - Our system shares the same spirit as this past work, but we focus on object-level semantic editing, allowing users to interact with the entire scene and manipulate individual objects in the image.

## 实例级的图像合成
### The pix2pix Baseline
pix2pix 方法是一个完成图像到图像转换的cGAN框架，它包含一个生成器G和判别器D。而对于我们的任务中，生成器G是为了将语义标注图转换成一个类似真实的图片，同时D的目标是为了分辨图片是真实的还是转换而来的生成图片。这个框架使用监督的方式执行。即在训练数据集中，给出的是 images{(s_i,x_i)}
，s_i 是语义标注图，xi是对应的真实照片。cGAN则是为了通过“最大最小游戏：min(G)max(D)Loss_GAN(G,D)”，生成一个条件分布：P(x_i|s_i)，其中的Loss_GAN(G,D)为：
![loss(G,D)](/images/2018/04/loss-g-d.png)[eq2]

在pix2pix方法中，采用了一个U-Net作为生成器和一个patch-based fully convolutional network 作为判别器，给判别器输入的是一个channel-wise concatenation of the semantic label map and the corresponding image
然而，使用Cityscapes生成的图片的分辨率只有256*256，也同样测试直接将pix2pix应用在高分辨率的场合，但训练的十分不稳定且效果不好
### Improving Photorealism and Resolution
通过使用 coarse-to-fine generator, a multi-scale discriminator architecture, and
a robust adversarial learning objective function 来改进pix2pix框架

**1. 由粗到精的生成器**

将生成器分解成两个子网络G1和G2，将G1定义为全局的生成器网络，而G2作为局部增强的网络，G1生成一个1024×512分辨率的图片，G2输出4倍于G1生成的图片2048*1024，同时，为了合成更高分辨率的图片，可以增加额外的局部增强网络。

G1(Global Generator)建立在由Johnson等人提出的架构[22]上，该架构已被证明能够成功的运用在512*512的风格转换网络上。该架构包括三个组件：a convolutional front-end G1(F), a set of residual blocks G1(R) [18], and a transposed convolutional back-end G1(B)

G2同样包括三个组件：a convolutional front-end G2(F), a set of residual blocks G2(R), and a transposed convolutional back-end G2(B)
![architecture of generator](/images/2018/04/architecture-of-generator.png)
与G1不同的是，输入给G2残差网络的数据是G1的输出与G2的特征图(Feature map)相应元素相加。[???]

在训练的过程中，首先训练全局生成器G1,再训练G2,最后将两个结合在一起调优。

**2. 多粒度的判别器(mutil-scale discriminators)**

为了能够分别高分辨率的图片是真实的还是合成的，判别器需要一个大的接收域(receptive field)，因此，这需要一个更深的网络或是更大的卷积核。但无论用上面的哪种方法，都会增加网络的能力(capacity)，过拟合也就成了一个需要更加关注的问题。同时这两种方法在训练时也需要更大的内存。

**解决办法**：使用三个判别器D1、D2、D3，有着相同的网络结构(have an identical network structure)，但对不同的图片粒度(iamge scales)操作。对真实的和合成的图片进行下采样，并乘以2和4,用来生成三个不同粒度的图片金字塔。之后D1/D2/D3分别被训练用来在三个不同的尺度区分图片。尽管他们都有相同的结构，但是那个对粗粒度操作的判别器，有着最大的接收域。它对一个图片有着整体的把握，引导生成器来生成大体一致的图片。另一方面，对最细粒度操作的判别器有最小的接收域，它引导生成器生成更好的细节。
同时目标方程也转换成：
![new minmax game](/images/2018/04/new-minmax-game.png)

**3. 改进的对抗损失(Adversarival loss)**

通过增加基于判别器的特征的损失[eq2]来改进GAN loss，生成器需要在多粒度的条件下产生自然的统计，这使得训练变的稳定(This loss stabilizes the training as the
generator has to produce natural statistics at multiple scales)。特别的，从多层判别器中抽取特征，然后学习去在真实与生成的图片中，匹配这些中间表示(intermediate representations)
![feature matching loss](/images/2018/04/feature-matching-loss.png)
其中判别器D_k的第i层的特征抽取器用D_k^(i)表示，Ni表示每一层的元素数量，这个GAN discriminator feature matching loss与perceptual loss相关

完整的目标方程为：
![full objective](/images/2018/04/full-objective.png)
（lambda控制两项的重要性，需要注意的是 feature matching loss L_FM, D_k only serves as a feature extractor and does not maximize the loss L_FM.）

### 使用实例图 (Instance Maps)
目前的图片合成方法，都是使用的语义标注图，而它无法分辨出同一类别的不同物体，当他们重合的时候，效果则会更差。而实例级的语义标注图，则对不同的物体都分配一个不同的ID。

直接把语义标注图传入一个网络或将其编码成一个one-hot vector，这存在的问题是，很难确定一张图中有多少物体

一个简单的方法是提前设置一张图片中存在多少物体，但若设置的物体数目少于真实的物体数目时，方法可能会失效，反之则会浪费内存空间。
但考虑到实体图给出的有效信息主要是物体的边界，而不是在语义标注图中可用(which is not available in the semantic label map, is the object boundary)。

为了抽取这些信息，首先计算物体边界图(b)。作者使用的方法是，如过一个物体的id和周围四个都不相同，则把它设置为1（白色），反之设置为0（黑色）。如图所示

![boundary map](/images/2018/04/boundary-map.png)

之后让物体边界图与one-hot vector相连，来代表语义标注图传给生成器。类似的，输入给判别器的是物体边界图、语义标注图、真实/合成的图片 在通道级别上相连接( channel-wise concatenation)


### 学习实体级特征嵌入(Learning an Instance-level Feature Embedding)
从语义标注图来合成图片是一个一对多映射的问题，一个理想的合成算法应该能够使用同一个语义标注图来生成多种不同的仿现实的图片。

**目前存在的方式**：给定一个相同的输入产生一个固定数量的离散输出，或由一个隐含的编码来编码整个图片来合成多种模式。

**存在的问题**：尽管这些方法能够解决多模态图像合成问题，但它并不适合本文提到的图像编辑任务。原因如下：
1. 用户不知道什么样的图片应该生成
2. 这些方法都只关注于全局颜色风格和纹理的变化，并允许在生成内容上的无实例级别的控制(allow no object-level control on the generated contents)

**解决办法**：为了能够生成多样的图片和允许实体级别的控制，作者提出了增加一个额外的低维度特征通道作为生成网络的输入，通过修改这些特征，我们可以灵活的图片的合成过程。

为了生成低维度特征，可以训练一个编码网络E，来寻找与图像中每个实例的真实目标对应的低维度特征向量。这个特征编码器的架构是一个标准的标准的编码解码网络。为了保证特征与每个实例是一致的，作者添加来一个实体间的(instance-wise)average pooling layer来输出编码器计算的实体的平均特征
见图：
![feature encoder network E](/images/2018/04/feature-encoder-network-e.png)

将[eq5]中的G(s) 替换为 G(s,E(x))，并且与生成器和判别器一起训练encoder E。
1. 在编码器训练之后，将它运行在训练图片的每个实体上，并记录下获得的特征。
2. 对每一个语义分类上的特征做一次K-means聚类，这时每一个聚类都对一个特定的风格编码出一个特征。
3. 在干预的时候(inference time)，随机选择一个聚类的中心，并把它作为编码后的特征，这些特征和标注图相连，作为生成器的输入。
