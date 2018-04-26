---
layout: post
title: >-
  论文阅读：High-Resolution Image Synthesis and Semantic Manipulation with
  Conditional GANs
date: '2018-04-27 00:02'
categories: cvpr gans paper-read
---

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
pix2pix 方法是一个完成图像到图像转换的cGAN框架，它包含一个生成器G和判别器D。而对于我们的任务中，生成器G是为了将语义标注图转换成一个类似真实的图片，同时D的目标是为了分辨图片是真实的还是转换而来的生成图片。这个框架使用监督的方式执行。即在训练数据集中，给出的是$$ images{(s_i,x_i)} $$
，s_i 是语义标注图，xi是对应的真实照片。cGAN则是为了通过“最大最小游戏：min(G)max(D)Loss_GAN(G,D)”，生成一个条件分布：P(x_i|s_i)，其中的Loss_GAN(G,D)为：
![loss(G,D)](images/2018/04/loss-g-d.png)

在pix2pix方法中，采用了一个U-Net作为生成器和一个patch-based fully convolutional network 作为判别器，给判别器输入的是一个channel-wise concatenation of the semantic label map and the corresponding image
然而，使用Cityscapes生成的图片的分辨率只有256*256，也同样测试直接将pix2pix应用在高分辨率的场合，但训练的十分不稳定且效果不好
### Improving Photorealism and Resolution
通过使用 coarse-to-fine generator, a multi-scale discriminator architecture, and
a robust adversarial learning objective function 来改进pix2pix框架

1. 由粗到精的生成器

将生成器分解成两个子网络G1和G2，将G1定义为全局的生成器网络，而G2作为局部增强的网络，G1生成一个1024×512分辨率的图片，G2输出4倍于G1生成的图片2048*1024，同时，为了合成更高分辨率的图片，可以增加额外的局部增强网络。

G1(Global Generator)建立在由Johnson等人提出的架构[22]上，该架构已被证明能够成功的运用在512*512的风格转换网络上。该架构包括三个组件：a convolutional front-end G1(F), a set of residual blocks G1(R) [18], and a transposed convolutional back-end G1(B)

G2同样包括三个组件：a convolutional front-end G2(F), a set of residual blocks G2(R), and a transposed convolutional back-end G2(B)
![architecture of generator](images/2018/04/architecture-of-generator.png)
