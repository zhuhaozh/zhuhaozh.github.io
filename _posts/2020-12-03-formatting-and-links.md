---
layout: post
title:  AOT: Appearance Optimal Transport Based Identity Swapping for Forgery Detection
date:   2020-12-03 16:40:16
description: march & april, looking forward to summer
---
## AOT: Appearance Optimal Transport Based Identity Swapping for Forgery Detection
<hr/>

[Hao Zhu](https://www.zhuhaozh.xyz)<sup>1,2</sup>, [Chaoyou Fu](https://scholar.google.com/citations?user=4A1xYQwAAAAJ&hl=en)<sup>2</sup>, [Qianyi Wu](https://qianyiwu.github.io)<sup>3</sup>, [Wayne Wu](https://wywu.github.io)<sup>3</sup>, [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=en)<sup>3</sup>, [Ran He](https://scholar.google.com/citations?user=ayrg9AUAAAAJ&hl=en)<sup>2</sup>

<sup>1</sup>Anhui University  ｜ <sup>2</sup>NLPR&CRIPAC, CASIA ｜ <sup>3</sup>SenseTime Research

[Paper](https://arxiv.org/abs/2011.02674) ｜ [Demo](#)  ｜ [Github](#)

<img src="/assets/posts/aot/banner.gif" style="zoom:100%;" />
<hr/>


### Abstract
> Recent studies have shown that the performance of forgery detection can be improved with diverse and challenging Deepfakes datasets. However, due to the lack of Deepfakes datasets with large variance in appearance, which can be hardly produced by recent identity swapping methods, the detection algorithm may fail in this situation. In this work, we provide a new identity swapping algorithm with large differences in appearance for face forgery detection. The appearance gaps mainly arise from the large discrepancies in illuminations and skin colors that widely exist in real-world scenarios. However, due to the difficulties of modeling the complex appearance mapping, it is challenging to transfer fine-grained appearances adaptively while preserving identity traits. This paper formulates appearance mapping as an optimal transport problem and proposes an Appearance Optimal Transport model (AOT) to formulate it in both latent and pixel space. Specifically, a relighting generator is designed to simulate the optimal transport plan. It is solved via minimizing the Wasserstein distance of the learned features in the latent space, enabling better performance and less computation than conventional optimization. To further refine the solution of the optimal transport plan, we develop a segmentation game to minimize the Wasserstein distance in the pixel space. A discriminator is introduced to distinguish the fake parts from a mix of real and fake image patches. Extensive experiments reveal that the superiority of our method when compared with state-of-the-art methods and the ability of our generated data to improve the performance of face forgery detection.

### Overview
<img src="/assets/posts/aot/pipeline.png" alt="pipeline" style="zoom:80%;" />



### Important Note

ANY RESOUCES PROVIDED HERE ARE NOT TO BE USED FOR MALICIOUS OR INAPPROPRIATE USE CASES.


### Face Forensics
The following datasets and the checkpoints of detectors can be downloaded from the [BaiduYun](https://pan.baidu.com/s/143Xuvea-ICcFuvgYfyY-Wg) (code: dzgc).

#### Manipulated Dataset

We currently provide 100 manipulated videos of [FF++](#) by refining the results of [DeepFaceLab](#) and  [DeeperForensics-1.0](#) respectively.

The more manipulated videos are coming soon.

 <img src="/assets/posts/aot/datasets.png" width="100%"  align=center />

[comment]: <> (![image-20201014221758682]&#40;/assets/posts/aot/datasets.png&#41;)


#### Face Forgery Detection

Binary detection accuracy of two video classification baselines: [I3D](https://github.com/piergiaj/pytorch-i3d) and [TSN](https://github.com/yjxiong/tsn-pytorch) on the hidden set provided by [DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0).
1) We trained the baselines on four manipulated datasets of [FF++](#) produced by [DeepFakes](#),  [Face2Face](#), [FaceSwap](#), and [NeuralTextures](#) **(Green bars)**.

2) Then, we add 100 manipulated videos produced by our method to the training set. All detection accuracies are improved with the addition of our data. **(Blue bars)**.



 <img src="/assets/posts/aot/detection_results.png" width = "80%"  align=center />

[comment]: <> (![image-20201011125755388]&#40;/assets/posts/aot/detection_results.png&#41;)



### Citation
    @inproceedings{aot2020neurips,
      title={AOT: Appearance Optimal Transport Based Identity Swapping for Forgery Detection},
      author={Zhu, Hao and Fu, Chaoyou and Wu, Qianyi and Wu, Wayne and Qian, Chen and He, Ran},
      booktitle={Neural Information Processing Systems (NeurIPS)},
      year={2020}
    }



### Acknowledgments
This work is partially funded by Beijing Natural Science Foundation (Grant No. JQ18017) and Youth
Innovation Promotion Association CAS (Grant No. Y201929).
