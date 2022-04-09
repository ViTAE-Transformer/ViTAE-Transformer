<h1 align="left">ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond<a href="https://arxiv.org/abs/2202.10108"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a></h1> 

<p align="center">
  <a href="#Updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#statement">Statement</a> |
</p>

## Current applications

> **Image Classification**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification">ViTAE-Transformer for image classification</a>;

> **Object Detection**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection">ViTAE-Transformer for object detection</a>;

> **Sementic Segmentation**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation">ViTAE-Transformer for semantic segmentation</a>;

> **Animal Pose Estimation**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation">ViTAE-Transformer for animal pose estimation</a>;

> **Matting**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting">ViTAE-Transformer for matting</a>;

> **Remote Sensing**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing">ViTAE-Transformer for Remote Sensing</a>;


## Updates

***09/04/2021***
- The pretrained models for ViTAE on [matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting) and [remote sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing) are released! Please try and have fun!

***24/03/2021***
- The pretrained models for both ViTAE and ViTAEv2 are released. The code for downstream tasks are also provided for reference.

***07/12/2021***
- The code is released!

***19/10/2021***
- The paper is accepted by Neurips'2021! The code will be released soon!
  
***06/08/2021***
- The paper is post on arxiv! The code will be made public available once cleaned up.

## Introduction

<p align="left">This repository contains the code, models, test results for the paper <a href="https://arxiv.org/pdf/2106.03348.pdf">ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias</a>. It contains several reduction cells and normal cells to introduce scale-invariance and locality into vision transformers. In <a href="https://arxiv.org/pdf/2202.10108.pdf">ViTAEv2</a>, we explore the usage of window attentions without shift operations to obtain a better balance between memory footprint, speed, and performance. We also stack the proposed RC and NC in a multi-stage manner to faciliate the learning on other vision tasks including detection, segmentation, and pose.

<figure>
<img src="figs/NetworkStructure.png">
<figcaption align = "center"><b>Fig.1 - The details of RC and NC design in ViTAE.</b></figcaption>
</figure>

<figure>
<img src="figs/ViTAEv2.png">
<figcaption align = "center"><b>Fig.2 - The multi-stage design of ViTAEv2.</b></figcaption>
</figure>


## Statement
This project is for research purpose only. For any other questions please contact [yufei.xu at outlook.com](mailto:yufei.xu@outlook.com) [qmzhangzz at hotmail.com](mailto:qmzhangzz@hotmail.com) .

## Citing ViTAE and ViTAEv2
```
@article{xu2021vitae,
  title={Vitae: Vision transformer advanced by exploring intrinsic inductive bias},
  author={Xu, Yufei and Zhang, Qiming and Zhang, Jing and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
@article{zhang2022vitaev2,
  title={ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond},
  author={Zhang, Qiming and Xu, Yufei and Zhang, Jing and Tao, Dacheng},
  journal={arXiv preprint arXiv:2202.10108},
  year={2022}
}
```

## Other Links

> **Image Classification**: See [ViTAE for Image Classification](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification)

> **Object Detection**: See [ViTAE for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection).

> **Semantic Segmentation**: See [ViTAE for Semantic Segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation).

> **Animal Pose Estimation**: See [ViTAE for Animal Pose Estimation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation).

> **Matting**: See [ViTAE for Matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting).

> **Remote Sensing**: See [ViTAE for Remote Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing).