<h1 align="left">ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias</h1> 

<p align="center">
  <a href="#Updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#Usage">Usage</a> |
  <a href="#results">Results&Pretrained Models</a> |
  <a href="#statement">Statement</a> |
</p>

## Current applications

> **Image Classification**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification">ViTAE-Transformer for image classification</a>;

> **Object Detection**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection">ViTAE-Transformer for object detection</a>;

> **Sementic Segmentation**: Please see <a href="#Usage">Usage</a> for a quick start;

> **Animal Pose Estimation**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation">ViTAE-Transformer for animal pose estimation</a>;

> **Matting**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting">ViTAE-Transformer for matting</a>;

> **Remote Sensing**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing">ViTAE-Transformer for Remote Sensing</a>;

## Updates

***24/03/2022***
ViTAEv2 with various downstream tasks released!

## Introduction

<p align="left">This repository contains the code, models, and logs demonstrated in <a href="https://arxiv.org/abs/2202.10108">ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond</a>. It contains several reduction cells and normal cells to introduce scale-invariance and locality into vision transformers. Compared to the first version, we stack the two cells in a multi-stage manner and explore the benefits of another inductive bias, i.e., window-based attentions without shifts for a better trade-off between speed, memory footprint, and performance.<p align="left">This repository contains the code, models, test results for the paper <a href="https://arxiv.org/pdf/2106.03348.pdf">ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias</a>. It contains several reduction cells and normal cells to introduce scale-invariance and locality into vision transformers.</p>

<!-- <img src="figs/NetworkStructure.png"> -->

## Results and Models
### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | UperNet | 512x512 | 160k | 42.05 | 42.78 | 67M | - | - | - |
| Swin-T | UPerNet | 512x512 | 160K | 44.51 | 45.81 | 60M | - | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.log.json)/[baidu](https://pan.baidu.com/s/1dq0DdS17dFcmAzHlM_1rgw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth)/[baidu](https://pan.baidu.com/s/17VmmppX-PUKuek9T5H3Iqw) |
| ViTAEv2-S | UPerNet | 512x512 | 160K | 44.95 | 47.98 | 49M | [config](configs/vitaev2/upernet_vitaev2_s_512x512_160k_ade20k.py) | [github](log/upernet_vitaev2_s_512x512_160k_ade20k.log.json) | coming soon |

**Notes:**

- The drop path rate needs to be tuned for various models and heads.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train an UPerNet model with a `ViTAEv2-S` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/vitaev2/upernet_vitaev2_s_512x512_160k_ade20k.py 8 --options model.pretrained=<PRETRAIN_MODEL> 
```

**Notes:** 
- `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
- The default learning rate and training schedule is for 8 GPUs and 2 imgs/gpu.


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

> **Image Classification**: See [ViTAE for Image Classification](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation)

> **Object Detection**: See [ViTAE for Object Detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection).

> **Semantic Segmentation**: See [ViTAE for Semantic Segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation).

> **Animal Pose Estimation**: See [ViTAE for Animal Pose Estimation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation).

> **Matting**: See [ViTAE for Matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting).

> **Remote Sensing**: See [ViTAE for Remote Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing).


## Statement
This project is for research purpose only. For any other questions please contact [yufei.xu at outlook.com](mailto:yufei.xu@outlook.com) [qmzhangzz at hotmail.com](mailto:qmzhangzz@hotmail.com) .
