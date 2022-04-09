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

> **Object Detection**: Please see <a href="#Usage">Usage</a> for a quick start;

> **Sementic Segmentation**: Please see <a href="https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation">ViTAE-Transformer for semantic segmentation</a>;

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

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 43.7 | 39.8 | 48M | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/19UOW0xl0qc-pXQ59aFKU5w) |
| ViTAEv2-S | ImageNet-1K | 1x | 46.3 | 41.8 | 37M | [config](configs/vitaev2/mask_rcnn_vitaev2_s_mstrain_480-800_adamw_1x_coco.py) | [github](log/mask_rcnn_vitaev2_s-480-800_adamw_1x_coco.out) | Coming Soon |
| Swin-T | ImageNet-1K | 3x | 46.0 | 41.6 | 48M | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1Te-Ovk4yaavmE4jcIOPAaw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1YpauXYAFOohyMi3Vkb6DBg) |
| ViTAEv2-S | ImageNet-1K | 3x | 47.8 | 42.6 | 37M | [config](configs/vitaev2/mask_rcnn_vitaev2_s_mstrain_480-800_adamw_3x_coco.py) | [github](log/mask_rcnn_vitaev2_s-480-800_adamw_3x_coco.out) | Coming Soon |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 48.1 | 41.5 | 86M | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1eOdq1rvi0QoXjc7COgiM7A) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1-gbY-LExbf0FgYxWWs8OPg) |
| ViTAEv2-S | ImageNet-1K | 1x | 50.6 | 43.6 | 75M | [config](configs/vitaev2/cascade_mask_rcnn_vitaev2_s_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) | [github](log/cascade_mask_rcnn_vitaev2_s_mstrain_480-800_giou_4conv1f_adamw_1x_coco.out) | Coming Soon |
| Swin-T | ImageNet-1K | 3x | 50.2 | 43.5 | 86M | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_3x.log.json)/[baidu](https://pan.baidu.com/s/1zEFXHYjEiXUCWF1U7HR5Zg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_3x.pth)/[baidu](https://pan.baidu.com/s/1FMmW0GOpT4MKsKUrkJRgeg) |
| ViTAEv2-S | ImageNet-1K | 3x | 51.4 | 44.5 | 75M | [config](configs/vitaev2/cascade_mask_rcnn_vitaev2_s_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](log/cascade_mask_rcnn_vitaev2_s_mstrain_480-800_giou_4conv1f_adamw_3x_coco.out) | Coming Soon |

**Notes:**

- The drop path rate needs to be tuned for various models and heads.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Mask R-CNN model with a `ViTAEv2-S` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/vitaev2/mask_rcnn_vitaev2_s_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/vitaev2):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

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
