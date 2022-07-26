# AnyNet
###### tags: `AnyNet`
## :beginner: Introduction
![demo.gif animation](readme_images/demo.gif)
The goal of this repo is to re-implement the amazing work of [Yan Wang](https://github.com/mileyan) et al. for **Anytime Stereo Image Depth Estimation on Mobile Devices**. Original code and paper could be found via the following links:
1. [Original repo](https://github.com/mileyan/AnyNet)
2. [Original paper](https://arxiv.org/abs/1810.11408)

This repo support pytorch 1.10.0+

## :milky_way: Difference bwtween original repo and this repo
1. [build residual cost volume](https://github.com/gyes00205/AnyNet/blob/b042b4470d6cf40e4726904a66ef93b00ec50887/models/anynet.py#L39)
    * The reason we can see this [issue](https://github.com/mileyan/AnyNet/issues/41#issue-1315917582).
2. [refine network without spn](https://github.com/gyes00205/AnyNet/blob/b042b4470d6cf40e4726904a66ef93b00ec50887/models/anynet.py#L17)

## :key: Training
### Pretrain on SceneFlow dataset
We pretrain our model on SceneFlow dataset for 10 epochs.
<details>
  <summary>pretrained script</summary>

```
python main.py --save_path results/pretrained_anynet_refine \
               --with_refine \
               --datapath your_path
```
</details>

### Finetune on KITTI 2015
We finetune our model on KITTI 2015 dataset for 300 epochs. Split 80% data for training and 20% for validation.
<details>
  <summary>finetune KITTI 2015 script</summary>

```
python finetune.py --save_path results/finetune_anynet_refine \
                   --pretrained results/pretrained_anynet_refine/checkpoint.tar \
                   --with_refine \
                   --datapath your_path \
                   --datatype 2015 \
                   --split_file dataset/KITTI2015_val.txt
```
</details>

### Finetune on KITTI 2012
We finetune our model on KITTI 2012 dataset for 300 epochs. Split 80% data for training and 20% for validation.
<details>
  <summary>finetune KITTI 2012 script</summary>

```
python finetune.py --save_path results/finetune_anynet_refine_2012 \
                   --pretrained results/pretrained_anynet_refine/checkpoint.tar \
                   --with_refine \
                   --datapath your_path \
                   --datatype 2012 \
                   --split_file dataset/KITTI2012_val.txt
```
</details>

## :mag_right: Result on KITTI 2012, 2015 Validation


| Dataset    | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| ---------- | ------- | ------- | ------- | ------- |
| KITTI 2012 | 16.86%  | 10.68%  | 7.15%   | 7.15%   |
| KITTI 2015 | 16.57%  | 10.58%  | 6.33%   | 6.16%   |

## :fire: Runtime on 2080 Ti

|             | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| ----------- | ------- | ------- | ------- | ------- |
| **Runtime** | 5.3ms   | 8.45ms  | 11.1ms  | 11.4ms  |