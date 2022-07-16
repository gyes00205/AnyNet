# AnyNet
###### tags: `AnyNet`
The goal of this repo is to re-implement the amazing work of [Yan Wang](https://github.com/mileyan) et al. for **Anytime Stereo Image Depth Estimation on Mobile Devices**. Original code and paper could be found via the following links:
1. [Original repo](https://github.com/mileyan/AnyNet)
2. [Original paper](https://arxiv.org/abs/1810.11408)

This repo support pytorch 1.10.0+
## Difference bwtween original repo and this repo
1. build residual cost volume
2. refine network without spn
## Result on KITTI 2012, 2015 Validation


| Dataset    | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| ---------- | ------- | ------- | ------- | ------- |
| KITTI 2012 | 16.86%  | 10.68%  | 7.15%   | 7.15%   |
| KITTI 2015 | 16.57%  | 10.58%  | 6.33%   | 6.16%   |


