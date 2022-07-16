#!/bin/bash

python main.py --save_path results/pretrained_anynet_refine \
               --with_refine



python finetune.py --save_path results/finetune_anynet_refine \
                   --pretrained results/pretrained_anynet_refine/checkpoint.tar \
                   --with_refine

python finetune.py --save_path results/finetune_anynet_refine_2012 \
                   --pretrained results/pretrained_anynet_refine/checkpoint.tar \
                   --with_refine \
                   --datapath /media/bsplab/62948A5B948A3219/data_stereo_flow_2012/training/ \
                   --datatype 2012 \
                   --split_file dataset/KITTI2012_val.txt