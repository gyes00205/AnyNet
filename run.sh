#!/bin/bash

python main.py --save_path results/pretrained_anynet_refine \
               --with_refine



python finetune.py --save_path results/finetune_anynet_refine \
                   --pretrained results/pretrained_anynet_refine/checkpoint.tar \
                   --with_refine

