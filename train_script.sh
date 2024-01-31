#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
dataset_path="./data/"
cfg_path="./config/batch_config/config_cifar100.txt"

python3 main.py --datapath $dataset_path --config $cfg_path