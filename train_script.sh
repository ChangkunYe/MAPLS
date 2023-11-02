#!/bin/bash
dataset_path="./data/"
save_path="./saved_models/"
cfg_path="./config/batch_config/config_cifar100.txt"

python3 main.py --datapath $dataset_path --config $cfg_path

mkdir $save_path
mv ./log/* $save_path
mv ./saved_models/* $save_path