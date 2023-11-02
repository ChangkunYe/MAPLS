#!/bin/bash
data_path="./data/"
ckpt_name="./PreTrained/ImageNet-resnet50-ncCNn-20220929-211312.pth"

# cfg_path="./config/batch_imb_knockout/"
cfg_path="./config/batch_imb_LT/"
# cfg_path="./config/batch_imb_shuffle/"
# cfg_path="./config/batch_imb_dirichlet/"

for i in $(find $cfg_path -name \*.txt* -print0  | sort -z | xargs -r0); do
        echo $i
        python3 main.py --is_test True --datapath $data_path --checkpoint $ckpt_name  --test_config "$i"

        if [[ "$i" == "./config/batch_imb_LT/imb_test.txt4" ]]
        then
                python3 main.py --is_test True --datapath $data_path --checkpoint $ckpt_name
                echo 'uniform'
        fi
done