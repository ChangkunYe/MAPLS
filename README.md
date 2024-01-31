# Label Shift Estimation for Class-Imbalance Problem: A Bayesian Approach
This is the official implementation for WACV 2024 paper "Label Shift Estimation for Class-Imbalance Problem: A Bayesian Approach".

If you find this repository useful or use this code in your research, please cite the following paper: 
 ```
 @InProceedings{Ye_2024_WACV,
    author    = {Ye, Changkun and Tsuchida, Russell and Petersson, Lars and Barnes, Nick},
    title     = {Label Shift Estimation for Class-Imbalance Problem: A Bayesian Approach},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {1073-1082}
}
 ```
## Requirements
The code is written in [PyTorch](https://pytorch.org/). It is recommned to install via conda:
```
conda install scipy
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge cvxpy
```

When train the Neural Network classifier from scratch, the recommended hardware setup is as follows:

|   Dataset   |      #GPU      |        #CPU         |
|:-----------:|:--------------:|:-------------------:|
| CIFAR10/100 |   &ge; 2 Gb    | &gt; 4 + 1 threads  |
|  ImageNet   | &ge; 4 x 12 Gb | &gt; 16 + 1 threads |
|   Places    | &ge; 6 x 12 Gb | &gt; 24 + 1 threads |


## Dataset Details
Our code support CIFAR10/100, ImageNet 2012 and Places365 datasets.

- CIFAR10/100 dataset: Please download with build-in function of pytorch.
- ImageNet dataset: Please download the ImageNet 2012 at official site https://image-net.org/.  
- Places dataset: Please download at offical site http://places2.csail.mit.edu/

For Long-Tailed version of ImageNet and Places, please download the split at [here](https://drive.google.com/drive/u/0/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).
This split provided by [Large-Scale Long-Tailed Recognition in an Open World](https://github.com/zhmiao/OpenLongTailRecognition-OLTR) paper.

```
data
  |--CIFAR10
    |--cifar-10-batches-py
  |--CIFAR100
    |--cifar-100-python
  |--ImageNet
    |--train
    |--val
    |--ImageNet_LT_train.txt
    |--ImageNet_LT_test.txt
    |--ImageNet_LT_val.txt
  |--Places
    |--data_256
    |--val_256
    |--test_256
    |--Places_LT_train.txt
    |--Places_LT_test.txt
    |--Places_LT_val.txt
```

## Train the classifier
To train the classifier from scratch, please adjust the GPU ids "CUDA_VISIBLE_DEVICES", dataset path "$data_path" and config path "$cfg_path" in the bash script "./train_script.sh" and run:
```
./train_script.sh
```
Config examples are provided in "./config/".



## Test Label Shift Estimation Model

To test existing models performance under label shift, the dataset path "$data_path" and checkpoint path "$ckpt_path" in the bash script "./test_script.sh" and run: 
```
./test_script.sh
```
The "$cfg_path" in "./test_script.sh" determines the type of label shift, including:

- "./config/batch_imb_LT": Ordered Long-Tailed Shift
- "./config/batch_imb_shuffle": Shuffled Long-Tailed Shift
- "./config/batch_imb_dirichlet": Dirichlet Shift
- "./config/batch_imb_knockout": Knockout Shift



## License
Please see LICENSE

## Questions?
Pleas raise issues or contact author at changkun.ye@anu.edu.au.