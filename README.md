# cas3d
![](https://img.shields.io/badge/python-3.8.1-green)
![](https://img.shields.io/badge/pytorch-1.10.1-green)
![](https://img.shields.io/badge/cudatoolkit-10.2.1-green)
![](https://img.shields.io/badge/cudnn-7.6.5-green)
 
This repo provides a reference implementation of **CasCNN** as described in the paper:

## Basic Usage

### Requirements

The code was tested with `python 3.8.1`, `pytorch 1.10.1`, `cudatoolkit 10.2`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name cas3d python=3.8 

# activate environment
conda activate cas3d

# install tensorflow and other requirements
conda install pytorch==1.10.1 numpy cudatoolkit=10.2 cudnn=7.6.5 -c pytorch
```

### Run the code
```shell
cd ./cas3d

# run cas3d model
python run_model.py twitter 0
```
The first option is the dataset, which can be twitter or weibo; The second option is GPU device number.

## Dataset 
The dataset we used is from CasFlow (https://github.com/Xovee/casflow) , which you can download from this link: 
[Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) or [Baidu Drive (password: `1msd`)](https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ).


## Cite
