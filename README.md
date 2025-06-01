# FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting
<p align="center">
<a href=""><img src="https://img.shields.io/badge/Arxiv-XXXX.XXXX-B31B1B.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Video-Youtube-d61c1c.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Project-Page-048C3D"></a>
<!-- <a href="https://github.com/VITA-Group/LightGaussian"><img src="https://img.shields.io/github/stars/VITA-Group/LightGaussian"></a> -->
</p>
<p>

This repository contains the code release for CVPR 2025 paper: "FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting" by Hengyu Liu*, Yuehao Wang*, Chenxin Li*, Ruisi Cai‡, Kevin Wang, Wuyang Li, Pavlo Molchanov, Peihao Wang, Zhangyang Wang. (* denotes equal contribution, ‡ denotes project lead)
</p>




## Environmental Setups

```shell
git clone --recursive https://github.com/LiuHengyu321/FlexGS.git
cd FlexGS
git submodule update --init --recursive

conda create -n flexgs python=3.10
conda activate flexgs

pip install -r requirements.txt
pip install -e submodules/compress-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```
In our environment, we use pytorch=1.13.1+cu121.


## Data Preparation

The used datasets are [MipNeRF360](https://jonbarron.info/mipnerf360/), [Zip-NeRF](https://jonbarron.info/zipnerf/) and [Tank & Temple](https://github.com/graphdeco-inria/gaussian-splatting), which can be directly downloaded from the links.


## Train
Train FlexGS with the following scripts:
```python
python train.py \
    -s /path/to/dataset \
    -m /path/to/ouput_dir \
    --time_ratios 0.20 0.15 0.10 0.05 0.01 \
    --configs ./arguments/flex1_small1105.py \
    --resolution 4 \
    --gumbel_weight 1.0 \
    --select_interval 1000 
```
- `-s`: path to the source data
- `-m`: output path
- `--time_ratios`: the given elastic ratios for training
- `--configs`: path to the config
- `--resolution`: resolution downscale of each image
- `--gumbel_weight`: weight of the Gumbel loss 
- `--select_interval`: interval for updating the Global Importance


## Render and Eval
Render the image on the novel view on various elastic ratios.
```python
python render.py \
    --configs ./arguments/flex1_small1105.py \
    -m /path/to/ouput_dir \
    --time_ratios 0.15 0.10 0.05 0.01 \
```
- `-m`: output path
- `--time_ratios`: the given elastic ratios for rendering
- `--configs`: path to the config
- `--skip_video`: skip rendering the video
- `--skip_train`: skip rendering the train views
- `--skip_test`: skip rendering the novel views

Metrics calculation: calculating the PSNR, SSIM and LPIPS for rendering results under all the given elastic ratios.

```python
python metrics.py \
    -m /path/to/ouput_dir \
```


## Acknowledgements
Our code is based on the following awesome repositories:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [4DGaussians](https://github.com/hustvl/4DGaussians)
- [LightGaussian](https://github.com/VITA-Group/LightGaussian)

We thank the authors for releasing their code!

## BibTeX
If you find our work useful for your project, please consider citing the following paper.

```


```
