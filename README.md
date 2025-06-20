# FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting
<p align="center">
<a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_FlexGS_Train_Once_Deploy_Everywhere_with_Many-in-One_Flexible_3D_Gaussian_CVPR_2025_paper.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-blue" alt="Paper PDF"></a>
<a href="https://arxiv.org/abs/2506.04174"><img src="https://img.shields.io/badge/Arxiv-2506.04174-B31B1B.svg"></a>
<a href="https://flexgs.github.io/"><img src="https://img.shields.io/badge/Project-Page-orange"></a>
<a href="https://youtu.be/k6aDJUfxs4Q"><img src="https://img.shields.io/badge/Video-Youtube-k6aDJUfxs4Q.svg"></a>
</p>
<p>

This repository contains the code release for CVPR 2025 paper: *"FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting"* by Hengyu Liu*, Yuehao Wang*, Chenxin Li*, Ruisi Cai‡, Kevin Wang, Wuyang Li, Pavlo Molchanov, Peihao Wang, Zhangyang Wang. (* denotes equal contribution, ‡ denotes project lead)
</p>




## Environmental Setup

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
We use pytorch=1.13.1+cu121 as our experiment environment.


## Data Preparation

Our experiments are conducted on [MipNeRF360](https://jonbarron.info/mipnerf360/), [Zip-NeRF](https://jonbarron.info/zipnerf/) and [Tank & Temple](https://github.com/graphdeco-inria/gaussian-splatting) datasets.


## Train
Train FlexGS with the following scripts:
```python
python train.py \
    -s /path/to/dataset \
    -m /path/to/ouput_dir \
    --time_ratios 0.20 0.15 0.10 0.05 0.01 \
    --configs ./arguments/flex.py \
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
Render the image on the novel view at various elastic ratios.
```python
python render.py \
    --configs ./arguments/flex.py \
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
If you find our work useful for your projects, please consider citing the paper:

```
@inproceedings{liu2025flexgs,
  title={FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting},
  author={Liu, Hengyu and Wang, Yuehao and Li, Chenxin and Cai, Ruisi and Wang, Kevin and Li, Wuyang and Molchanov, Pavlo and Wang, Peihao and Wang, Zhangyang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
