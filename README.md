# Efficient View Path Planning for Autonomous Implicit Reconstruction

The official implementation of the paper [Efficient View Path Planning for Autonomous Implicit Reconstruction]().

Accepted by ICRA 2023.

Project Page: https://small-zeng.github.io/EVPP/

Paper Link: https://ieeexplore.ieee.org/abstract/document/10160793

GitHub Repo: https://github.com/small-zeng/EVPP

This project is built on [ashawkey/torch-ngp](https://github.com/ashawkey/torch-ngp)'s NGP and TensoRF implementation.

## Unity Project
百度云盘 https://pan.baidu.com/s/12lXVDL-f1cGk8O9LpVAzXA?pwd=j5ey

## Installation

```bash
git clone https://github.com/small-zeng/EVPP.git
cd EVPP
```

### Install with conda

```bash
conda env create -f environment.yml
conda activate EVPP
```

### Code Structure

Based on the implementation of the repo, we slightly modified the files in `nerf` to fit our needs.

The main entrances are `nerfServer_VPP` (NGP backbone) and `planServre_.VPP` (TensoRF backbone).

`nerfServer_VPP`  defines the online implicit reconstruction

`planServre_.VPP` defines the view path planning.



## BibTeX

```bibtex
@inproceedings{zeng2023efficient,
  title={Efficient view path planning for autonomous implicit reconstruction},
  author={Zeng, Jing and Li, Yanxu and Ran, Yunlong and Li, Shuo and Gao, Fei and Li, Lincheng and He, Shibo and Chen, Jiming and Ye, Qi},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4063--4069},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement

Use this code under the MIT License. No warranties are provided. Keep the laws of your locality in mind!

Please refer to [torch-ngp#acknowledgement](https://github.com/ashawkey/torch-ngp#acknowledgement) for the acknowledgment of the original repo.



10.15.198.53:7100/isfinish/?finish=yes