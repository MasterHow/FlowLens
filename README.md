### <p align="center">FlowLens: Seeing Beyond the FoV via Flow-guided Clip-Recurrent Transformer
<br>
<div align="center">
  <a href="https://www.researchgate.net/profile/Shi-Hao-10" target="_blank">Hao&nbsp;Shi</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.researchgate.net/profile/Qi-Jiang-63" target="_blank">Qi&nbsp;Jiang</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.researchgate.net/profile/Kailun-Yang" target="_blank">Kailun&nbsp;Yang</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.researchgate.net/profile/Yin-Xiaoting" target="_blank">Xiaoting&nbsp;Yin</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.researchgate.net/profile/Kaiwei-Wang-4" target="_blank">Kaiwei&nbsp;Wang</a>
  <br> <br>
  <a href="https://arxiv.org/pdf/2211.11293.pdf" target="_blank">Paper</a>

####
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flowlens-seeing-beyond-the-fov-via-flow/seeing-beyond-the-visible-on-kitti360-ex)](https://paperswithcode.com/sota/seeing-beyond-the-visible-on-kitti360-ex?p=flowlens-seeing-beyond-the-fov-via-flow)

[comment]: <> (<a href="https://arxiv.org/" target="_blank">Paper</a> &emsp;)

[comment]: <> (  <a href="https://arxiv.org/" target="_blank">Demo Video &#40;Youtube&#41;</a> &emsp;)

[comment]: <> (  <a href="https://arxiv.org/" target="_blank">演示视频 &#40;B站&#41;</a> &emsp;)
</div>

[comment]: <> (<br>)

[comment]: <> (<p align="center">:hammer_and_wrench: :construction_worker: :rocket:</p>)

[comment]: <> (<p align="center">:fire: We will release code and checkpoints in the future. :fire:</p>)

[comment]: <> (<br>)

<div align=center><img src="assets/flowlens.png" width="800" height="368" /></div>

### Update
- 2022.11.19 Init repository.
- 2022.11.21 Release the [arXiv](https://arxiv.org/abs/2211.11293) version with supplementary materials.
- 2023.04.04 :fire: Our code is publicly available.
- 2023.04.04 :fire: Release pretrained models. 
- 2023.04.04 :fire: Release KITTI360-EX dataset.

### TODO List

- [x] Code release. 
- [x] KITTI360-EX release.
- [x] Towards higher performance with extra small costs.


### Abstract
Limited by hardware cost and system size, camera's Field-of-View (FoV) is not always satisfactory. 
However, from a spatio-temporal perspective, information beyond the camera’s physical FoV is off-the-shelf and can actually be obtained ''for free'' from past video streams. 
In this paper, we propose a novel task termed Beyond-FoV Estimation, aiming to exploit past visual cues and bidirectional break through the physical FoV of a camera.
We put forward a FlowLens architecture to expand the FoV by achieving feature propagation explicitly by optical flow and implicitly by a novel clip-recurrent transformer, 
which has two appealing features: 1) FlowLens comprises a newly proposed Clip-Recurrent Hub with 3D-Decoupled Cross Attention (DDCA) to progressively process global information accumulated in the temporal dimension. 2) A multi-branch Mix Fusion Feed Forward Network (MixF3N) is integrated to enhance the spatially-precise flow of local features. To foster training and evaluation, we establish KITTI360-EX, a dataset for outer- and inner FoV expansion. 
Extensive experiments on both video inpainting and beyond-FoV estimation tasks show that FlowLens achieves state-of-the-art performance.

### Demos

<p align="center">
    (Outer Beyond-FoV)
</p>
<p align="center">
    <img width="750" alt="Animation" src="assets/out_beyond.gif"/>
</p>
<br><br>

<p align="center">
    (Inner Beyond-FoV)
</p>
<p align="center">
    <img width="750" alt="Animation" src="assets/in_beyond.gif"/>
</p>
<br><br>

<p align="center">
    (Object Removal)
</p>
<p align="center">
    <img width="750" alt="Animation" src="assets/breakdance.gif"/>
</p>
<br><br>

### Dependencies
This repo has been tested in the following environment:
```angular2html
torch == 1.10.2
cuda == 11.3
mmflow == 0.5.2
```

### Usage
To train FlowLens(-S), use:
```angular2html
python train.py --config configs/KITTI360EX-I_FlowLens_small_re.json
```

To eval on KITTI360-EX, run:
```angular2html
python evaluate.py \
--model flowlens \
--cfg_path configs/KITTI360EX-I_FlowLens_small_re.json \
--ckpt release_model/FlowLens-S_Out_500000.pth --fov fov5
```

Turn on ```--reverse``` for test time augmentation (TTA).

Trun on ```--save_results``` to save your output.

### Pretrained Models
The pretrained model can be found there:
```angular2html
https://share.weiyun.com/6G6QEdaa
```

### KITTI360-EX for Beyond-FoV Estimation
The preprocessed KITTI360-EX can be downloaded from here:
```angular2html
https://share.weiyun.com/BReRdDiP
```

### Results
#### KITTI360EX-InnerSphere
| Method    | Test Logic | TTA | PSNR | SSIM    | VFID | Runtime (s/frame) |
| :--------- | :----------: | :----------: | :----------: | :--------: | :---------: | :------------: |
| _FlowLens-S (Paper)_ |_Beyond-FoV_|_wo_| _36.17_ | _0.9916_ | _0.030_ | _0.023_ |
| FlowLens-S (This Repo) |Beyond-FoV|wo| 37.31 | 0.9926 | 0.025 | **0.015** |
| FlowLens-S+ (This Repo) |Beyond-FoV|with| 38.36 | 0.9938 | 0.017 | 0.050 |
| FlowLens-S (This Repo) |Video Inpainting|wo| 38.01 | 0.9938 | 0.022 | 0.042 |
| FlowLens-S+ (This Repo) |Video Inpainting|with| **38.97** | **0.9947** | **0.015** | 0.142 |

| Method    | Test Logic | TTA | PSNR | SSIM    | VFID | Runtime (s/frame) |
| :--------- | :----------: | :----------: | :----------: | :--------: | :---------: | :------------: |
| _FlowLens (Paper)_ |_Beyond-FoV_|_wo_| _36.69_ | _0.9916_ | _0.027_ | _0.049_ |
| FlowLens (This Repo) |Beyond-FoV|wo| 37.65 | 0.9927 | 0.024 | **0.033** |
| FlowLens+ (This Repo) |Beyond-FoV|with| 38.74 | 0.9941 | 0.017 | 0.095 |
| FlowLens (This Repo) |Video Inpainting|wo| 38.38 | 0.9939 | 0.018 | 0.086 |
| FlowLens+ (This Repo) |Video Inpainting|with| **39.40** | **0.9950** | **0.015** | 0.265 |
###

#### KITTI360EX-OuterPinhole
| Method    | Test Logic | TTA | PSNR | SSIM    | VFID | Runtime (s/frame) |
| :--------- | :----------: | :----------: | :----------: | :--------: | :---------: | :------------: |
| _FlowLens-S (Paper)_ |_Beyond-FoV_|_wo_| _19.68_ | _0.9247_ | _0.300_ | _0.023_ |
| FlowLens-S (This Repo) |Beyond-FoV|wo| 20.41 | 0.9332 | 0.285 | **0.021** |
| FlowLens-S+ (This Repo) |Beyond-FoV|with| 21.30 | 0.9397 | 0.302 | 0.056 |
| FlowLens-S (This Repo) |Video Inpainting|wo| 21.69 | 0.9453 | **0.245** | 0.048 |
| FlowLens-S+ (This Repo) |Video Inpainting|with| **22.40** | **0.9503** | 0.271 | 0.146 |

| Method    | Test Logic | TTA | PSNR | SSIM    | VFID | Runtime (s/frame) |
| :--------- | :----------: | :----------: | :----------: | :--------: | :---------: | :------------: |
| _FlowLens (Paper)_ |_Beyond-FoV_|_wo_| _20.13_ | _0.9314_ | _0.281_ | _0.049_ |
| FlowLens (This Repo) |Beyond-FoV|wo| 20.85 | 0.9381 | 0.259 | **0.035** |
| FlowLens+ (This Repo) |Beyond-FoV|with| 21.65 | 0.9432 | 0.276 | 0.097 |
| FlowLens (This Repo) |Video Inpainting|wo| 22.23 | 0.9507 | **0.231** | 0.085 |
| FlowLens+ (This Repo) |Video Inpainting|with| **22.86** | **0.9543** | 0.253 | 0.260 |

Note that when using the ''Video Inpainting'' logic for output,
the model is allowed to use more reference frames from the future,
and each local frame is estimated at least twice,
thus higher accuracy can be obtained while result in slower inference speed,
and it is not realistic for real-world deployment.

### Citation

   If you find our paper or repo useful, please consider citing our paper:

   ```bibtex
   @article{shi2022flowlens,
  title={FlowLens: Seeing Beyond the FoV via Flow-guided Clip-Recurrent Transformer},
  author={Shi, Hao and Jiang, Qi and Yang, Kailun and Yin, Xiaoting and Wang, Kaiwei},
  journal={arXiv preprint arXiv:2211.11293},
  year={2022}
}
   ```
### Acknowledgement
This project would not have been possible without the following outstanding repositories:

[STTN](https://github.com/researchmm/STTN), [MMFlow](https://github.com/open-mmlab/mmflow)


### Devs
Hao Shi

### Contact
Feel free to contact me if you have additional questions or have interests in collaboration. Please drop me an email at haoshi@zju.edu.cn. =)
