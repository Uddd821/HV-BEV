<div align="center">   
  
# HV-BEV: Decoupling Horizontal and Vertical Feature Sampling for Multi-View 3D Object Detection
</div>

> - [Paper in arXiv](https://arxiv.org/pdf/2412.18884)
>
> - # Abstract
> - The application of vision-based multi-view environmental perception system has been increasingly recognized in autonomous driving technology, especially the BEV-based models. Current state-of-the-art solutions primarily encode image features from each camera view into the BEV space through explicit or implicit depth prediction. However, these methods often focus on improving the accuracy of projecting 2D features into corresponding depth regions, while overlooking the structured correlations among different parts of objects in 3D space and the fact that different categories of objects often occupy distinct local height ranges. For example, trucks appear at higher elevations, whereas traffic cones are near the ground. In this work, we propose a novel approach that decouples feature sampling in the BEV grid queries paradigm into Horizontal feature aggregation and Vertical adaptive height-aware reference point sampling (HV-BEV), aiming to improve both the aggregation of objects' complete information and awareness of diverse objects' height distribution. Specifically, a set of relevant neighboring points is dynamically constructed for each 3D reference point on the ground-aligned horizontal plane, enhancing the association of the same instance across different BEV grids, especially when the instance spans multiple image views around the vehicle. Additionally, instead of relying on uniform sampling within a fixed height range, we introduce a height-aware module that incorporates historical information, enabling the reference points to adaptively focus on the varying heights at which objects appear in different scenes. Extensive experiments validate the effectiveness of our proposed method, demonstrating its superior performance over the baseline across the nuScenes dataset. Moreover, our best-performing model achieves a remarkable 50.5\% mAP and 59.8\% NDS on the nuScenes testing set.
>
> - # Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md)

> - # Logs
| Backbone | Method | Lr Schd | NDS | mAP | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: |
| R101-DCN | HV-BEV_small | 24ep | 35.9 | 25.7 |[config](projects/configs/bevformer/bevformer_small.py) | [log](log/Res101_small.log) |
| VoV-99| HV-BEV-base| 24ep | 35.4 | 25.2 |[config](projects/configs/bevformer/bevformer_base.py) | [log](log/vov_base.log)| 
