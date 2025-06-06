<h1 align="center">M3CAD: Towards Generic Cooperative Autonomous Driving Benchmark</h1>

<div align="center">
  <video width="800" controls>
    <source src="https://zhumorui.github.io/m3cad/video/m3cad_demo_video_h264.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## Overview
M³CAD is currently the most comprehensive benchmark for both single-vehicle and cooperative autonomous driving research, supporting multiple tasks like object detection and tracking, mapping, motion forecasting, occupancy, and path planning, while supporting more realistic vehicle movements and interactions in complex environments.


## 🚀 Getting Started

This project follows the structure of [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) and extends it to support cooperative autonomous driving tasks.

- [Installation](docs/INSTALL.md)
- [Data Preparation](docs/DATA_PREP.md)
- [Training and Evaluation](docs/TRAIN_EVAL.md)


## 📊 Results Visualization

We provide two visualization tools for both single-vehicle and cooperative autonomous driving tasks. 

1. Basic Visualization (BEV)
```bash
python m3cad/uniad/analysis_tools/visualize/run.py \
    --dataroot data/m3cad_carla_ue5 \
    --version v1.0-test \
    --predroot output/results.pkl \
    --out_folder output_vis \
    --demo_video output.mp4 \
    --project_to_cam # optional
```

2. We extend [Rerun Example](https://github.com/rerun-io/rerun/tree/docs-latest/examples/python/nuscenes_dataset) to visualize M3CAD dataset including lidar, images, ground truth bounding boxes and predicted bounding boxes in 3D space.
![rerun_m3cad_vis](assets/rerun_m3cad_vis.png)
Make sure you have installed rerun-sdk by `pip install rerun-sdk`.


```bash
python m3cad/uniad/analysis_tools/visualize/rerun_visualizer.py \
    --root-dir data/m3cad_carla_ue5 \
    --scene-name '2025_06_24_18_33_22_60', '2025_06_24_18_33_22_51', '2025_06_24_18_33_22_75'  \ # support multiple scenes
    --serve \ 
    --dataset-version v1.0-test \
    --seconds 20 
```

## TODO
- [ ] support VAD
- [ ] support UniV2X



## 🙏 Citation 

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{zhu2025m3cad,
      title={M3CAD: Towards Generic Cooperative Autonomous Driving Benchmark}, 
      author={Morui Zhu and Yongqi Zhu and Yihao Zhu and Qi Chen and Deyuan Qu and Song Fu and Qing Yang},
      year={2025},
      eprint={2505.06746},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Acknowledgements

We thank the authors of the following repositories for their contributions to this project:
* [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)
* [UniAD](https://github.com/OpenDriveLab/UniAD)
* [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
* [NuScenes](https://github.com/nutonomy/nuscenes-devkit)
* [F-COOPER](https://github.com/Aug583/F-COOPER)