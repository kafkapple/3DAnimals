# 3D Animals Codebase

## Quick Start: Mouse 6-View Experiment

```bash
# Step 1: 데이터셋 생성 (crop 전처리 자동 적용)
python scripts/setup_multiview_fauna_dataset.py \
    --session_dir /path/to/sam3d_gui/sessions/mouse_batch_XXXXXX \
    --output_dir data/fauna/mouse_6view_posesplatter \
    --mode pose_splatter_debug

# Step 2: Debug 학습 (~15분) - 먼저 검증
python run.py --config-name train_fauna_mouse_6view_debug

# Step 3: 결과 확인
ls results/fauna_mouse_6view_debug/

# Step 4: Finetune 학습 (~2-3시간)
python run.py --config-name train_fauna_mouse_6view_finetune
```

---

https://github.com/user-attachments/assets/c0dbb792-2ce8-424c-98db-8c8e6a3e2f29


This repository contains the unified codebase for several projects on articulated 3D animal reconstruction and motion generation, including:

- [MagicPony: Learning Articulated 3D Animals in the Wild](https://3dmagicpony.github.io/) (CVPR 2023) [![arXiv](https://img.shields.io/badge/arXiv-2211.12497-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2211.12497) - a category-specific single-image 3D animal reconstruction model
- [Learning the 3D Fauna of the Web](https://kyleleey.github.io/3DFauna/) (CVPR 2024) [![arXiv](https://img.shields.io/badge/arXiv-2401.02400-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.02400) - a pan-category single-image 3D quadruped reconstruction model
- [Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos](https://keqiangsun.github.io/projects/ponymation/) (ECCV 2024) [![arXiv](https://img.shields.io/badge/arXiv-2312.13604-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.13604) - an articulated 3D animal motion generative model


## 📚 Documentation

**NEW: Comprehensive Guides Available!**

- **[Dataset Preparation Guide](./docs/FAUNA_DATASET_PREPARATION_GUIDE.md)** - How to prepare your own animal datasets
  - ✅ No uniform image size required (auto-resized)
  - ✅ Auto-mask generation with SAM/GrabCut
  - ✅ Complete Python scripts included
  - ✅ Minimal requirements: Images + Masks

- **[System Comprehensive Guide](./docs/reports/251121_3danimals_system_comprehensive_guide.md)** - Complete system overview
  - Dataset structure and requirements
  - Training workflows (debug-first principle)
  - Inference and visualization
  - Adding new animal datasets (step-by-step)
  - Troubleshooting common issues

## Installation
See [INSTALL.md](./INSTALL.md).

## Data
### Tetrahedral Grids
We adopt the hybrid SDF-mesh representation from [DMTet](https://research.nvidia.com/labs/toronto-ai/DMTet/) to represent the 3D shape of the animals. It uses tetrahedral grids to extract meshes from underlying SDF representation.

Download the pre-computed tetrahedral grids:
```shell
cd data/tets
sh download_tets.sh
```

### Datasets
Download the preprocessed datasets for each project using the download scripts provided in `data/`. All datasets should be downloaded in the same directory as the download script, for example:
```shell
cd data/magicpony
sh download_horse_combined.sh
```
See the notes [below](#data-1) for the details of each dataset.


## Pretrained Models
The pretrained models can be downloaded using the scripts provided in `results/`. All pretrained models should be downloaded in the same directory as the download script, for example:
```shell
cd results/magicpony
sh download_pretrained_horse.sh
```


## Run
Once the data is prepared, both training and inference of all models can be executed using a single command:
```shell
python run.py --config-name CONFIG_NAME
```
or for training with DDP using multiple GPUs:
```shell
accelerate launch --multi_gpu run.py --config-name CONFIG_NAME
```
`CONFIG_NAME` can be any of the configs specified in `config/`, e.g., `test_magicpony_horse` or `train_magicpony_horse`.

### Testing using the Pretrained Models
The simplest use case is to test the pretrained models on test images. To do this, use the configs in `configs/` that start with `test_*`. Open the config files to check the details, including the path of the test images.

Note that only the RGB images are required during testing. The DINO features are not required. The mask images are only required if you wish to finetune the texture with higher precision for visualization (see [below](#test-time-texture-finetuning)).

When running the command with the default test configs, it will automatically save some basic visualizations, including the reconstructed views and 3D meshes. For more advanced and customized visualizations, use `scripts/visualize_results.py` as explained [below](#visualization).

### Training
See the instructions for each specific model [below](#training-1).

### Visualization
We provide some scripts that we used to generate the visualizations on our project pages ([MagicPony](https://3dmagicpony.github.io/), [3D-Fauna](https://kyleleey.github.io/3DFauna/), [Ponymation](https://keqiangsun.github.io/projects/ponymation/)). To render such visualizations, simply run the following command with the proper test config, e.g.:
```shell
python visualization/visualize_results.py --config-name test_magicpony_horse
```

For 3D-Fauna, use `visualize_results_fauna.py` instead:
```shell
python visualization/visualize_results_fauna.py --config-name test_fauna
```

Check the `#Visualization` section in the config files for specific visualization configurations.

#### Rendering Modes
The visualization script supports the following `render_modes`, which can be specified in the config:
- `input_view`: image rendered from the input viewpoint of the reconstructed textured mesh, shading map, gray shape visualization.
- `other_views`: image rendered from 12 viewpoints rotating around the vertical axis of the reconstructed textured mesh, gray shape visualization.
- `rotation`: video rendered from continuously rotating viewpoints around the vertical axis of the reconstructed textured mesh, gray shape visualization.
- `animation` (only supported for quadrupeds): two videos rendered from both a side viewpoint and continuously rotating viewpoints of the reconstructed textured mesh animated by interpolating a sequence of pre-configured articulation parameters. `arti_param_dir` can be set to `./visualization/animation_params` which contains a sequence of pre-computed keyframe articulation parameters.
- `canonicalization` (only supported for quadrupeds): video of the reconstructed textured mesh morphing from the input pose to a pre-configured canonical pose.

#### Test-time Texture Finetuning
To enable texture finetuning at test time, set `finetune_texture: true` in the config, and (optionally) adjust the number of finetune iterations `finetune_iters` and learning rate `finetune_lr`.

For more precise texture optimization, provide instance masks in the same folder as `*_mask.png`. Otherwise, the background pixels might be pasted onto the object if shape predictions are not perfectly aligned.


## MagicPony [![arXiv](https://img.shields.io/badge/arXiv-2211.12497-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2211.12497)
[MagicPony](https://3dmagicpony.github.io/) learns a category-specific model for single-image articulated 3D reconstruction of an animal species.

### Data
We trained MagicPony models on image collections of horses, giraffes, zebras, cows, and birds. The data download scripts in `data/magicpony` provide access to the following preprocessed datasets:
- `horse_videos` and `bird_videos` were released by [DOVE](https://dove3d.github.io/).
- `horse_combined` consists of `horse_videos` and additional images selected from [Weizmann Horse Database](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database), [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/), and [Horse-10](http://www.mackenziemathislab.org/horse10).
- `giraffe_coco`, `zebra_coco` and `cow_coco` are filtered subsets of the [COCO dataset](https://cocodataset.org/).

### Training
To train MagicPony on the provided horse dataset or bird dataset from scratch, simply use the training configs: `train_magicpony_horse` or `train_magicpony_bird`, e.g.:
```shell
python run.py --config-name train_magicpony_horse
```
For multi-GPU training, use the `accelerator launch` command, e.g.:
```shell
accelerator launch --multi_gpu run.py --config-name train_magicpony_horse
```

To train it on the provided giraffe, zebra, or cow datasets, which are much smaller, please finetune from a _pretrained_ horse model using the finetuning configs: `finetune_magicpony_giraffe`, `finetune_magicpony_zebra`, or `finetune_magicpony_cow`.


## 3D-Fauna [![arXiv](https://img.shields.io/badge/arXiv-2401.02400-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.02400)
[3D-Fauna](https://kyleleey.github.io/3DFauna/) learns a pan-category model for single-image articulated 3D reconstruction of any quadruped species.

### Data
The `Fauna Dataset`, which can be downloaded via the script `data/fauna/download_fauna_dataset.sh`, consists of video frames and images sourced from the Internet, as well as images from [DOVE](https://dove3d.github.io/), [APT-36K](https://github.com/pandorgan/APT-36K), [Animal3D](https://xujiacong.github.io/Animal3D/), and [Animals-with-Attributes](https://cvml.ista.ac.at/AwA2/).

#### Adding Your Own Animal Dataset

**Quick Start**: See [Dataset Preparation Guide](./docs/FAUNA_DATASET_PREPARATION_GUIDE.md) for detailed instructions.

**Minimal Requirements**:
```
data/fauna/Fauna_dataset/large_scale/my_animal/
└── train/
    └── seq_000/
        ├── 0000000_rgb.png    # Image (any size, auto-resized to 256×256)
        ├── 0000000_mask.png   # Mask (binary: 0=background, 255=foreground)
        ├── 0000001_rgb.png
        ├── 0000001_mask.png
        └── ...
```

**Key Facts**:
- ✅ **Image size**: Any size (auto-resized), no need for uniform sizes
- ✅ **Mask generation**: Use SAM, GrabCut, or manual annotation (scripts provided)
- ✅ **Auto-generation**: `box.txt` and `metadata.json` can be auto-generated from masks
- ✅ **Minimum data**: 50-100 images recommended (more is better)

**Quick Setup**:
```bash
# 1. Use provided script to convert your images
python scripts/convert_to_fauna_format.py \
  --input_dir ~/my_animal_images \
  --output_dir data/fauna/Fauna_dataset/large_scale \
  --animal_name my_animal \
  --auto_generate_masks True  # Auto-generate masks with SAM/GrabCut

# 2. Copy config templates
cp config/dataset/fauna_new_animal_template.yaml config/dataset/fauna_my_animal.yaml
cp config/model/fauna_new_animal_template.yaml config/model/fauna_my_animal.yaml
cp config/train_fauna_new_animal_template.yaml config/train_fauna_my_animal.yaml

# 3. Edit configs (adjust animal size parameters)
# See templates for detailed parameter guidelines

# 4. Run debug training first (15-30 min)
python run.py --config-name train_fauna_my_animal_debug

# 5. Run full training (10-12 hours on RTX 3060)
python run.py --config-name train_fauna_my_animal
```

### Training
To train 3D-Fauna on the Fauna Dataset, simply run:
```shell
python run.py --config-name train_fauna
```

**Debug-First Principle**: Always run debug mode first before full training!
```shell
# Debug mode (5K iterations, ~15-30 min)
python run.py --config-name train_fauna_mouse_debug

# Full training (200K iterations, ~10-12 hours)
python run.py --config-name train_fauna_mouse_from_scratch
```


## Ponymation [![arXiv](https://img.shields.io/badge/arXiv-2312.13604-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.13604)
[Ponymation](https://keqiangsun.github.io/projects/ponymation/) learns a generative model of articulated 3D motions of an animal species.

### Data
Dataset can be downloaded via the script `data/ponymation/download_ponymation_dataset.sh`, including video data of horse, cow, giraffe, and zebra.

### Training
Ponymation is trained in two stages. In the first stage, we pretrain a 3D reconstruction model that takes in a sequence of frames and reconstructs a sequence of articulated 3D shapes of the animal. This stage can be initiated using the stage 1 config `train_ponymation_horse_stage1`:
```shell
python run.py --config-name train_ponymation_horse_stage1
```

After this video reconstruction model is pretrained, we then train a generative model of the articulated 3D motions in the second stage, using the stage 2 config `train_ponymation_horse_stage2`:
```shell
python run.py --config-name train_ponymation_horse_stage2
```


## 4D Reconstruction of Animal Video (WIP)
This repo contains code to reconstruct 4D animal videos, i.e., a sequence of articulated 3D shapes of an animal in a video, using 3D-Fauna as the backbone. The code works on animal video dataset processed by [Animal Video Processing repo](https://github.com/briannlongzhao/Animal-Video-Processing).

```shell
python run_4d_reconstruction.py
```



## Citation
If you use this repository or find the papers useful for your research, please consider citing the following publications, as well as the original publications of the datasets used:
```
@InProceedings{wu2023magicpony,
  title     = {{MagicPony}: Learning Articulated 3D Animals in the Wild},
  author    = {Wu, Shangzhe and Li, Ruining and Jakab, Tomas and Rupprecht, Christian and Vedaldi, Andrea},
  booktitle = {CVPR},
  year      = {2023}
}
```

```
@InProceedings{li2024fauna,
  title     = {Learning the 3D Fauna of the Web},
  author    = {Li, Zizhang and Litvak, Dor and Li, Ruining and Zhang, Yunzhi and Jakab, Tomas and Rupprecht, Christian and Wu, Shangzhe and Vedaldi, Andrea and Wu, Jiajun},
  booktitle = {CVPR},
  year      = {2024}
}
```

```
@InProceedings{sun2024ponymation,
  title     = {{Ponymation}: Learning Articulated 3D Animal Motions from Unlabeled Online Videos},
  author    = {Sun, Keqiang and Litvak, Dor and Zhang, Yunzhi and Li, Hongsheng and Wu, Jiajun and Wu, Shangzhe},
  booktitle = {ECCV},
  year      = {2024}
}
```

## FAQ

### Q: Do all images need to be the same size?

**A**: No! Images are automatically resized to 256×256 during loading. You can have images of any size (1920×1080, 640×480, etc.) in the same dataset.

### Q: What if I don't have masks?

**A**: Masks are required, but can be auto-generated using:
- **SAM (Segment Anything)** - Best quality
- **GrabCut** - Good balance
- **Threshold** - Simple but effective

See [Dataset Preparation Guide](./docs/FAUNA_DATASET_PREPARATION_GUIDE.md) for auto-generation scripts.

### Q: What's the minimum dataset size?

**A**:
- **Minimum**: 30-50 images (may work but lower quality)
- **Recommended**: 100-200 images (good quality)
- **Ideal**: 200+ images (best quality)

Quality also depends on diversity (various poses/viewpoints).

### Q: How long does training take?

**A**: On RTX 3060 12GB:
- **Debug mode**: 5K iters, ~15-30 minutes
- **Few-shot**: 50-100K iters, ~2-5 hours
- **Full training**: 200K iters, ~10-12 hours

### Q: Can I use videos?

**A**: Yes! Extract frames using the provided scripts:
```python
python scripts/extract_frames_from_video.py --video_path ~/animal.mp4 --fps 2
```
Then follow the normal dataset preparation workflow.

### Q: What GPU do I need?

**A**:
- **RTX 3060 12GB**: Small animals (grid_res=64, batch_size=4-6)
- **RTX 3090 24GB**: Medium animals (grid_res=128, batch_size=8-12)
- **A100 40GB**: Large animals (grid_res=256, batch_size=12-16)

See [System Guide](./docs/reports/251121_3danimals_system_comprehensive_guide.md#appendix-b-gpu-memory-requirements) for detailed memory requirements.

## TODO

- [ ] Ponymation dataset update
- [ ] Data processing script
- [ ] Metrics evaluation script
- [x] Dataset preparation guide
- [x] System comprehensive documentation
