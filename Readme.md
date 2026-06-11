# GMG-Net: Gradient-aware and Mid-frequency Guided Network for Low-light Image Enhancement

<p align="center">
  <img src="pic/Framework.jpg" width="900" alt="GMG-Net Framework">
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.knosys.2026.116390">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2026.116390-blue" alt="DOI">
  </a>
  <a href="https://www.sciencedirect.com/science/article/pii/S0950705126011160">
    <img src="https://img.shields.io/badge/Paper-ScienceDirect-red" alt="Paper">
  </a>
  <a href="https://github.com/Century-lu/GMG-Net">
    <img src="https://img.shields.io/badge/Code-GitHub-black" alt="Code">
  </a>
  <img src="https://img.shields.io/badge/Journal-Knowledge--Based%20Systems-green" alt="Journal">
  <img src="https://img.shields.io/badge/Year-2026-orange" alt="Year">
</p>

This repository contains the official implementation of **GMG-Net: Gradient-aware and Mid-frequency Guided network for low-light image enhancement**, published in **Knowledge-Based Systems**.

**Authors:** Huaping Zhou, Shiji Lu, Kelei Sun, Tao Wu, Bin Deng, and Man Chen

**Paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126011160) | [DOI](https://doi.org/10.1016/j.knosys.2026.116390)

## News

- **2026.06.08:** The paper is available online in *Knowledge-Based Systems*.
- **2026.06.03:** GMG-Net is accepted by *Knowledge-Based Systems*.
- Code, pretrained models, and evaluation scripts are released in this repository.

## Abstract

Low-light image enhancement aims to restore visually natural and structurally faithful images from scenes with insufficient and spatially varying illumination. Existing deep models often struggle with large illumination changes, which can lead to noise amplification, detail degradation, and color distortion.

GMG-Net addresses these issues with a **gradient-aware mechanism** and a **mid-frequency guided mechanism**. It introduces a Gradient-aware Contextual Attention Block (GCAB) to enhance blurred edge features, a Mid-frequency Guided Multi-scale Frequency Enhancement Module (MGMFE) to adaptively modulate low- and high-frequency information, and a Deformable Cross-branch Fusion Attention (DCFA) module to align and fuse HVI-space branch features. Experiments on paired, unpaired, and extremely dark datasets show that GMG-Net achieves strong visual quality with moderate parameters and FLOPs.

## Highlights

- **GCAB:** preserves edges and details while suppressing noise amplification through gradient-aware contextual attention.
- **MGMFE:** uses informative mid-frequency features to guide frequency-domain enhancement and balance texture recovery with noise suppression.
- **DCFA:** dynamically aligns and fuses the HV branch and I branch in HVI color space with deformable offset sampling.
- **Efficient enhancement:** achieves competitive or superior results on multiple low-light datasets with 3.82M parameters and 30.44G FLOPs.

## Method

### Overall Framework

<p align="center">
  <img src="pic/Framework.jpg" width="900" alt="Overall Framework">
</p>

### Motivation

<p align="center">
  <img src="pic/Multi-channel.jpg" width="900" alt="RGB multi-channel intensity distributions">
</p>

### Main Components

<p align="center">
  <img src="pic/Components.jpg" width="900" alt="GMG-Net Components">
</p>

### Gradient-aware Contextual Attention Block

<p align="center">
  <img src="pic/Gradient.jpg" width="900" alt="Gradient-aware Contextual Attention Block">
</p>

GCAB explicitly models gradient cues so that the network can better preserve edge and texture structures during illumination correction. In the paper, the Scharr operator is adopted for gradient-aware correction.

### Mid-frequency Guided Frequency Enhancement

<p align="center">
  <img src="pic/Spectral.jpg" width="900" alt="Mid-frequency Guided Frequency Enhancement">
</p>

MGMFE performs multi-scale frequency analysis and uses mid-frequency components as guidance. The module adaptively adjusts illumination-related low-frequency information and detail-related high-frequency information while retaining phase stability.

### Deformable Cross-branch Fusion Attention

<p align="center">
  <img src="pic/Module.jpg" width="800" alt="GMG-Net Module">
</p>

DCFA aligns and fuses features between the HV branch and I branch in HVI color space. Deformable sampling improves robustness to uneven illumination, feature distribution shifts, and local geometric variation.

## Visual Results

### LOL-v1

<p align="center">
  <img src="pic/LOL-V1.jpg" width="900" alt="LOL-v1 Results">
</p>

### LOL-v2 Real

<p align="center">
  <img src="pic/LOL-V2-Real.jpg" width="900" alt="LOL-v2 Real Results">
</p>

### LOL-v2 Synthetic

<p align="center">
  <img src="pic/LOL-V2-Syn.jpg" width="900" alt="LOL-v2 Synthetic Results">
</p>

### Unpaired Real-world Datasets

<p align="center">
  <img src="pic/Unpaired.jpg" width="900" alt="Unpaired Dataset Results">
</p>

### Sony Total Dark

<p align="center">
  <img src="pic/Sony-Total-Dark.jpg" width="900" alt="Sony Total Dark Results">
</p>

## Quantitative Results

### Paired Benchmarks

The following results are reported without the GT-Mean strategy.

| Dataset | Params (M) | FLOPs (G) | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| LOL-v1 | 3.82 | 30.44 | 24.741 | 0.866 | 0.083 |
| LOL-v2 Real | 3.82 | 30.44 | 23.696 | 0.871 | 0.109 |
| LOL-v2 Synthetic | 3.82 | 30.44 | 25.995 | 0.943 | 0.042 |

### Sony Total Dark

| Dataset | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Sony Total Dark | 22.552 | 0.677 | 0.449 |

### Unpaired Real-world Benchmarks

NIQE and BRISQUE are no-reference metrics. Lower values indicate better perceptual naturalness.

| Metric | DICM | LIME | MEF | NPE | VV | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NIQE | 3.69 | 4.14 | 3.55 | 3.75 | 3.23 | 3.67 |
| BRISQUE | 28.09 | 18.45 | 14.54 | 19.45 | 29.56 | 22.02 |

### Efficiency

Runtime is reported on an RTX 4090D with 256 x 256 input, averaged over 300 repetitions after 50 warmup iterations.

| Method | Params (M) | FLOPs (G) | Runtime (ms) | FPS |
| --- | ---: | ---: | ---: | ---: |
| GMG-Net | 3.82 | 30.44 | 33.36 | 29.97 |

## Ablation Study

### Core Components

The ablation below is reported on LOL-v2 Synthetic.

| GCAB | MGMFE | DCFA | PSNR | SSIM | LPIPS |
| --- | --- | --- | ---: | ---: | ---: |
| No | No | No | 24.915 | 0.932 | 0.055 |
| Yes | No | No | 25.168 | 0.936 | 0.049 |
| No | Yes | No | 25.635 | 0.939 | 0.047 |
| No | No | Yes | 25.588 | 0.939 | 0.047 |
| Yes | Yes | Yes | 25.995 | 0.943 | 0.042 |

### Frequency Guidance Strategy

| Strategy | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Low-frequency guidance | 24.521 | 0.912 | 0.058 |
| High-frequency guidance | 25.103 | 0.928 | 0.051 |
| Mid-frequency guidance (GMG-Net) | 25.995 | 0.943 | 0.042 |

### Gradient Operator

| Operator | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Roberts | 24.023 | 0.860 | 0.088 |
| Laplacian | 24.367 | 0.862 | 0.088 |
| Sobel | 24.267 | 0.863 | 0.090 |
| Prewitt | 24.465 | 0.861 | 0.087 |
| Scharr (GMG-Net) | 24.741 | 0.866 | 0.083 |

## Requirements

The code was tested with the following environment:

| Item | Version / Hardware |
| --- | --- |
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | 12.1 |
| PyTorch | 2.1.x |
| GPU | NVIDIA RTX 4090D 24GB |
| CPU | AMD EPYC 9754 |

The pinned `requirements.txt` follows the package versions used in this environment, including `torch==2.1.2+cu121` and `torchvision==0.16.2+cu121`.

Install dependencies with:

```bash
conda create -n gmgnet python=3.10
conda activate gmgnet
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your CUDA driver or PyTorch image already provides PyTorch 2.1.x with CUDA 12.1, you can install the remaining dependencies after removing the `torch` and `torchvision` lines from `requirements.txt`.

## Repository Structure

```text
GMG-Net/
  data/                 Dataset loaders and training options
  loss/                 Loss functions and image quality utilities
  net/                  Network implementation
  pic/                  Figures and visual results used in this README
  weights/              Pretrained checkpoints
  app.py                Gradio demo
  eval.py               Evaluation and inference script
  train.py              Training script
  measure.py            PSNR, SSIM, and LPIPS measurement
  measure_niqe_bris.py  NIQE and BRISQUE measurement
  requirements.txt      Python dependencies
```

## Pretrained Models

The released checkpoints should be placed in `weights/`.

| Checkpoint | Description |
| --- | --- |
| `weights/LOL_V1.pth` | Model for LOL-v1 |
| `weights/LOL_V2_Real.pth` | Model for LOL-v2 Real |
| `weights/LOL_V2_Syn.pth` | Model for LOL-v2 Synthetic |
| `weights/SID.pth` | Model for Sony Total Dark |

## Dataset Preparation

Please organize datasets following the default paths in `data/options.py`.

### LOL-v1

```text
datasets/LOLdataset/
  our485/
    low/
    high/
  eval15/
    low/
    high/
```

### LOL-v2 Real

```text
datasets/LOLv2/Real_captured/
  Train/
    Low/
    Normal/
  Test/
    Low/
    Normal/
```

### LOL-v2 Synthetic

```text
datasets/LOLv2/Synthetic/
  Train/
    Low/
    Normal/
  Test/
    Low/
    Normal/
```

### Sony Total Dark

```text
datasets/Sony_total_dark/
  train/
    short/
    long/
  eval/
    short/
    long/
```

### Unpaired Datasets

```text
datasets/Five unpaired datasets/
  DICM/
  LIME/
  MEF/
  NPE/
  VV/
```

## Training

Before training, select one dataset switch in `data/options.py`. For example, to train on LOL-v1:

```python
parser.add_argument('--lol_v1', type=bool, default=True)
parser.add_argument('--lolv2_real', type=bool, default=False)
parser.add_argument('--lolv2_syn', type=bool, default=False)
parser.add_argument('--SID', type=bool, default=False)
```

Then run:

```bash
python train.py
```

The paper uses the following training settings:

| Dataset | Patch size | Epochs | Batch size |
| --- | ---: | ---: | ---: |
| LOL-v1 | 128 x 128 | 1000 | 8 |
| LOL-v2 Synthetic | 128 x 128 | 1000 | 8 |
| Sony Total Dark | 128 x 128 | 1000 | 8 |
| LOL-v2 Real | 256 x 256 | 800 | 8 |

Checkpoints will be saved to:

```text
weights/train/
```

Validation results and metric logs will be saved to:

```text
results/
```

## Evaluation

Run evaluation on paired datasets:

```bash
python eval.py --lol
python eval.py --lol_v2_real
python eval.py --lol_v2_syn
```

Run evaluation on unpaired datasets:

```bash
python eval.py --unpaired --DICM
python eval.py --unpaired --LIME
python eval.py --unpaired --MEF
python eval.py --unpaired --NPE
python eval.py --unpaired --VV
```

For custom low-light images:

```bash
python eval.py --unpaired --custome --custome_path path/to/your/images
```

Enhanced images will be saved to `output/`.

> Note: if you use different checkpoint filenames, please update the corresponding `weight_path` in `eval.py`.

## Gradio Demo

Launch the local demo with:

```bash
python app.py
```

For CPU-only inference:

```bash
python app.py --cpu
```

The demo runs on:

```text
http://127.0.0.1:7862
```

## Measurement

Calculate full-reference metrics:

```bash
python measure.py
```

Calculate no-reference metrics:

```bash
python measure_niqe_bris.py
```

Please update image paths inside the scripts according to your local dataset and output directories.

## Citation

If you find our work useful for your research, please cite our paper:

```bibtex
@article{Lu2026GMG,
  title = {GMG-Net: Gradient-aware and Mid-frequency Guided network for low-light image enhancement},
  journal = {Knowledge-Based Systems},
  volume = {348},
  pages = {116390},
  year = {2026},
  issn = {0950-7051},
  doi = {https://doi.org/10.1016/j.knosys.2026.116390},
  url = {https://www.sciencedirect.com/science/article/pii/S0950705126011160},
  author = {Huaping Zhou and Shiji Lu and Kelei Sun and Tao Wu and Bin Deng and Man Chen},
}
```

## License

This project is released under the license provided in `LICENSE`.

## Contact

If you have any questions, please open an issue in this repository.

## Acknowledgement

We thank the authors of public low-light image enhancement datasets and related open-source projects for their valuable contributions to the community.
