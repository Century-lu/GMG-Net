# GMG-Net: Gradient-aware and Mid-frequency Guided Network for Low-light Image Enhancement

<p align="center">
  <img src="pic/Framework.jpg" width="850" alt="GMG-Net Framework">
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.knosys.2026.116390">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2026.116390-blue" alt="DOI">
  </a>
  <a href="https://www.sciencedirect.com/science/article/pii/S0950705126011160">
    <img src="https://img.shields.io/badge/Paper-ScienceDirect-red" alt="Paper">
  </a>
  <img src="https://img.shields.io/badge/Journal-Knowledge--Based%20Systems-green" alt="Journal">
  <img src="https://img.shields.io/badge/Year-2026-orange" alt="Year">
</p>

This repository provides the official implementation of **GMG-Net: Gradient-aware and Mid-frequency Guided network for low-light image enhancement**, published in **Knowledge-Based Systems**.

**Authors:** Huaping Zhou, Shiji Lu, Kelei Sun, Tao Wu, Bin Deng, and Man Chen

**Paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126011160) | [DOI](https://doi.org/10.1016/j.knosys.2026.116390)

## News

- **2026:** GMG-Net is published in *Knowledge-Based Systems*.
- Code, testing scripts, and pretrained checkpoints are available in this repository.

## Introduction

Low-light image enhancement aims to recover high-quality images from under-exposed scenes. This task is challenging because low-light images usually suffer from weak illumination, color distortion, detail loss, noise amplification, and degraded structural information.

GMG-Net is designed to enhance low-light images by jointly exploiting **gradient-aware structural guidance** and **mid-frequency feature guidance**. The network focuses on preserving edge structures while restoring informative texture details, leading to visually natural and detail-preserving enhancement results.

## Highlights

- **Gradient-aware enhancement:** uses gradient information to preserve structure and edge details.
- **Mid-frequency guidance:** strengthens useful texture restoration and suppresses unwanted artifacts.
- **Effective low-light restoration:** improves brightness, contrast, color fidelity, and perceptual quality.
- **Broad evaluation:** supports paired and unpaired low-light enhancement benchmarks.

## Method

### Overall Framework

<p align="center">
  <img src="pic/Framework.jpg" width="900" alt="Overall Framework">
</p>

### Main Components

<p align="center">
  <img src="pic/Components.jpg" width="900" alt="Components">
</p>

### Gradient-aware Guidance

<p align="center">
  <img src="pic/Gradient.jpg" width="900" alt="Gradient-aware Guidance">
</p>

### Mid-frequency and Spectral Guidance

<p align="center">
  <img src="pic/Spectral.jpg" width="900" alt="Spectral Guidance">
</p>

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

### Unpaired Datasets

<p align="center">
  <img src="pic/Unpaired.jpg" width="900" alt="Unpaired Dataset Results">
</p>

### Sony Total Dark

<p align="center">
  <img src="pic/Sony-Total-Dark.jpg" width="900" alt="Sony Total Dark Results">
</p>

## Requirements

The code was tested with the following main dependencies:

- Python 3.7
- PyTorch 1.13.1
- torchvision 0.14.1
- CUDA-enabled GPU is recommended

Install dependencies with:

```bash
conda create -n gmgnet python=3.7
conda activate gmgnet
pip install -r requirements.txt
```

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
  author = {Huaping Zhou and Shiji Lu and Kelei Sun and Tao Wu and Bin Deng and Man Chen}
}
```

## License

This project is released under the license provided in `LICENSE`.

## Contact

If you have any questions, please open an issue in this repository.

## Acknowledgement

We thank the authors of public low-light image enhancement datasets and related open-source projects for their valuable contributions to the community.
