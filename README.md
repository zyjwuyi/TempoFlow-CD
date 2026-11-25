# TempoFlow-CD

**TempoFlow-CD: Flow Matching With Local Time Conditioning for Remote Sensing Change Detection**

Yijun Zhou, Chen Lyu, Zhihui You, and Shuting He

TempoFlow-CD models remote sensing change detection as continuous evidence-to-geometry transport. Nuisance-Invariant Geometry Flow (NIGF) aligns nuisance-varied observations with shared mask-distance geometry endpoints. Change-Adaptive Spatial Clock (CASC) introduces geometry-supervised local phase conditioning and boundary-guided velocity refinement on a shared integration grid.

This repository contains the model and training/validation code supplied with the paper's supplementary material. The implementation retains the original class names `NuisanceOrbitFlowRWKV` and `NIGFCASCRWKV`.

## Repository structure

| Path | Contents |
|---|---|
| `train.py` | Single-GPU/distributed training, validation, EMA, and checkpoint resume |
| `model/NuisanceOrbitFlowRWKV.py` | Integrated NIGF-CASC model |
| `model/nigf.py`, `model/orbit_evidence.py` | Nuisance-invariant geometry flow and evidence encoding |
| `model/flow_rwkv.py`, `model/flow_solver.py` | Spatial clock, vector field, and Euler integration |
| `model/geometry_field.py` | Geometry encoder and decoder |
| `model/VRWKV.py`, `model/cuda/` | Visual RWKV backbone and CUDA extension |
| `utils/` | Dataset loading, geometry targets, nuisance transforms, losses, and checkpoints |

## Installation

Use Linux, Python 3.10 or newer, an NVIDIA GPU, a CUDA-enabled PyTorch installation, and a compatible CUDA toolkit with `nvcc` and a C++ compiler. The training code uses POSIX filesystem operations. Set `CUDA_HOME` if the toolkit is not in the default location. The RWKV CUDA extension is compiled when the backbone is loaded.

Install a CUDA-enabled PyTorch/torchvision build appropriate for your environment, then install the remaining dependencies:

```bash
git clone https://github.com/zyjwuyi/TempoFlow-CD.git
cd TempoFlow-CD
pip install -r requirements.txt
```

## Dataset preparation

Use matching filenames for paired RGB images and binary masks. Prepare 256 x 256 images for the default configuration; the loader does not automatically resize them. Masks use 0 for background and 255 for foreground.

```text
dataset/
  A/
  B/
  label/
  list/
    train.txt
    val.txt
```

Each list contains one filename per line. The directory layout `dataset/{train,val}/{A,B,label}` is also supported.

## Training

Train on a single GPU:

```bash
python train.py --data_dir /path/to/dataset --pretrain_dir /path/to/encoder.pth --output_dir outputs/run
```

The encoder checkpoint is optional: omit `--pretrain_dir` to train without loading pretrained encoder weights. By default, the training entry point enables NIGF and CASC with four Euler steps. Training uses 200 epochs, batch size 16, AdamW, one warmup epoch, BF16 mixed precision, and EMA with decay 0.999. Encoder and remaining parameter learning rates are both 1e-5, matching the manuscript. Checkpoints are selected on the validation split at threshold 0.5. These are the supplied entry-point defaults, not a claim that one command reproduces every experiment in the paper.

Train with two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --batch_size 8 --data_dir /path/to/dataset --pretrain_dir /path/to/encoder.pth --output_dir outputs/run
```

Resume a run:

```bash
python train.py --data_dir /path/to/dataset --resume outputs/run/last.pth --output_dir outputs/run
```

Use the same model and optimization arguments when resuming, and retain the run's `metrics.csv` and companion `last_ema.pth` checkpoint. Batch size is per process with `torchrun`; use `16 / world_size` per process to retain the paper's total batch size of 16 (the two-GPU example uses 8). Set `--solver_steps` to 1, 2, 4, 6, or 8. Training options are listed by `python train.py --help`.

## Validation and outputs

The training entry point evaluates the validation split and records precision, recall, F1, IoU, and accuracy. Its output directory contains `config.json`, `metrics.csv`, `last.pth`, and the validation-selected `best_f1.pth` and `best_iou.pth` checkpoints. When EMA is enabled, it also saves `last_ema.pth`.

The supplied package does not include datasets, pretrained encoder weights, trained checkpoints, or standalone test-set, boundary-metric, and robustness evaluation scripts. The released loader and training entry point operate on paired RGB images with binary change masks. Additional protocols reported in the manuscript require their corresponding data preparation and evaluation code.

## Release notes

This revision aligns explicit manuscript settings: batch size 16, learning rate 1e-5 for both parameter groups, CASC enabled and clock loss weight 0.1 across model/argument defaults, and smooth-L1 supervision of raw distance outputs. The original license is retained. Settings not specified by the manuscript (including EMA, warmup, mixed precision, and BCE positive weight) retain the supplied training defaults. See `PAPER_ALIGNMENT.txt` for the parameter audit and validation scope. Full CUDA training and reproduction of reported results have not been run.

## License

The supplementary package includes the [Apache License 2.0](LICENSE), preserved in this repository.
