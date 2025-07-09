# Continual SSL→CL Pipeline for Chest Radiograph Classification

A modular PyTorch Lightning implementation of a self-supervised DINO pretraining on ChestMNIST, followed by finetuning on Shenzhen and continual learning on Montgomery & MIAS using replay + EWC.

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
  - [Pretraining](#pretraining)  
  - [Finetuning](#finetuning)  
  - [Continual Learning](#continual-learning)  
- [Results](#results)  
- [Architecture](#architecture)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

This repository provides:

1. **Self-Supervised Pretraining**  
   DINO loss with center update on ChestMNIST (grayscale → 3-channel).  
2. **Finetuning**  
   ViT-Tiny on Shenzhen for binary classification (normal vs. abnormal).  
3. **Continual Learning**  
   Task sequence: Shenzhen → Montgomery → MIAS, using replay buffer and Elastic Weight Consolidation (EWC) to mitigate forgetting.

All code is organized in reusable modules, driven by configurable YAML files.

---

## Features

- **DINOLoss** with softmax sharpening and center EMA  
- Student/teacher ViT-Tiny backbones via timm  
- Automatic teacher momentum update  
- PyTorch Lightning DataModules for each dataset  
- Early stopping & ModelCheckpoint in finetuning  
- Replay buffer + EWC in continual phases  
- Clear separation of scripts, configs, notebooks, and source modules

---

## Installation

```
conda env create -f environment.yml
conda activate ssl_cl_chest
```

---

## Configuration

Edit `configs/*.yaml` to point to your local dataset and adjust:

* `data_path`: location of ChestMNIST `.npz` or CXR CSV + images
* `batch_size`, `learning_rate`, `max_epochs`, etc.
* Checkpoint save directories

---

## Usage

### Pretraining

```
bash scripts/run_pretrain.sh --config configs/pretrain.yaml
```

### Finetuning

```
bash scripts/run_finetune.sh --config configs/finetune_shenzhen.yaml
```

### Continual Learning

```
# Shenzhen → Montgomery
bash scripts/run_continual.sh --config configs/continual_montgomery.yaml

# Shanghai → Montgomery → MIAS
bash scripts/run_continual.sh --config configs/continual_mias.yaml
```

---

## Results

| Stage                   | Dataset    | Metric |  Value |
| ----------------------- | ---------- | -----: | -----: |
| **Finetuning**          | Shenzhen   |    AUC |  0.879 |
| **Continual (Final)**   | Shenzhen   |    AUC |  0.760 |
|                         | Montgomery |    AUC |  0.724 |
|                         | MIAS       |    AUC |  0.696 |
| **Backward Transfer Δ** | Shenzhen   |  Δ AUC | +0.002 |

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for:

* Model block diagrams
* DINO loss & EMA updates
* Replay buffer & EWC mechanics

---

## Contributing

Feel free to open issues or pull requests. For major changes, please discuss via issue first.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
