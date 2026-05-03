# xai-proto-vision

Coursework for Autonomous Driving Systems. Implementation and comparison of four
prototype-based self-interpretable neural network architectures for fine-grained
image classification in PyTorch: **ProtoPNet**, **ProtoTree**, **TesNet**, and **PIPNet**.

---

## Overview

The central question this project investigates is the accuracy-interpretability
tradeoff in deep learning: can a model that explains its own decisions come close
to matching a black-box baseline? Each of the four methods approaches this from a
different angle, and each one builds on the limitations of the previous:

| Method | Year | Core Idea | Key Limitation |
|---|---|---|---|
| [ProtoPNet](https://arxiv.org/abs/1806.10574) | 2019 | Class-specific prototype patches in latent space | Prototypes are abstract, class-specific — redundant |
| [ProtoTree](https://arxiv.org/abs/2012.02046) | 2021 | Shared prototypes organized as a soft decision tree | Soft routing blurs decision paths |
| [TesNet](https://arxiv.org/abs/2105.02968) | 2021 | Orthogonal concept basis vectors in embedding space | Concepts may not be human-meaningful |
| [PIPNet](https://arxiv.org/abs/2307.03672) | 2023 | Sparse activation over real training image patches | Sensitive sparsity threshold hyperparameter |

**Backbone:** ResNet-34 pretrained on ImageNet  
**Dataset:** CUB-200-2011 (200 fine-grained bird species, 11,788 images)  
**Secondary dataset (optional):** Stanford Cars

---

## Repository Structure

```
xai-proto-vision/
│
├── README.md
├── pyproject.toml               ← dependency management (uv)
├── requirements.txt             ← pip fallback
│
├── data/
│   ├── cub200/                  ← download CUB-200-2011 here (gitignored)
│   └── stanford_cars/           ← optional second dataset (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── datasets.py              ← Member A: CUB200Dataset + load_dataset()
│   ├── base_model.py            ← Member A: BaseModel + Trainer classes
│   ├── evaluate.py              ← Member A: evaluate_model()
│   ├── protopnet.py             ← Member B: ProtoPNet implementation
│   ├── prototree.py             ← Member C: ProtoTree implementation
│   ├── tesnet.py                ← Member D: TesNet implementation
│   └── pipnet.py                ← Member D: PIPNet implementation
│
├── train.py                     ← unified entry point
│   # python train.py --method protopnet --dataset cub200 --epochs 100
│
├── experiments/
│   ├── baseline.sh
│   ├── protopnet.sh
│   ├── prototree.sh
│   ├── tesnet.sh
│   └── pipnet.sh
│
├── notebooks/
│   ├── visualize_prototypes.ipynb
│   ├── visualize_tree.ipynb
│   └── results_analysis.ipynb
│
├── checkpoints/                 ← saved model weights (gitignored)
└── results/                     ← logged metrics + figures (gitignored)
```

---

## Setup

### Prerequisites

- Python **3.10** or newer
- [`uv`](https://docs.astral.sh/uv/) — recommended package manager

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS — alternatively via Homebrew
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and create the environment

```bash
git clone <repo-url>
cd xai-proto-vision

# Create a virtual environment and activate it
uv venv --python 3.11
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install PyTorch

PyTorch must be installed separately to match your hardware. Run the command
for your platform before installing the rest of the dependencies:

```bash
# macOS (CPU + Apple MPS)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Linux with CUDA 12.4 (most lab machines / Colab)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Linux CPU-only
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> Check [pytorch.org](https://pytorch.org/get-started/locally/) if you need a
> different CUDA version.

### 4. Install project dependencies

```bash
# Install core + notebook extras
uv pip install -e ".[notebooks]"

# Core only (no Jupyter)
uv pip install -e .
```

### 5. Download CUB-200-2011

```bash
mkdir -p data/cub200
# Download from the official source and extract:
# https://data.caltech.edu/records/65de6-vp158
# The expected layout after extraction:
# data/cub200/
#   images/
#   train_test_split.txt
#   classes.txt
#   image_class_labels.txt
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz -C data/cub200 --strip-components=1
```

---

## Usage

### Training

```bash
# Baseline (backbone + FC head)
python train.py --method baseline --dataset cub200 --epochs 100

# ProtoPNet
python train.py --method protopnet --dataset cub200 --epochs 100 --num-prototypes 10

# ProtoTree
python train.py --method prototree --dataset cub200 --epochs 100 --depth 6

# TesNet
python train.py --method tesnet --dataset cub200 --epochs 100 --num-concepts 32

# PIPNet
python train.py --method pipnet --dataset cub200 --epochs 100 --sparsity-threshold 0.1
```

### Running experiment sweeps

```bash
bash experiments/baseline.sh
bash experiments/protopnet.sh
# etc.
```

### Evaluation and visualisation

Open the notebooks in `notebooks/` for prototype visualisations, tree decision
paths, and the results comparison table/figures.

---

## Interface Contracts

All four model classes inherit from `BaseModel` (defined by Member A). Do not
change these signatures — the shared `Trainer` and `evaluate_model` depend on them.

```python
class BaseModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_classes) logits."""
        raise NotImplementedError

    def explain(self, x: torch.Tensor) -> dict:
        """
        Method-specific explanation dict:
          ProtoPNet:  {"prototype_similarities": Tensor, "patch_locations": Tensor}
          ProtoTree:  {"routing_probs": list[Tensor], "leaf_reached": int}
          TesNet:     {"concept_scores": Tensor}
          PIPNet:     {"active_prototypes": Tensor, "patch_locations": Tensor}
        """
        raise NotImplementedError


class Trainer:
    def __init__(self, model: BaseModel, optimizer, loss_fn): ...
    def train_epoch(self, loader: DataLoader) -> float: ...
    def validate(self, loader: DataLoader) -> float: ...
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...


def evaluate_model(model: BaseModel, loader: DataLoader) -> dict:
    """Returns {"accuracy": float, "inference_time_ms": float, "flops": int}"""
    ...


def load_dataset(name: str, split: str) -> Dataset:
    """name: 'cub200' | 'stanford_cars'  /  split: 'train' | 'val' | 'test'"""
    ...
```

---

## Team Structure

| Member | Primary Responsibility | Secondary |
|---|---|---|
| **A** | Backbone + training harness + integration | Experiments coordinator |
| **B** | ProtoPNet | ProtoPNet experiments + report section |
| **C** | ProtoTree | ProtoTree experiments + report section |
| **D** | TesNet + PIPNet | Discussion section + slides |

> Member A is the **critical path** — `BaseModel` and `Trainer` must be
> delivered by end of Weekend 1. Everything else depends on it.

---

## Project Phases

### Phase 1 — Foundation *(Weekend 1)*

| Member | Deliverable |
|---|---|
| A | `base_model.py`, `trainer.py`, `datasets.py`, baseline accuracy on CUB-200 |
| B | ProtoPNet design doc + `protopnet.py` skeleton |
| C | ProtoTree design doc + `prototree.py` skeleton |
| D | TesNet + PIPNet design docs + `tesnet.py`, `pipnet.py` skeletons |

> Do not start Phase 2 until the baseline accuracy number exists.

### Phase 2 — Parallel Implementation

| Member | Deliverable |
|---|---|
| A | Unified `train.py`, `evaluate.py`, integration review |
| B | Full ProtoPNet with explanation output + prototype visualisation |
| C | Full ProtoTree with soft routing + tree visualisation |
| D | TesNet (concept basis + orthogonality loss) + PIPNet (sparse activation) |

### Phase 3 — Experiments & Report

| Member | Deliverable |
|---|---|
| A | Results spreadsheet + all comparison figures |
| B | Tuned ProtoPNet + report section |
| C | Tuned ProtoTree + report section |
| D | Tuned TesNet/PIPNet + Discussion section + slides |

---

## Recommended Hyperparameter Sweeps

| Method | Parameter | Values to try |
|---|---|---|
| ProtoPNet | Prototypes per class | 5, 10, 20, 50 |
| ProtoTree | Tree depth | 4, 6, 8 |
| TesNet | Number of concepts | 16, 32, 64 |
| PIPNet | Sparsity threshold | 0.05, 0.1, 0.2 |

---

## Expected Results (CUB-200)

| Method | Expected Top-1 Acc. | Gap vs Baseline |
|---|---|---|
| ResNet-34 Baseline | ~74% | — |
| ProtoPNet | ~70–72% | -2 to -4% |
| ProtoTree | ~68–71% | -3 to -6% |
| TesNet | ~71–73% | -1 to -3% |
| PIPNet | ~72–74% | ~0 to -2% |

**Key finding:** interpretability costs roughly 2–5% accuracy; PIPNet closes the
gap most effectively through real-patch constraints and sparse activation.

---

## Git Workflow

```
main                 ← stable only; never push broken code here
├── feature/backbone       ← Member A (Phase 1)
├── feature/protopnet      ← Member B
├── feature/prototree      ← Member C
├── feature/tesnet         ← Member D
└── feature/pipnet         ← Member D
```

**Rule:** only merge to `main` when the method trains end-to-end without errors.

```bash
# Start your feature branch
git checkout -b feature/protopnet

# Commit incrementally; merge to main only when end-to-end works
git push origin feature/protopnet
# then open a PR for review before merging
```
