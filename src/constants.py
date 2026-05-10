from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_DATASETS = ["cub200"] # TODO: extend with additional datasets (e.g. stanford cars)

## Data loading and processing ##
DATA_ROOT        = PROJECT_ROOT / "data"
CUB200_ROOT      = DATA_ROOT / "CUB_200_2011" / "CUB_200_2011"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
RESULTS_ROOT     = PROJECT_ROOT / "results"

VAL_SEED     = 42
VAL_FRACTION = 0.2
