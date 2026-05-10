from torch.utils.data import Dataset

from src.data.datasets import CUB200Dataset
from src.data.transforms import get_transforms
from src.constants import CUB200_ROOT, SUPPORTED_DATASETS


def load_dataset(
    name: str,
    split: str,
    root=None,
    image_size: int = 224,
    use_bbox_crop: bool = False,
    **dataset_kwargs,
) -> Dataset:
    """
    Factory method that returns a Dataset for the given split.
    DataLoader wrapping (batch_size, num_workers, etc.) is the caller's responsibility.

    :param name: "cub200" | "stanford_cars"
    :param split: "train" | "val" | "test"
    :param root: dataset root directory; defaults to the constant for each dataset
    :param image_size: resize target passed to get_transforms (default 224)
    :param use_bbox_crop: crop image to bird bounding box before the backbone sees it
    :param dataset_kwargs: forwarded to the Dataset constructor
                          (e.g. val_fraction, val_seed)
    :return: Dataset instance for the specified split
    """
    transform = get_transforms(split, image_size)

    if name == "cub200":
        return CUB200Dataset(
            root=root or CUB200_ROOT,
            split=split,
            transform=transform,
            use_bbox_crop=use_bbox_crop,
            **dataset_kwargs,
        )

    raise ValueError(f"Unknown dataset {name!r}. Supported: {', '.join(SUPPORTED_DATASETS)}")
