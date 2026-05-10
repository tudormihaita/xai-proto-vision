import random
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.constants import CUB200_ROOT, VAL_FRACTION, VAL_SEED


class CUB200Dataset(Dataset):
    NUM_CLASSES = 200

    def __init__(
        self,
        root: str | Path = CUB200_ROOT,
        split: str = "train",
        transform=None,
        use_bbox_crop: bool = False,
        val_fraction: float = VAL_FRACTION,
        val_seed: int = VAL_SEED,
    ):
        assert split in {"train", "val", "test"}, f"split must be train/val/test, got {split!r}"

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_bbox_crop = use_bbox_crop
        self.val_fraction = val_fraction
        self.val_seed = val_seed

        self.samples: list[tuple[Path, int, tuple | None]] = []
        self._parse_annotations()

    def _parse_annotations(self) -> None:
        images = self._load_id_map("images.txt")
        labels = {
            int(iid): int(cls) - 1  # re-index to 0-based
            for iid, cls in (
                line.strip().split() for line in
                (self.root / "image_class_labels.txt").open()
            )
        }
        is_train_flag = {
            int(iid): int(flag) == 1
            for iid, flag in (
                line.strip().split() for line in
                (self.root / "train_test_split.txt").open()
            )
        }
        bboxes = self._load_bboxes() if self.use_bbox_crop else {}

        train_ids = sorted(iid for iid, flag in is_train_flag.items() if flag)
        test_ids  = sorted(iid for iid, flag in is_train_flag.items() if not flag)

        # extract val split from train with a private RNG so global state is untouched
        rng = random.Random(self.val_seed)
        shuffled = train_ids.copy()
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * self.val_fraction)

        split_ids = {
            "train": shuffled[n_val:],
            "val":   shuffled[:n_val],
            "test":  test_ids,
        }[self.split]

        self.samples = [
            (self.root / "images" / images[iid], labels[iid], bboxes.get(iid))
            for iid in split_ids
        ]

    def _load_id_map(self, filename: str) -> dict[int, str]:
        result = {}
        for line in (self.root / filename).open():
            iid, value = line.strip().split(maxsplit=1)
            result[int(iid)] = value
        return result

    def _load_bboxes(self) -> dict[int, tuple[float, float, float, float]]:
        bboxes = {}
        for line in (self.root / "bounding_boxes.txt").open():
            iid, x, y, w, h = line.strip().split()
            bboxes[int(iid)] = (float(x), float(y), float(w), float(h))
        return bboxes


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label, bbox = self.samples[idx]
        image = Image.open(path).convert("RGB")

        # crop to bird bounding box on the raw PIL image, before any resize/normalize
        if self.use_bbox_crop and bbox is not None:
            x, y, w, h = bbox
            image = image.crop((x, y, x + w, y + h))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __repr__(self) -> str:
        return (
            f"CUB200Dataset(split={self.split!r}, n={len(self)}, "
            f"bbox_crop={self.use_bbox_crop})"
        )
