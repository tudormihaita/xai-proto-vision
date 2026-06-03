import random
from pathlib import Path

import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.constants import CUB200_ROOT, STANFORD_CARS_ROOT, VAL_FRACTION, VAL_SEED


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
        with (self.root / "image_class_labels.txt").open() as f:
            labels = {
                int(iid): int(cls) - 1  # re-index to 0-based
                for iid, cls in (line.strip().split() for line in f)
            }
        with (self.root / "train_test_split.txt").open() as f:
            is_train_flag = {
                int(iid): int(flag) == 1
                for iid, flag in (line.strip().split() for line in f)
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
        with (self.root / filename).open() as f:
            for line in f:
                iid, value = line.strip().split(maxsplit=1)
                result[int(iid)] = value
        return result

    def _load_bboxes(self) -> dict[int, tuple[float, float, float, float]]:
        bboxes = {}
        with (self.root / "bounding_boxes.txt").open() as f:
            for line in f:
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


class StanfordCarsDataset(Dataset):
    NUM_CLASSES = 196

    def __init__(
        self,
        root: str | Path = STANFORD_CARS_ROOT,
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

    # actual on-disk layout of the official Stanford / Kaggle download:
    #   <root>/car_devkit/devkit/   ← annotation .mat files
    #   <root>/cars_train/cars_train/  ← training images (double-nested)
    #   <root>/cars_test/cars_test/    ← test images (double-nested)

    _DEVKIT_DIR   = Path("car_devkit") / "devkit"
    _TRAIN_IMG_DIR = Path("cars_train") / "cars_train"
    _TEST_IMG_DIR  = Path("cars_test")  / "cars_test"

    def _parse_annotations(self) -> None:
        devkit = self.root / self._DEVKIT_DIR
        if not devkit.exists():
            raise FileNotFoundError(f"Stanford Cars devkit not found at {devkit}.")

        train_img_dir = self.root / self._TRAIN_IMG_DIR
        test_img_dir  = self.root / self._TEST_IMG_DIR

        train_mat = devkit / "cars_train_annos.mat"
        test_mat  = devkit / "cars_test_annos_withlabels.mat"
        if not train_mat.exists():
            raise FileNotFoundError(f"Stanford Cars training annotations not found at {train_mat}.")
        if not test_mat.exists():
            raise FileNotFoundError(
                f"Stanford Cars test annotations not found at {test_mat}.\n"
                "Download labels file separately."
            )

        train_samples = self._load_mat(train_mat, train_img_dir)
        test_samples  = self._load_mat(test_mat, test_img_dir)

        rng = random.Random(self.val_seed)
        shuffled = train_samples.copy()
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * self.val_fraction)

        self.samples = {
            "train": shuffled[n_val:],
            "val":   shuffled[:n_val],
            "test":  test_samples,
        }[self.split]

    @staticmethod
    def _load_mat(mat_path: Path, img_dir: Path) -> list[tuple[Path, int, tuple]]:
        mat = sio.loadmat(str(mat_path), squeeze_me=True)
        annos = mat["annotations"]
        samples = []
        for anno in annos:
            fname = str(anno["fname"])
            label = int(anno["class"]) - 1
            x1, y1 = float(anno["bbox_x1"]), float(anno["bbox_y1"])
            x2, y2 = float(anno["bbox_x2"]), float(anno["bbox_y2"])
            samples.append((img_dir / fname, label, (x1, y1, x2 - x1, y2 - y1)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label, bbox = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.use_bbox_crop and bbox is not None:
            x, y, w, h = bbox
            image = image.crop((x, y, x + w, y + h))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __repr__(self) -> str:
        return (
            f"StanfordCarsDataset(split={self.split!r}, n={len(self)}, "
            f"bbox_crop={self.use_bbox_crop})"
        )
