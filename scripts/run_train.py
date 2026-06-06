"""
Unified training entry point for all prototype-based methods.

Usage examples
--------------
python run_train.py --method baseline  --epochs 100
python run_train.py --method tesnet    --epochs 50  --num-concepts 10 --warm-epochs 5 --scheduler step
python run_train.py --method protopnet --epochs 100 --num-prototypes 10 --push-epoch 80
python run_train.py --method prototree --epochs 100 --depth 6             --push-epoch 80
python run_train.py --method pipnet    --epochs 100 --sparsity-threshold 0.1

Adding a new method
-------------------
1. Implement src/models/<method>.py (subclass PrototypeModel).
2. Change the branch in build_model() below with correct configuration, and add any method-specific hyperparameters to parse_args().
3. The shared Trainer handles training automatically, no Trainer subclass needed
   unless the method requires phased optimization (e.g. ProtoPNet 3-phase training).
"""

import argparse
import csv
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.constants import CHECKPOINTS_ROOT, RESULTS_ROOT
from src.data import get_transforms, load_dataset
from src.evaluate import evaluate_model, print_results
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a prototype-based model on CUB-200")

    p.add_argument("--method",   required=True,
                   choices=["baseline", "tesnet", "protopnet", "prototree", "pipnet"])
    p.add_argument("--backbone", default="resnet34", choices=["resnet34", "vgg16"])
    p.add_argument("--dataset",  default="cub200", choices=["cub200", "stanford_cars"])
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler",    default="cosine", choices=["none", "cosine", "step"])
    p.add_argument("--step-size",    dest="step_size",  type=int, default=None,
                   help="StepLR step size (only used when --scheduler step). Default: epochs // 3.")
    p.add_argument("--device",   default=(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    ))
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--patience",   type=int, default=15)
    p.add_argument("--val-every",  dest="val_every",  type=int, default=1)
    p.add_argument("--log-every",   dest="log_every",   type=int, default=0,
                   help="log intra-epoch loss every N batches for smoother curves (0 = epoch-level only)")
    p.add_argument("--eval-only", dest="eval_only", action="store_true", default=False,
                   help="skip training and load the existing checkpoint for test evaluation only")
    p.add_argument("--no-save", dest="no_save", action="store_true", default=False,
                   help="disable checkpoint saving (useful for quick overfit tests)")
    p.add_argument("--warm-epochs",   dest="warm_epochs",   type=int,  default=0,
                   help="freeze backbone for first N epochs; paper value for TesNet: 5")
    p.add_argument("--use-bbox-crop", dest="use_bbox_crop", action="store_true", default=False,
                   help="crop images to bounding-box annotation before backbone (recommended for TesNet)")

    # method-specific hyperparameters
    p.add_argument("--num-concepts",       dest="num_concepts",       type=int,   default=10,
                   help="TesNet: concepts PER CLASS (total = this × num_classes). Paper value: 10. Sweep: 5, 10, 20")
    p.add_argument("--concept-dim",        dest="concept_dim",        type=int,   default=64,
                   help="TesNet: bottleneck concept dimension (paper value: 64)")
    p.add_argument("--num-prototypes",     dest="num_prototypes",     type=int,   default=10)
    p.add_argument("--depth",              type=int,                              default=6)
    p.add_argument("--sparsity-threshold", dest="sparsity_threshold", type=float, default=0.1)
    p.add_argument("--push-epoch",         dest="push_epoch",         type=int,   default=None)
    p.add_argument("--push-batch-size",    dest="push_batch_size",    type=int,   default=None)
    p.add_argument("--save-every",         dest="save_every",         type=int,   default=None,
                   help="ProtoPNet only: save *_latest recovery checkpoint every N joint epochs")
    p.add_argument("--lambda-ortho",       dest="lambda_ortho",       type=float, default=1e-4,
                   help="TesNet: within-class orthogonality weight (paper value: 1e-4)")
    p.add_argument("--lambda-clst",        dest="lambda_clst",        type=float, default=0.8,
                   help="TesNet: cluster loss — pull features toward same-class concepts (paper value: 0.8)")
    p.add_argument("--lambda-sep",         dest="lambda_sep",         type=float, default=0.08,
                   help="TesNet: separation loss — push features from other-class concepts (paper value: 0.08)")
    p.add_argument("--lambda-ss",          dest="lambda_ss",          type=float, default=0.08,
                   help="TesNet: subspace separation on Grassmann manifold (mean over pairs; paper uses sum with 1e-7, equiv scale here is 0.08)")
    p.add_argument("--lambda-l1",          dest="lambda_l1",          type=float, default=1e-4,
                   help="TesNet: L1 sparsity on classifier weights (paper value: 1e-4)")

    return p.parse_args()


NUM_CLASSES = {"cub200": 200, "stanford_cars": 196}


def build_model(args) -> nn.Module:
    num_classes = NUM_CLASSES[args.dataset]

    if args.method == "baseline":
        from src.models import BaselineModel
        return BaselineModel(backbone_name=args.backbone, num_classes=num_classes)

    if args.method == "tesnet":
        from src.models.tesnet import TesNet
        return TesNet(
            backbone_name=args.backbone,
            num_classes=num_classes,
            num_concepts_per_class=args.num_concepts,
            concept_dim=args.concept_dim,
            lambda_clst=args.lambda_clst,
            lambda_sep=args.lambda_sep,
            lambda_ortho=args.lambda_ortho,
            lambda_ss=args.lambda_ss,
            lambda_l1=args.lambda_l1,
        )

    if args.method == "protopnet":
        from src.models.protopnet import ProtoPNet
        return ProtoPNet(backbone_name=args.backbone, num_classes=num_classes,
                         num_prototypes_per_class=args.num_prototypes)

    if args.method == "prototree":
        from src.models.prototree import ProtoTree
        return ProtoTree(
            backbone_name=args.backbone,
            num_classes=num_classes,
            depth=args.depth,
            lambda_cluster=args.lambda_clst,
        )

    if args.method == "pipnet":
        from src.models.pipnet import PIPNet
        return PIPNet(backbone_name=args.backbone, num_classes=num_classes,
                      sparsity_threshold=args.sparsity_threshold)

    raise NotImplementedError(f"Method '{args.method}' not yet implemented")


def build_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    from src.models import PrototypeModel
    from src.models.tesnet import TesNet
    wd = args.weight_decay

    if isinstance(model, TesNet):
        return torch.optim.Adam([
            {"params": model.backbone.parameters(),        "lr": 1e-4,  "weight_decay": 1e-3},
            {"params": list(model.add_on_layers.parameters()) + [model.concept_vectors],
                                                           "lr": 3e-3,  "weight_decay": 1e-3},
            {"params": model.classifier.parameters(),      "lr": 1e-4,  "weight_decay": wd},
        ])

    if isinstance(model, PrototypeModel):
        return torch.optim.Adam([
            {"params": model.get_backbone_params(),  "lr": args.lr * 0.1, "weight_decay": wd},
            {"params": model.get_prototype_params(), "lr": args.lr,        "weight_decay": wd},
        ])

    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)


def build_scheduler(args, optimizer, num_epochs: int):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if args.scheduler == "step":
        step_size = args.step_size if args.step_size is not None else max(1, num_epochs // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    return None


def build_trainer(args, model: nn.Module) -> Trainer:
    # ProtoPNet does phased optimization (warm/joint/last with its own per-phase
    # optimizers + LR schedule), so it manages its own optimizer/scheduler rather
    # than the generic ones built below. --epochs maps to the joint stage.
    if args.method == "protopnet":
        from src.models.protopnet import ProtoPNetTrainer

        warm_epochs = min(5, args.epochs)
        return ProtoPNetTrainer(
            model,
            device=args.device,
            warm_epochs=warm_epochs,
            joint_epochs=max(1, args.epochs - warm_epochs),
            push_interval=10,
            last_layer_iters=20,
            joint_lr_step_size=5,
            joint_lr_gamma=0.1,
            weight_decay=args.weight_decay,
        )

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)

    return Trainer(model, optimizer, nn.CrossEntropyLoss(), args.device,
                   scheduler=scheduler, log_every=args.log_every,
                   warm_epochs=args.warm_epochs)


def generate_training_run_tag(args) -> str:
    # format: <method>_<dataset>_<backbone>_<key-hyperparam>
    # e.g. tesnet_cub200_resnet34_k32
    ds  = args.dataset
    bb  = args.backbone
    if args.method == "tesnet":
        dim_suffix = f"_d{args.concept_dim}" if args.concept_dim != 64 else ""
        return f"tesnet_{ds}_{bb}_k{args.num_concepts}{dim_suffix}"
    if args.method == "protopnet":
        return f"protopnet_{ds}_{bb}_p{args.num_prototypes}"
    if args.method == "prototree":
        return f"prototree_{ds}_{bb}_d{args.depth}"
    if args.method == "pipnet":
        return f"pipnet_{ds}_{bb}_s{str(args.sparsity_threshold).replace('.', '')}"
    return f"baseline_{ds}_{bb}"


def main() -> None:
    args = parse_args()

    # Push at 70% so the remaining 30% of epochs recalibrate the classifier;
    # ensures post-push epochs fall within the early-stopping patience window.
    if args.method == "tesnet" and args.push_epoch is None:
        args.push_epoch = max(1, int(args.epochs * 0.70))

    run_tag = generate_training_run_tag(args)

    loader_kwargs  = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    dataset_kwargs = dict(use_bbox_crop=args.use_bbox_crop)
    train_loader = DataLoader(load_dataset(args.dataset, "train", **dataset_kwargs), shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(load_dataset(args.dataset, "val",   **dataset_kwargs), shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(load_dataset(args.dataset, "test",  **dataset_kwargs), shuffle=False, **loader_kwargs)

    push_loader = None
    if args.method == "protopnet":
        push_ds = load_dataset(args.dataset, "train", **dataset_kwargs)
        # clean deterministic views; source indices must be stable across runs
        push_ds.transform = get_transforms("test", 224)
        push_loader = DataLoader(
            push_ds,
            batch_size=args.push_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    model = build_model(args)
    model.to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMethod: {args.method}  |  Run: {run_tag}  |  Device: {args.device}  |  Trainable params: {num_params:,}\n")

    trainer = build_trainer(args, model)

    method_ckpt_dir = CHECKPOINTS_ROOT / args.method
    method_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(method_ckpt_dir / f"{run_tag}_best.pt")

    config = vars(args)
    with open(method_ckpt_dir / f"{run_tag}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.eval_only and args.no_save:
        raise ValueError("--eval-only and --no-save cannot be combined: eval-only mode requires an existing checkpoint")

    ckpt_path_to_use = None if args.no_save else ckpt_path

    if args.eval_only:
        print(f"Eval-only mode; loading checkpoint: {ckpt_path}")
        train_minutes = 0.0
    else:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)

        t0 = time.time()
        train_kwargs = dict(
            epochs=args.epochs,
            val_every=args.val_every,
            push_epoch=args.push_epoch,
            patience=args.patience,
            checkpoint_path=ckpt_path_to_use,
        )
        if args.method == "protopnet":
            train_kwargs["push_loader"] = push_loader
            train_kwargs["save_every"] = args.save_every
        trainer.train(train_loader, val_loader, **train_kwargs)
        train_minutes = (time.time() - t0) / 60
        print(f"\nTraining complete in {train_minutes:.1f} min; Best checkpoint: {ckpt_path}")

    # eval-only always loads; normal training loads best checkpoint unless --no-save
    if args.eval_only or not args.no_save:
        trainer.load_checkpoint(ckpt_path)

    results = evaluate_model(model, test_loader, args.device)
    per_class = results.pop("per_class_accuracy")   # numpy array -> save stats, not raw array
    results["per_class_acc_mean"] = float(per_class.mean())
    results["per_class_acc_std"]  = float(per_class.std())
    results["method"]             = args.method
    results["run_tag"]            = run_tag
    results["training_time_min"]  = round(train_minutes, 2)
    results["peak_gpu_memory_mb"] = (
        torch.cuda.max_memory_allocated(args.device) / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )
    results["num_params"]         = num_params
    results["num_concepts"]       = getattr(model, "num_concepts", None) or getattr(model, "num_prototypes", None)

    print_results(results, model_name=run_tag)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / f"{run_tag}_results.csv"
    fields = [
        "method", "run_tag", "accuracy", "topk_accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "per_class_acc_mean", "per_class_acc_std",
        "training_time_min", "peak_gpu_memory_mb",
        "num_params", "num_concepts", "inference_time_ms", "flops",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(results)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
