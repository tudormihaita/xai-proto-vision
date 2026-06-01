import time
import torch
import numpy as np

from torchinfo import summary


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Fraction of samples where the true class appears in the top-k predictions."""
    topk_preds = logits.topk(k, dim=1).indices  # (N, k)
    correct = topk_preds.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: str | torch.device,
    topk: int = 5,
) -> dict:
    """
    Runs inference on the full test split and returns a results dict suitable
    for direct inclusion in the comparison table.

    Returns:
        {
            "accuracy":           float,        top-1 accuracy over the test split
            "topk_accuracy":      float,        top-k accuracy (k=topk, default 5)
            "per_class_accuracy": np.ndarray,   (num_classes,) — per-species accuracy
            "inference_time_ms":  float,        mean per-image inference time in ms
            "flops":              int,           multiply-accumulate ops for one image
        }
    """
    model.eval()
    model.to(device)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    batch_times: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            t0 = time.perf_counter()
            logits = model(images)
            t1 = time.perf_counter()

            ms_per_image = (t1 - t0) / images.size(0) * 1_000
            batch_times.append(ms_per_image)

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    preds  = all_logits_t.argmax(dim=1).numpy()
    labels = all_labels_t.numpy()

    accuracy = float((preds == labels).mean())

    num_classes = model.num_classes
    per_class_acc = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (preds[mask] == labels[mask]).mean()

    # FLOPs: run torchinfo on a single-image dummy input
    dummy = next(iter(loader))[0][:1].to(device)
    info = summary(model, input_data=dummy, verbose=0)
    flops = info.total_mult_adds

    return {
        "accuracy":           accuracy,
        "topk_accuracy":      topk_accuracy(all_logits_t, all_labels_t, k=topk),
        "per_class_accuracy": per_class_acc,
        "inference_time_ms":  float(np.mean(batch_times)),
        "flops":              flops,
    }


def print_results(results: dict, model_name: str = "") -> None:
    """Pretty-prints the evaluate_model() output for quick inspection."""
    prefix = f"[{model_name}] " if model_name else ""
    # per_class_accuracy may be a numpy array (direct from evaluate_model)
    # or already summarised as mean/std keys (after run_train.py post-processing)
    if "per_class_accuracy" in results:
        pca_str = f"mean_per_class_acc={results['per_class_accuracy'].mean():.4f}"
    else:
        pca_str = f"mean_per_class_acc={results.get('per_class_acc_mean', float('nan')):.4f}"
    lines = [
        f"{prefix}acc={results['accuracy']:.4f}",
        f"top5_acc={results['topk_accuracy']:.4f}",
        pca_str,
        f"inference={results['inference_time_ms']:.2f}ms/img",
        f"FLOPs={results['flops']:,}",
    ]
    print("  ".join(lines))