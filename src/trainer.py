import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.models import PrototypeModel


class Trainer:
    """
    Generic training loop shared by all models.

    Method-specific behavior is injected via the model, not via subclassing:
      - model.compute_loss(): called automatically for PrototypeModel instances
      - model.push_prototypes(): called automatically at push_epoch for PrototypeModel instances

    Subclass Trainer and override training_step() only if you need phased
    optimization (e.g. ProtoPNet's 3-phase training that freezes/unfreezes the backbone).
    Everything else (validation, early stopping, checkpointing) stays the same.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str | torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        log_every: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.log_every = log_every  # log intra-epoch every N batches; 0 = epoch-level only
        self._step_log: list[tuple[float, float]] = []


    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        val_every: int = 1,
        push_epoch: int | None = None,
        patience: int = 10,
        checkpoint_path: str | None = None,
    ) -> dict:
        """
        Runs the full training loop and returns a history dict ready for plotting:
        {
            "train_loss":   [float, …],   # total loss, one per epoch
            "train_acc":    [float, …],
            "val_loss":     [float, …],   # one entry per validated epoch
            "val_acc":      [float, …],
            "val_epochs":   [int, …],     # which epochs were validated
            "<component>":  [float, …],   # one list per extra loss term (cls, cluster…)
        }

        :param train_loader:  DataLoader for the training split
        :param val_loader:    DataLoader for the validation split
        :param epochs:        maximum number of epochs to train
        :param val_every:     validate every N epochs (default: every epoch)
        :param push_epoch:    epoch at which push_prototypes() fires (None = never)
        :param patience:      early-stopping patience in epochs on val accuracy
        :param checkpoint_path: if not None, path to save the best model checkpoint
        """
        history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
            "val_epochs": [],
            # step-level series (populated only when log_every > 0):
            #   step_x     — fractional epoch (e.g. 1.33 = epoch 1, 33% through)
            #   step_loss  — running-average total loss at that step
            "step_x": [], "step_loss": [],
        }

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            self._step_log: list[tuple[float, float]] = []
            train_metrics = self._train_epoch(train_loader, epoch=epoch)
            history["train_loss"].append(train_metrics.pop("total"))
            history["train_acc"].append(train_metrics.pop("acc"))
            if self._step_log:
                xs, losses = zip(*self._step_log)
                history["step_x"].extend(xs)
                history["step_loss"].extend(losses)
            for k, v in train_metrics.items(): # extra loss components
                history.setdefault(k, []).append(v)

            # prototype push step (only relevant for ProtoPNet and ProtoTree)
            if push_epoch is not None and epoch == push_epoch:
                if isinstance(self.model, PrototypeModel):
                    print("Pushing prototypes...")
                    self.model.push_prototypes(train_loader, self.device)
                    # Optional: model-specific post-push initialization
                    if hasattr(self.model, "post_push_init"):
                        self.model.post_push_init()
                else:
                    print(f"Warning: --push-epoch set but {type(self.model).__name__} is not a PrototypeModel")

            # validation
            if epoch % val_every == 0:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["acc"])
                history["val_epochs"].append(epoch)

                base_keys = {"train_loss", "train_acc", "val_loss", "val_acc", "val_epochs", "step_x", "step_loss"}
                extra = "  ".join(
                    f"{k}={v[-1]:.4f}"
                    for k, v in history.items()
                    if k not in base_keys and v
                )
                print(
                    f"train_loss={history['train_loss'][-1]:.4f}\n"
                    f"train_acc={history['train_acc'][-1]:.4f}\n"
                    f"val_loss={val_metrics['loss']:.4f}\n"
                    f"val_acc={val_metrics['acc']:.4f}\n"
                    + (f"[{extra}]" if extra else "")
                )

                # early stopping + best-model checkpoint
                if val_metrics["acc"] > best_val_acc:
                    best_val_acc = val_metrics["acc"]
                    epochs_no_improve = 0
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path, epoch, history)
                else:
                    epochs_no_improve += val_every
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered (no improvement for {patience} epochs)")
                        break

            if self.scheduler is not None:
                self.scheduler.step()

        return history


    def _train_epoch(self, loader, epoch: int = 0) -> dict:
        """Runs one training epoch. Returns a metrics dict."""
        self.model.train()

        running: dict[str, float] = {}
        correct = 0
        total = 0
        n_batches = len(loader)

        for i, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            step = self.training_step(images, labels)
            step["total"].backward()
            self.optimizer.step()

            for k, v in step.items():
                if k == "logits":
                    continue
                running[k] = running.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)

            with torch.no_grad():
                preds = step["logits"].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            if self.log_every > 0 and (i + 1) % self.log_every == 0:
                step_x = epoch + i / n_batches
                step_loss = running.get("total", 0.0) / (i + 1)
                self._step_log.append((step_x, step_loss))

        metrics = {k: v / n_batches for k, v in running.items()}
        metrics["acc"] = correct / total
        return metrics

    def validate(self, loader) -> dict:
        """Runs one validation pass. Returns {"loss": float, "acc": float}."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="  val", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                total_loss += self.loss_fn(logits, labels).item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        return {"loss": total_loss / len(loader), "acc": correct / total}

    # Overridable hooks
    def training_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Default step: forward pass + loss computation.
        Returns at minimum {"total": Tensor, "cls": Tensor, "logits": Tensor}.

        For PrototypeModel subclasses, calls model.compute_loss() automatically
        so per-component loss terms (ortho, sep, cluster, …) are logged separately.
        Intermediate tensors needed by compute_loss (e.g. backbone features) are
        cached on the model during forward() — no second backbone pass required.

        Override this only if you need phased optimization (e.g. ProtoPNet's
        3-phase training where backbone is frozen for the first N epochs).
        """
        logits = self.model(images)
        if isinstance(self.model, PrototypeModel):
            losses = self.model.compute_loss(logits, labels)
            losses["logits"] = logits
            return losses
        loss = self.loss_fn(logits, labels)
        return {"total": loss, "cls": loss, "logits": logits}

    def save_checkpoint(self, path: str, epoch: int, history: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "history": history,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Loads full training state. Returns the saved history dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")
        return checkpoint["history"]
