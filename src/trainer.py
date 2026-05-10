import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.models import BaselineModel, PrototypeModel


class Trainer:
    """
    Generic training loop shared by all models.

    The loop is fixed; method-specific behavior is injected via two hooks:
      - training_step(): override to add extra loss terms or phased optimization
      - push_prototypes(): override in ProtoPNet / ProtoTree trainers

    Prototype-method trainers should subclass this and override those two hooks.
    Everything else (validation, early stopping, checkpointing) stays the same.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str | torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler


    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        val_every: int = 1,
        push_epoch: int | None = None,
        patience: int = 10,
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
        """
        history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
            "val_epochs": [],
        }

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # training step
            train_metrics = self._train_epoch(train_loader)
            history["train_loss"].append(train_metrics.pop("total"))
            history["train_acc"].append(train_metrics.pop("acc"))
            for k, v in train_metrics.items():       # extra loss components
                history.setdefault(k, []).append(v)

            # prototype push step
            if push_epoch is not None and epoch == push_epoch:
                print("Running prototype push step...")
                self.push_prototypes(train_loader)

            # validation
            if epoch % val_every == 0:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["acc"])
                history["val_epochs"].append(epoch)

                print(
                    f"  train_loss={history['train_loss'][-1]:.4f} "
                    f"train_acc={history['train_acc'][-1]:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['acc']:.4f}"
                )

                # early stopping on validation accuracy
                if val_metrics["acc"] > best_val_acc:
                    best_val_acc = val_metrics["acc"]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += val_every
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered (no improvement for {patience} epochs)")
                        break

            if self.scheduler is not None:
                self.scheduler.step()

        return history


    def _train_epoch(self, loader) -> dict:
        """Runs one training epoch. Returns a metrics dict."""
        self.model.train()

        running: dict[str, float] = {}
        correct = 0
        total = 0

        for batch in tqdm(loader, desc="  train", leave=False):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            step = self.training_step(images, labels)
            step["total"].backward()
            self.optimizer.step()

            # accumulate loss components
            for k, v in step.items():
                if k == "logits":
                    continue
                running[k] = running.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)

            with torch.no_grad():
                preds = step["logits"].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        n = len(loader)
        metrics = {k: v / n for k, v in running.items()}
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

    ##  Overridable hooks ##
    def training_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Default step: forward pass + cross-entropy loss.
        Returns {"total": Tensor, "cls": Tensor, "logits": Tensor}.

        Prototype-method trainers override this to call model.compute_loss()
        and return the full dict of loss components for separate logging.
        """
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        return {"total": loss, "cls": loss, "logits": logits}

    def push_prototypes(self, train_loader) -> None:
        """
        No-op in the base Trainer. Overridden by ProtoPNet and ProtoTree
        trainers to scan the training set and anchor prototypes to real patches.
        """


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
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Loads full training state. Returns the saved history dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")
        return checkpoint["history"]
