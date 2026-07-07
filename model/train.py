import time
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import AdamW
from torch.amp import GradScaler
from tqdm import tqdm
import os
import numpy as np
from .dataset import create_loaders
from .model import create_classifier
from .config import get_config
import random
import torch.multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", message="Palette images with Transparency")
mp.set_start_method("spawn", force=True)
torch.backends.cudnn.benchmark = True


def mixup_data(x, y, alpha=0.2):
    """
    Applies MixUp augmentation to a batch of images and labels.

    MixUp creates convex combinations of image pairs and their corresponding labels,
    which helps improve model generalisation and robustness.

    Args:
        x: Input batch of images (B, C, H, W).
        y: Corresponding labels (B, num_classes) as multi-hot vectors.
        alpha: Parameter for the Beta distribution (controls the mixing strength).

    Returns:
        mixed_x: Mixed batch of images.
        y_a, y_b: The two original label sets (used for loss calculation with lambda).
        lam: The mixing coefficient sampled from Beta(alpha, alpha).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device):
    """
    Trains the model for one epoch.

    This function handles the training loop, including:
    - Applying MixUp augmentation with 50% probability.
    - Mixed precision training (AMP) via GradScaler.
    - Updating the learning rate scheduler (step after each batch).
    - Skipping the optimiser step if loss is zero (rare safeguard).

    Args:
        model: The model to train.
        loader: DataLoader for training data.
        optimizer: Optimizer (AdamW) for updating weights.
        criterion: Loss function (BCEWithLogitsLoss).
        scaler: GradScaler for mixed precision training.
        scheduler: Learning rate scheduler (CosineAnnealingWarmRestarts).
        device: Device to run training on (cuda/cpu).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    train_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Apply MixUp with 50% probability
        if torch.rand(1).item() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                logits = model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(
                    logits, labels_b
                )
        else:
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

        # Backward pass with mixed precision scaling
        scaler.scale(loss).backward()

        # Only update weights if loss is non-zero (prevents potential issues)
        if loss.item() != 0:
            scaler.step(optimizer)
            scaler.update()
        else:
            # If loss is zero, fallback to standard optimizer step
            optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return train_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, f1_metric, ap_metric, device):
    """
    Evaluates the model on the validation set.

    Computes validation loss, macro F1 score, and mean Average Precision (mAP).

    Args:
        model: The model to evaluate.
        loader: DataLoader for validation data.
        criterion: Loss function.
        f1_metric: TorchMetrics F1Score calculator.
        ap_metric: TorchMetrics AveragePrecision calculator.
        device: Device to run validation on.

    Returns:
        val_loss: Average validation loss.
        f1_score: Macro F1 score.
        map_score: Mean Average Precision (mAP).
    """
    model.eval()
    val_loss = 0.0
    f1_metric.reset()
    ap_metric.reset()
    loop = tqdm(loader, desc="Validation", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        val_loss += loss.item()

        # Compute metrics from probabilities
        probs = torch.sigmoid(logits)
        binary_labels = (labels > 0.5).long()
        f1_metric.update(probs, binary_labels)
        ap_metric.update(probs, binary_labels)

    return val_loss / len(loader), f1_metric.compute(), ap_metric.compute()


def main(cfg):
    """
    Main training script.

    Orchestrates the entire training pipeline:
    - Loads data, creates model, sets up optimizer and scheduler.
    - Handles checkpoint loading and saving.
    - Runs training and validation loops.
    - Tracks and saves the best model based on F1 score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and ingredient vocabulary
    (
        train_loader,
        val_loader,
        ingredient_list,
        ingredient_to_idx,
        NUM_INGREDIENTS,
        val_idx,
    ) = create_loaders(cfg)

    # Set random seeds for reproducibility
    seed = cfg.get("SEED", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create model with the desired unfreeze strategy
    model = create_classifier(NUM_INGREDIENTS, unfreeze_stage="Disc_FT")
    model = model.to(device)

    # Loss function: BCEWithLogitsLoss (standard for multi-label classification)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer: AdamW with separate learning rates for backbone and head
    HEAD_LR = cfg["HEAD_LR"]
    BACKBONE_LR = cfg["BACKBONE_LR"]
    WEIGHT_DECAY = cfg["WEIGHT_DECAY"]
    optimizer = AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not n.startswith("head.")
                    and "stages.0" not in n
                    and "stages.1" not in n
                ],
                "lr": BACKBONE_LR,
            },
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler: CosineAnnealingWarmRestarts with increasing cycle length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,          # initial cycle length in epochs
        T_mult=2,       # multiply cycle length by 2 each cycle
        eta_min=1e-6    # minimum learning rate
    )

    # Mixed precision scaler
    scaler = GradScaler("cuda")

    # Metrics: macro F1 and mAP
    f1_metric = torchmetrics.F1Score(
        task="multilabel", num_labels=NUM_INGREDIENTS, average="macro"
    ).to(device)
    ap_metric = torchmetrics.AveragePrecision(
        task="multilabel", num_labels=NUM_INGREDIENTS, average="macro"
    ).to(device)

    # Checkpoint loading
    start_epoch = 1
    best_val_f1 = 0.0
    best_val_loss = float("inf")

    if os.path.exists(cfg["OLD_CHECKPOINT_PATH"]):
        ckpt = torch.load(
            cfg["OLD_CHECKPOINT_PATH"], map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        # Note: scaler state is not restored to avoid compatibility issues
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        # Restore vocabulary and indices (useful for reproducibility)
        ingredient_list = ckpt.get("ingredient_list", ingredient_list)
        ingredient_to_idx = ckpt.get("ingredient_to_idx", ingredient_to_idx)
        NUM_INGREDIENTS = ckpt.get("NUM_INGREDIENTS", NUM_INGREDIENTS)
        val_idx = ckpt.get("val_indices", val_idx)
        print(f"Loaded checkpoint epoch {start_epoch}, best F1={best_val_f1:.4f}")

    # Training loop
    for epoch in range(start_epoch, start_epoch + cfg["EPOCHS"] + 1):
        torch.cuda.empty_cache()  # Clear GPU cache before each epoch

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, device
        )
        val_loss, f1, mAP = validate(
            model, val_loader, criterion, f1_metric, ap_metric, device
        )

        scheduler.step()  # Update learning rate after each epoch

        # Save checkpoint if F1 score improves
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "ingredient_list": ingredient_list,
                    "ingredient_to_idx": ingredient_to_idx,
                    "NUM_INGREDIENTS": NUM_INGREDIENTS,
                    "best_val_f1": best_val_f1,
                    "best_val_loss": best_val_loss,
                    "val_indices": val_idx,
                    "mAP": mAP,
                },
                cfg["NEW_CHECKPOINT_PATH"],
            )
            save_msg = "|Checkpoint saved"
        else:
            save_msg = ""

        # Print progress
        print(
            f"Epoch{epoch:02d}/{start_epoch + cfg['EPOCHS']}|loss={train_loss:.4f}|val_loss={val_loss:.4f}|mAP={mAP:.4f}|lr={optimizer.param_groups[0]['lr']:.6f}|F1={f1:.4f}|{save_msg}"
        )

        torch.cuda.empty_cache()  # Clear cache again after epoch

    print("Training finished.")


if __name__ == "__main__":
    print("Imports OK")
    main(get_config())