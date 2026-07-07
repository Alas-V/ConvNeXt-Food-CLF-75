import os
import pickle
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_from_disk
from PIL import Image
import io


class IngredientDataset(Dataset):
    """
    Custom PyTorch Dataset for multi-label food ingredient classification.

    Handles loading images from HuggingFace datasets in various formats (PIL, numpy, bytes)
    and converts multi-hot labels to FloatTensor. Applies Albumentations transforms if provided.
    """
    def __init__(self, dataset, ingredient_to_idx, img_size, transform=None):
        self.dataset = dataset
        self.ingredient_to_idx = ingredient_to_idx
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Load image (handles PIL, numpy, or bytes format)
        try:
            image = sample["image"]
            if hasattr(image, "convert"):
                image = np.array(image)
            elif isinstance(image, dict):
                image = np.array(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
        except Exception:
            # Fallback to zero image and zero label if loading fails
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            label = torch.zeros(len(self.ingredient_to_idx), dtype=torch.float32)
            if self.transform is not None:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image, label

        # Load multi-hot label vector (prepared from the cleaning pipeline)
        raw_label = sample["label"]
        if isinstance(raw_label, list):
            label = torch.tensor(raw_label, dtype=torch.float32)
        elif isinstance(raw_label, torch.Tensor):
            label = raw_label.float().clone()
        else:
            # If label missing, return zero vector
            label = torch.zeros(len(self.ingredient_to_idx), dtype=torch.float32)

        # Apply transformations (if any) or do simple tensor conversion
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def create_loaders(cfg):
    """
    Builds training and validation DataLoaders for the classifier.

    This function:
      - Loads the cleaned dataset and ingredient vocabulary from disk.
      - Defines training and validation augmentation pipelines.
      - Splits the dataset into train/val (or restores val indices from checkpoint).
      - Creates DataLoaders with appropriate settings for training and validation.

    Args:
        cfg (dict): configuration dictionary with keys:
            - DATA_DIR: path to the cleaned dataset (HuggingFace dataset)
            - VOCAB_PATH: path to ingredient_dict.pkl (contains class list and mappings)
            - IMG_SIZE: target image size
            - BATCH_SIZE: batch size for both training and validation
            - VAL_SUBSET_SIZE: number of samples to use for validation
            - OLD_CHECKPOINT_PATH: optional path to load saved validation indices

    Returns:
        tuple: (train_loader, val_loader, ingredient_list, ingredient_to_idx, NUM_INGREDIENTS, val_idx)
    """
    # Load the cleaned dataset and ingredient vocabulary
    dataset = load_from_disk(cfg["DATA_DIR"])

    with open(cfg["VOCAB_PATH"], "rb") as f:
        d = pickle.load(f)
    ingredient_list = d["ingredient_list"]
    ingredient_to_idx = d["ingredient_to_idx"]
    NUM_INGREDIENTS = d["NUM_INGREDIENTS"]

    # Training augmentations (albumentations)
    train_transform = A.Compose([
        A.RandomResizedCrop(
            size=(cfg["IMG_SIZE"], cfg["IMG_SIZE"]),
            scale=(0.6, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    # Validation transformations (only resize and normalisation)
    val_transform = A.Compose([
        A.Resize(height=cfg["IMG_SIZE"], width=cfg["IMG_SIZE"]),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    # Split the dataset into training and validation (90/10)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    # Try to restore validation indices from a previous checkpoint (for reproducibility)
    val_idx = None
    if cfg.get("OLD_CHECKPOINT_PATH") and os.path.exists(cfg["OLD_CHECKPOINT_PATH"]):
        checkpoint = torch.load(
            cfg["OLD_CHECKPOINT_PATH"],
            map_location="cpu",
            weights_only=False
        )
        val_idx = checkpoint.get("val_indices", None)

    if val_idx is None:
        # Random split if no checkpoint is found
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val])
        train_idx = train_idx.indices
        val_idx = val_idx.indices
    else:
        # Use saved validation indices, training set becomes the rest
        val_idx = list(val_idx)
        train_idx = list(set(range(n_total)) - set(val_idx))

    # If VAL_SUBSET_SIZE is smaller than full validation set, move extra samples to training
    extra_train = val_idx[cfg["VAL_SUBSET_SIZE"]:]
    full_train_indices = train_idx + extra_train
    val_indices = val_idx[:cfg["VAL_SUBSET_SIZE"]]

    # Instantiate datasets
    train_dataset = IngredientDataset(
        dataset=dataset.select(full_train_indices),
        ingredient_to_idx=ingredient_to_idx,
        img_size=cfg["IMG_SIZE"],
        transform=train_transform,
    )

    val_dataset = IngredientDataset(
        dataset=dataset.select(val_indices),
        ingredient_to_idx=ingredient_to_idx,
        img_size=cfg["IMG_SIZE"],
        transform=val_transform,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return (
        train_loader,
        val_loader,
        ingredient_list,
        ingredient_to_idx,
        NUM_INGREDIENTS,
        val_idx,
    )