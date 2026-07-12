# Model Training & Usage Guide

This README contains all training details, hyperparameters, and results,  
followed by step‑by‑step instructions for re‑training the model and running inference.

## 🧪**Training Procedure**

### Model Structure

| Component | Description |
| :--- | :--- |
| **Backbone** | ConvNeXt-Tiny (pretrained on ImageNet-1k) |
| **Attention** | CBAM (Channel + Spatial) |
| **Pooling** | GeM (Generalized Mean) with learnable p |
| **Head** | Linear(1024 → 512) → GELU → Dropout(0.3) → Linear(512 → 75) |
| **Input** | RGB image resized to 640×640 |
| **Output** | Multi-label logits (binary relevance per ingredient) |


### Training Configuration

| Parameter | Value |
| :--- | :--- |
| Backbone | ConvNeXt_tiny (pretrained on ImageNet-1k) |
| Head | CBAM + GeM Pooling + 2‑layer MLP (hidden=512, dropout=0.3) |
| Loss | BCEWithLogitsLoss (class‑weighted: background=1, others=30) |
| Optimizer | AdamW |
| Initial LR | Head: 1e‑4, Backbone: 5e‑6 |
| Scheduler | OneCycleLR (early) / CosineAnnealingWarmRestarts (late)|
| Weight decay | 1e‑4 |
| Batch size | 10 |
| Input resolution | 640×640 |
| MixUp | Probability 0.5 (alpha=0.2) |
| Mixed precision | Yes (AMP) |

### Training Stages

1. **Feature Extraction (28 epochs):** Only the classification head is trained (LR=1e‑4 for head, backbone frozen, scheduler OneCycleLR).
2. **Discriminative Fine‑Tuning (8 epochs):** Head + stages 2 and 3 of backbone (LR: head 1e‑4, backbone 1e‑5 scheduler OneCycleLR).
3. **Staged Unfreezing (1 epoch):** Head (LR: head 1e‑4, scheduler CosineAnnealingWarmRestarts).
4. **Final Feature Extraction (5 epochs):** Only the classification head is trained (LR=5e‑5, cheduler CosineAnnealingWarmRestarts).

### Training Results by Stage

| Stage | Epoch | F1 (macro) | mAP | Val Loss |
| :--- | :--- | :--- | :--- | :--- |
| Feature Extraction | 28 | 0.3533 | 0.5067 | 0.0637 |
| Discriminative Fine-Tuning | 36 | 0.6121 | 0.7017 | 0.0469 |
| Staged Unfreezing | 37 | 0.6426 | 0.7097 | 0.0479 |
| Final Feature Extraction | 40 | **0.6546** | **0.7142** | 0.0489 |

---

### Hardware and Training Time

- **GPU:** NVIDIA RTX 2060 (6GB VRAM)
- **CPU:** Xeon e5 2660 v3
- **OS:** CachyOS (Arch‑based)  
- **Framework:** PyTorch 2.5 + torchvision 0.20 + timm 1.0.9
- **Training time:** Approximately 1 hour for 1 epoch
- **Batch size:** 10 
- **Mixed precision:** AMP via `torch.amp`

---

## 🏋️**Training from Scratch**

### 0. Clone the Repository
```bash
git clone https://github.com/Alas-V/ConvNeXt-Food-CLF-75.git
cd ConvNeXt-Food-CLF-75
```

### 1. Dataset Preparation
Follow the instructions in [`dataset_preparation/`](../dataset_preparation/) to clean the raw MM‑Food‑100K and generate `ingredient_dict.pkl`.

### 2. Configuration
Edit `config.py` or override parameters. Settings:

| Parameter | Description |
| :--- | :--- |
| `BATCH_SIZE` | 10 (optimal for 6GB VRAM) |
| `VAL_SUBSET_SIZE` | Image quantity for validation set |
| `IMG_SIZE` | 640 (default) |
| `WEIGHT_DECAY` | 1e-4 |
| `EPOCHS` | Quantity of epochs |
| `HEAD_LR` / `BACKBONE_LR` | Learning rates for head and backbone |
| `DATA_DIR` | Path to the cleaned HuggingFace dataset |
| `VOCAB_PATH` | Path to `ingredient_dict.pkl` |
| `OLD_CHECKPOINT_PATH` | Path for resuming training |
| `NEW_CHECKPOINT_PATH` | Path for saving checkpoint is |
| `SEED` | 42 (for validation set fixation) |
| `NUM_INGREDIENTS` | 75 |

### 3. Start Training
for Linux / MacOS
```bash
cd ConvNeXt-Food-CLF-75
python -m venv venv
source venv/bin/activate
# If you use terminal shell, activate accordingly (e.g. for fish: source venv/bin/activate.fish)
pip install -r requirements.txt
python -m model.train
```

for Windows 
```bash
cd ConvNeXt-Food-CLF-75
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
python -m model.train
```

## 🖩**Inference**

You can run predictions with a single command

### Local inference (command line)

for Linux / MacOS
```bash
cd ConvNeXt-Food-CLF-75
python -m venv venv
source venv/bin/activate
# If you use terminal shell, activate accordingly (e.g. for fish: source venv/bin/activate.fish)
pip install -r requirements-inference.txt
python -m model.inference --image path/to/photo.jpg --checkpoint checkpoint.pth.tar
```

for Windows
```bash
cd ConvNeXt-Food-CLF-75
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements-inference.txt
python -m model.inference --image path/to/photo.jpg --checkpoint checkpoint.pth.tar
```

You can find my [checkpoint on Hugging Face](https://huggingface.co/Alas-V/ConvNeXt-Food-CLF-75/blob/main/checkpoint.pth.tar)

| Argument | Description | Default | 
| :--- | :--- | :--- | 
| --image | Path to an image or a folder | required | 
| --checkpoint | Path to the model checkpoint | required | 
| --threshold | Confidence threshold (0.0–1.0) | 0.5 | 
| --mode | text prints ingredients, image saves annotated photo | text | 
| --output_dir | Folder for annotated images (only with --mode image) | predictions | 
| --device | cuda or cpu | cuda if available, else cpu | 

### For Python script 

```bash
from inference import load_model_and_vocab, predict_from_path

model, ingredient_list = load_model_and_vocab("best_checkpoint.pth.tar", "cuda")
preds, probs, img = predict_from_path("photo.jpg", model, ingredient_list, "cuda")
print(preds)
```
