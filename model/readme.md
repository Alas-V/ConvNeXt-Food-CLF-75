
## **Training Procedure**

## Model Structure

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
| Scheduler | CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e‑6) |
|           | OneCycleLR (max_lr=3e-3, pct_start=0.3, div_factor=10, final_div_factor=100)|
| Weight decay | 1e‑4 |
| Batch size | 10 |
| Input resolution | 640×640 |
| MixUp | Probability 0.5 (alpha=0.2) |
| Mixed precision | Yes (AMP) |

### Training Stages

1. **Feature Extraction (28 epochs):** Only the classification head is trained (LR=1e‑4 for head, backbone frozen, scheduler OneCycleLR).
2. **Discriminative Fine‑Tuning (8 epochs):** Head + stages 2 and 3 of backbone (LR: head 1e‑4, backbone 1e‑5 scheduler OneCycleLR).
3. **Staged Unfreezing (1 epoch):** Head (LR: head 1e‑4, scheduler CosineAnnealingWarmRestarts).
4. **Final Fine‑Tuning (5 epochs):** All layers unfrozen (LR=5e‑5, cheduler CosineAnnealingWarmRestarts).

### Training Results by Stage

| Stage | Epoch | F1 (macro) | mAP | Val Loss |
| :--- | :--- | :--- | :--- | :--- |
| Feature Extraction | 28 | 0.3533 | 0.5067 | 0.0637 |
| Discriminative Fine-Tuning | 36 | 0.6121 | 0.7017 | 0.0469 |
| Staged Unfreezing | 37 | 0.6426 | 0.7097 | 0.0479 |
| Final Feature Extraction after SU | 40 | **0.6546** | **0.7142** | 0.0489 |


---


## Hardware and Training Time

- **GPU:** NVIDIA RTX 2060 (6GB VRAM)
- **CPU:** Xeon e5 2660 v3
- **OS:** CachyOS (Arch‑based)  
- **Framework:** PyTorch 2.5 + torchvision 0.20 + timm 1.0.9
- **Training time:** Approximately 1 hour for 1 epoch
- **Batch size:** 10 
- **Mixed precision:** AMP via `torch.amp`
