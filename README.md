# ConvNeXt-CLF-75

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-Alas--V%2Fconvnext--clf--75-ffd21e?style=flat-square)](https://huggingface.co/Alas-V/convnext-clf-75)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)]()
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-red?style=flat-square&logo=pytorch)]()

[![F1 Macro](https://img.shields.io/badge/F1%20Macro-0.6546-brightgreen?style=flat-square)]()
[![mAP](https://img.shields.io/badge/mAP-0.7142-success?style=flat-square)]()

**Multi‑label food ingredient classifier** built on **ConvNeXt‑Tiny** with **CBAM attention** and **GeM pooling**.  
Trained on a cleaned subset of [MM‑Food‑100K](https://huggingface.co/datasets/Codatta/MM-Food-100K) for **75 ingredients**.

> 🚀 **Ready‑to‑use model and quick start**: [Hugging Face Model Card](https://huggingface.co/Alas-V/convnext-clf-75)

---

## 📖 What This Model Does

**ConvNeXt‑CLF‑75** takes an RGB food image, resizes it to 640×640, and predicts which of the **75 known ingredients** are present. It performs **multi‑label classification** – a single dish can contain multiple ingredients (e.g., “rice”, “sauce”).

Internally, the model produces a compact feature vector (embedding) that can be reused for downstream tasks. In our pipeline, this embedding is fed into a **segmentation network** to guide pixel‑wise mask prediction, forming part of a calorie‑estimation system.

- **Input:** RGB image (any size, automatically resized to 640×640 and normalized with ImageNet statistics)
- **Output:**
  - A list of ingredient names (threshold 0.5 by default)
  - Raw probability scores for each of the 75 classes
  - Optional: feature embedding from the penultimate layer (for transfer learning)

## Supported 75 Classes (in alphabetical order)
| | | | | |
| :--- | :--- | :--- | :--- | :--- |
| almond | cheese butter | grape | onion | sauce |
| apple | chicken duck | ice cream | orange | sausage |
| asparagus | chocolate | juice | pasta | seaweed |
| avocado | cilantro mint | kiwi | peach | shellfish |
| banana | coffee | lamb | peanut | shrimp |
| beans | corn | lemon | pear | soup |
| biscuit | crab | lettuce | peas | spring onion |
| bread | cucumber | mango | pepper | sprouts |
| broccoli | date | meat | pineapple | steak |
| cabbage | egg | melon | pizza | strawberry |
| cake | eggplant | milk | pork | tea |
| carrot | fish | mushroom | potato | tofu |
| cashew | french fries | noodle | pumpkin | tomato |
| cauliflower | garlic | okra | rice | watermelon |
| celery stick | ginger | olives | salad | wonton dumplings |

---
## 🧀 Prediction Examples
TODO

---

## 🧹 Dataset Challenges

The original [MM‑Food‑100K](https://huggingface.co/datasets/Codatta/MM-Food-100K) dataset is extremely noisy:
- Over 4,000 raw ingredient labels, many of which are visually indistinguishable (e.g., “enoki mushrooms” vs. “shiitake mushrooms”)
- Severe class imbalance (some classes have fewer than 10 images)
- Duplicates and inconsistent annotations

Designed a multi‑step cleaning pipeline that **reduces the chaos to a clean, balanced 75‑class dataset** suitable for training.

**The full pipeline is documented in the [`dataset_preparation/`](dataset_preparation/) folder.**  
It includes:
- Manual curation of target classes
- Automated extraction of matching ingredients
- Manual merging and removal of irrelevant entries
- Filtering classes with fewer than 100 images
- Building a multi‑hot label dataset and vocabulary

If you want to understand every detail or adapt the pipeline to your own food dataset, start with the README inside that folder.

---

## 📊 Model Performance

Final metrics after a 4‑stage training process:

| Metric | Value |
| :--- | :--- |
| **F1 (macro)** | 0.6546 |
| **mAP** | 0.7142 |
| Val Loss | 0.0489 |

**Complete training logs, hyperparameters, and stage‑by‑stage results are documented in the [`model/`](model/) folder.** 

---

## 💻 Hardware & Environment

All training and inference were performed on a single consumer GPU:

| Component | Detail |
| :--- | :--- |
| GPU | NVIDIA RTX 2060 (6 GB VRAM) |
| CPU | Intel Xeon E5‑2660 v3 |
| OS | CachyOS (Arch‑based) |
| Python | 3.10+ |
| PyTorch | 2.5 |
| timm | 1.0.9 |
| Mixed precision | AMP (`torch.amp`) |

Despite the limited 6 GB VRAM, the model trains comfortably with a batch size of 10 at 640×640 resolution.

---

## 🙏 Acknowledgements

[MM‑Food‑100K](https://huggingface.co/datasets/Codatta/MM-Food-100K) for the original dataset.

[timm](https://github.com/huggingface/pytorch-image-models) for the ConvNeXt backbone.

[Albumentations](https://albumentations.ai/) for image augmentations.

---

## 📄 License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
