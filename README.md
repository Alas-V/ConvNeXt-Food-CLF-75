# ConvNeXt-Food-CLF-75
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-Alas--V%2FConvNeXt--Food--CLF--75-ffd21e?style=flat-square)](https://huggingface.co/Alas-V/ConvNeXt-Food-CLF-75)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)

[![F1 Macro](https://img.shields.io/badge/F1%20Macro-0.6546-brightgreen?style=flat-square)]()
[![mAP](https://img.shields.io/badge/mAP-0.7142-success?style=flat-square)]()

[![HF Space](https://img.shields.io/badge/HF%20Space-🚀%20Try%20for%20free-FFB800?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Alas-V/ConvNeXt-Food-CLF-75)


**Multi‑label food ingredient classifier** built on **ConvNeXt‑Tiny** with **CBAM attention** and **GeM pooling**.  
Trained on a cleaned subset of [MM‑Food‑100K](https://huggingface.co/datasets/Codatta/MM-Food-100K) for **75 ingredients**.

**📊 Key Metrics:** Macro F1 = 0.6546 | mAP = 0.7142

> 🚀 **Ready‑to‑use model and quick start**: [Hugging Face Model Card](https://huggingface.co/Alas-V/ConvNeXt-Food-CLF-75)

**👉 [Try it live on Spaces](https://huggingface.co/spaces/Alas-V/ConvNeXt-Food-CLF-75) - no installation required!**

---

## 📖 What This Model Does

**ConvNeXt-Food-CLF-75** takes an RGB food image, resizes it to 640×640, and predicts which of the **75 known ingredients** are present. It performs **multi‑label classification** – a single dish can contain multiple ingredients (e.g., “rice”, “sauce”).

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

### 🚀 Quick Inference
For ready‑to‑run commands, see the [Hugging Face Model Card](https://huggingface.co/Alas-V/ConvNeXt-Food-CLF-75) or the [`model/`](model/) folder in this repository or [Try it live on Spaces](https://huggingface.co/spaces/Alas-V/ConvNeXt-Food-CLF-75) - no installation required!**

---

## 🧀 Prediction Examples

**Correct predictions**


<img src="https://github.com/user-attachments/assets/5e76c336-2790-4028-860d-643dd8dd830b" width="600"> <img src="https://github.com/user-attachments/assets/793c9be7-b35e-46fe-8acb-5774de1849fe" width="600"> <img src="https://github.com/user-attachments/assets/e42850b6-2891-46ac-9a2d-f960a27d3ba4" width="600"> <img src="https://github.com/user-attachments/assets/9361a11e-9267-4747-9ecc-afa0e008b4c2" width="600"> <img src="https://github.com/user-attachments/assets/3dbba780-5a33-4a91-9f33-a46db5e57a75" width="600">

---

### **⚠️ The model can occasionally hallucinate ingredients or fail to detect all ingredients present.**

<img src="https://github.com/user-attachments/assets/693eea71-ac2b-44fc-9f0b-839beaf2aebb" width="600"> <img src="https://github.com/user-attachments/assets/d8a7de56-eaed-4351-8027-4007b4de0f0d" width="600">

---

## 🙏 Acknowledgements

[MM‑Food‑100K](https://huggingface.co/datasets/Codatta/MM-Food-100K) for the original dataset.

[timm](https://github.com/huggingface/pytorch-image-models) for the ConvNeXt backbone.

[Albumentations](https://albumentations.ai/) for image augmentations.

---


## Citation

If you use this model or the cleaned dataset in your research or project, please cite:

```bibtex
@misc{alasv2026convnextfoodclf75,
  author       = {Artem V.},
  title        = {ConvNeXt-Food-CLF-75: A Multi-Label Food Ingredient Classifier},
  year         = {2026},
  howpublished = {\url{https://github.com/Alas-V/ConvNeXt-Food-CLF-75}},
  note         = {Model weights available at \url{https://huggingface.co/Alas-V/ConvNeXt-Food-CLF-75}}
}
```

---

## 📄 License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

