## Dataset Preparation

### Source Data
- **Original dataset:** [MM-Food-100K](https://huggingface.co/datasets/Codatta/MM-Food-100K) — a large-scale food dataset containing ~100,000 images with over 4,000 ingredient labels.
- **Challenge:** The dataset is highly noisy, contains many duplicates, and has severe class imbalance. Many ingredients are visually identical or semantically overlapping (e.g., "enoki mushrooms", "shiitake mushrooms" → "mushroom").

### Cleaning Pipeline

The dataset was cleaned in several stages to create a high-quality subset of 75 classes, designed specifically for use as an auxiliary classification network in a segmentation pipeline.

---

#### Step 1: Define target classes
**File:** `01_target_classes.txt`  
**Description:** A manually curated list of 103 ingredient classes (taken from the segmentation model). These are the master classes.

#### Step 2: Extract raw mapping from the dataset
**Script:** `02_extract_raw_mapping.py`  
**Input:** Original MM-Food-100K dataset (loaded via HuggingFace `datasets`) + `01_target_classes.txt`.  
**Process:** For each target class, it searches the entire dataset for all ingredient names that contain the class name (case‑insensitive).  
**Output:** `02_mapping_raw.txt` — a raw list of matches, e.g.:
=== chicken ===
chicken
fried chicken
roast chicken
...

    
#### Step 3: Manual cleaning
**File:** `04_mapping_cleaned.txt` ( manually edit `02_mapping_raw.txt`)  
**Process:** Remove irrelevant matches (e.g., `cheesecake` from the `cheese` group), merge semantically similar classes (e.g., `enoki mushrooms` + `shiitake mushrooms` → `mushroom`). The final mapping contains **75** master classes.

#### Step 4: Count images per class
**Script:** `03_count_images_per_class.py`  
**Input:** `04_mapping_cleaned.txt` + original dataset.  
**Process:** For each master class, count how many unique images contain at least one ingredient from that class.  
**Output:** `03_class_counts.txt` — distribution of images per class. Classes with fewer than **100** images are removed.

#### Step 5: Build the clean dataset
**Script:** `05_build_clean_dataset.py`  
**Input:** Original dataset + `04_mapping_cleaned.txt`.  
**Process:** 
- For each image, extract ingredients and map them to master classes.
- Create a multi‑hot label vector (75 bits).
- Discard images with no target class.
- Save the new dataset (images + labels) and a vocabulary file (`ingredient_dict.pkl`).

**Output:** Clean dataset with:
- **75 classes**
- **~88k total images** (86.3k train, 2k validation — split performed later in the training script)

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


### Data Splits
| Split | Images |
| :--- | :--- |
| Train | 86.3k |
| Validation | 2k |

The validation set is a fixed subset of 2,000 images (saved in the checkpoint during training for reproducibility).

### Preprocessing
- **Resizing:** Images are resized to 640×640 during training (random crop) and validation (resize).
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
- **Augmentations:** RandomResizedCrop, HorizontalFlip, Rotate (15°), ColorJitter.

### Implementation Details
The dataset is loaded via the `IngredientDataset` class, which handles:
- Loading images in various formats (PIL, numpy, bytes).
- Converting multi‑hot labels from the dataset into `float32` tensors.
- Applying augmentations via Albumentations.

The `create_loaders` function builds both training and validation data loaders, with support for restoring validation indices from checkpoint files for reproducible experiments.
