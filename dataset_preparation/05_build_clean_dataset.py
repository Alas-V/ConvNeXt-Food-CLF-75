import ast
import pickle
from collections import defaultdict
from datasets import load_from_disk, Dataset, DatasetDict

# Load the original MM-Food-100K dataset
original_dataset_path = "/mnt/SSD120/mm_food_100k_full"
mapping_file = "./search_results_clear.txt"
new_dataset_path = "./mm_food_100k_clean"

print("1/4 Loading dataset...")
original_dataset = load_from_disk(original_dataset_path)

print("2/4 Building mapping from ingredient to master class...")
mapping = {}          # ingredient_name -> master_class
current_class = None
ignored_lines = {"", "(no matching found)", "(no matching found)", "(none)"}

with open(mapping_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            current_class = line[4:-4].strip()
        elif current_class and line not in ignored_lines:
            mapping[line] = current_class

canonical_classes = sorted(set(mapping.values()))
class_to_idx = {cls: i for i, cls in enumerate(canonical_classes)}
NUM_CLASSES = len(canonical_classes)
print(f"   Found {len(mapping)} mapped ingredients for {NUM_CLASSES} base classes.")

print("3/4 Filtering dataset and converting labels to multi-hot vectors...")
def filter_and_transform(sample):
    # Extract ingredient list from the sample
    raw = sample["ingredients"]
    if isinstance(raw, str):
        try:
            items = ast.literal_eval(raw)
        except Exception:
            items = []
    else:
        items = raw

    # Determine which master classes are present
    selected_classes = set()
    for ingr in items:
        if ingr in mapping:
            selected_classes.add(mapping[ingr])

    # If no target class is present, discard the sample
    if not selected_classes:
        return None

    # Build multi-hot label vector
    label = [0] * NUM_CLASSES
    for cls_name in selected_classes:
        label[class_to_idx[cls_name]] = 1

    return {"image": sample["image"], "label": label}

new_dataset = original_dataset.map(
    filter_and_transform,
    remove_columns=original_dataset.column_names,
    batched=False,
    load_from_cache_file=False,
    desc="Filtering"
)

# Remove samples that returned None (no target classes)
new_dataset = new_dataset.filter(lambda x: x is not None, desc="Removing empty")
print(f"   Remaining images: {len(new_dataset)}")

print("4/4 Saving new dataset and vocabulary...")
new_dataset.save_to_disk(new_dataset_path)

# Save class vocabulary for later use in training and inference
vocab_dict = {
    "ingredient_list": canonical_classes,
    "ingredient_to_idx": class_to_idx,
    "NUM_INGREDIENTS": NUM_CLASSES
}
with open(f"{new_dataset_path}/ingredient_dict.pkl", "wb") as f:
    pickle.dump(vocab_dict, f)

print("\nDone. Clean dataset saved to:", new_dataset_path)
print(f"   Total classes: {NUM_CLASSES}")