import ast
from collections import defaultdict
from datasets import load_from_disk

# Load the full MM-Food-100K dataset
dataset = load_from_disk("./mm_food_100k_full")

# Build mapping: ingredient -> master class
# The mapping file was manually cleaned and contains lines like:
#   === chicken ===
#     chicken
#     fried chicken
#   === mushroom ===
#     enoki mushrooms
#     shiitake mushrooms
mapping = {}
current_class = None
with open("./image_quantity.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            current_class = line[4:-4].strip()
        elif current_class:
            if line.startswith("(no matching"):
                continue
            mapping[line] = current_class

# Count how many unique images contain each class
class_image_counts = defaultdict(set)  # class -> set of image indices

for idx, sample in enumerate(dataset):
    raw = sample["ingredients"]
    if isinstance(raw, str):
        try:
            items = ast.literal_eval(raw)
        except:
            items = []
    else:
        items = raw

    # Determine which master classes are present in this image
    image_classes = set()
    for ingr in items:
        if ingr in mapping:
            image_classes.add(mapping[ingr])

    for cls in image_classes:
        class_image_counts[cls].add(idx)

# Total number of images that contain at least one target class
all_valid_indices = set()
for indices in class_image_counts.values():
    all_valid_indices.update(indices)

total_images = len(all_valid_indices)
print(f"Total images with at least one target class: {total_images} out of {len(dataset)}")

# Print class distribution (sorted by frequency)
print("\nUnique images per class:")
sorted_classes = sorted(class_image_counts.items(), key=lambda x: len(x[1]))
for cls, indices in sorted_classes:
    print(f"  {cls:25s} - {len(indices)} images")