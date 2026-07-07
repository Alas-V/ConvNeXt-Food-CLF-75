import ast
from datasets import load_from_disk

# STEP 1: Load the entire MM-Food-100K dataset
# The dataset contains ~100k images with ingredient labels stored as strings
# or lists in the "ingredients" field.
dataset = load_from_disk("./mm_food_100k_full")

# STEP 2: Extract all unique ingredient names from the dataset
# We collect every distinct ingredient name to later search for keywords.
all_ingredients = set()

for sample in dataset:
    raw = sample["ingredients"]
    
    # Some samples store ingredients as a string representation of a list,
    # others as a native Python list. We handle both formats.
    if isinstance(raw, str):
        try:
            items = ast.literal_eval(raw)   # Convert string like "['apple','banana']" to list
        except (ValueError, SyntaxError):
            continue                        # Skip malformed entries
    else:
        items = raw                         # Already a list
    
    all_ingredients.update(items)           # Add all ingredients to the set

# STEP 3: Define target classes (from the segmentation model)
# These 75+ categories are the master classes we want to keep.
# The full list was manually curated and corresponds to the final 75 classes.
keywords_list = [
    "candy", "egg tart", "french fries", "chocolate", "biscuit", "cookie",
    "popcorn", "pudding", "ice cream", "cheese", "butter", "cake", "wine",
    "milkshake", "coffee", "juice", "milk", "tea", "almond", "walnut", "peanut",
    "cranberry", "strawberry", "cherry", "blueberry", "raspberry", "grape",
    "beans", "cashew", "soy", "egg", "apple", "date", "apricot", "avocado",
    "banana", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple",
    "kiwi", "melon", "orange", "watermelon", "steak", "pork",
    "chicken", "duck", "sausage", "meat", "lamb", "sauce", "crab", "fish",
    "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza",
    "hanamaki", "baozi", "wonton", "dumplings", "pasta", "noodle", "rice",
    "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato",
    "kelp", "seaweed", "onion", "rape", "ginger", "okra", "lettuce", "pumpkin",
    "cucumber", "white radish", "carrot", "asparagus", "bamboo", "broccoli",
    "celery", "mint", "peas", "cabbage", "sprouts", "pepper",
    "mushroom", "shiitake", "enoki", "salad"
]

# STEP 4: Search for matches and save results
# For each keyword, we find all ingredients from the dataset that contain
# the keyword (case-insensitive). This creates a mapping file that will be
# manually reviewed and cleaned.
output_file = "dirty_master_classes.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for keyword in keywords_list:
        matches = []
        kw_lower = keyword.lower()
        
        for ingr in all_ingredients:
            if kw_lower in ingr.lower():
                matches.append(ingr)
        
        f.write(f"=== {keyword} ===\n")
        if matches:
            for m in sorted(matches):
                f.write(f"  {m}\n")
        else:
            f.write("  (no matching found)\n")
        f.write("\n")

print(f"Search completed. Results saved to {output_file}")