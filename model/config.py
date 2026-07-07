def get_config():
    return {
        "BATCH_SIZE": 10,
        "VAL_SUBSET_SIZE": 2000,
        "IMG_SIZE": 640,
        "WEIGHT_DECAY": 1e-4,
        "EPOCHS": 5,
        "HEAD_LR": 1e-5,
        "BACKBONE_LR": 5e-6,
        "DATA_DIR": "./mm_food_100k_clean",
        "VOCAB_PATH": "./mm_food_100k_clean/ingredient_dict.pkl",
        "OLD_CHECKPOINT_PATH": "./feature_ectraction_after_su.pth.tar",
        "NEW_CHECKPOINT_PATH": "./feature_ectraction_after_su.pth.tar",
        "SEED": 42,
        "NUM_INGREDIENTS": 75,
    }
