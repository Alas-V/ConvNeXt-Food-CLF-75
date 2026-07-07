import torch
from .model import create_classifier
from .config import get_config


def predict_ingredients(
    image_tensor, model, ingredient_list, threshold=0.5, device,
):
    """
    Run inference on a batch of images and return predicted ingredient names.

    Args:
        image_tensor: Tensor of shape (1, C, H, W) representing a single image batch.
        model: The trained classifier model.
        ingredient_list: List of ingredient class names (order must match model output).
        threshold: Probability threshold for binary classification (default: 0.5).
        device: Device to run inference on ('cuda' or 'cpu').

    Returns:
        List of lists of ingredient names for each image in the batch.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Get raw logits from the model
        logits = model(image_tensor)
        # Convert logits to probabilities via sigmoid (multi-label)
        probs = torch.sigmoid(logits)
        # Apply threshold to get binary predictions
        preds = (probs > threshold).long().cpu().numpy()

    # Convert binary predictions to ingredient names
    result = []
    for pred in preds:
        ingredients = [ingredient_list[i] for i in range(len(pred)) if pred[i] == 1]
        result.append(ingredients)
    return result


def load_model_for_inference(checkpoint_path, device):
    """
    Load a trained model from a checkpoint file for inference.

    Args:
        checkpoint_path: Path to the saved model checkpoint (.pth file).
        device: Device to load the model on ('cuda' or 'cpu').

    Returns:
        The loaded model in evaluation mode.
    """
    cfg = get_config()
    # Create the model architecture with the correct number of classes
    # and load the checkpoint weights
    model = create_classifier(cfg["NUM_INGREDIENTS"], checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()
    return model