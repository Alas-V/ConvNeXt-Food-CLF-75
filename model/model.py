import timm
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention module from CBAM.
    Uses both average and max pooling, then applies an MLP to generate channel weights.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """
    Spatial attention module from CBAM.
    Generates a spatial attention map using average and max pooling across channels.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention followed by spatial attention in sequence.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class GeM(nn.Module):
    """
    Generalized Mean (GeM) pooling.
    A learnable pooling method that generalizes average and max pooling.
    The parameter p is learnable and controls the pooling behaviour.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return (x.clamp(min=self.eps).pow(self.p).mean(dim=[2, 3])).pow(1.0 / self.p)


class CustomHead(nn.Module):
    """
    Custom classification head for ConvNeXt.
    Applies CBAM attention, GeM pooling, and a two-layer MLP with dropout.
    """
    def __init__(self, in_channels, num_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.cbam = CBAM(in_channels)
        self.pool = GeM(p=3)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.cbam(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_classifier(num_ingredients, checkpoint_path=None, unfreeze_stage=None):
    """
    Creates a ConvNeXt-Tiny based classifier with a custom head.

    Args:
        num_ingredients: Number of output classes (ingredients).
        checkpoint_path: Optional path to a checkpoint file to load weights from.
        unfreeze_stage: Controls which parts of the model are trainable:
            - "FE": Only the head is trainable (Feature Extraction).
            - "Disc_FT": Head + stages 2 and 3 are trainable (Discriminative Fine-Tuning).
            - "Staged_unfreezing": Only stages 2 and 3 are trainable (staged unfreezing).
            - None: All layers are frozen (head only will be used for inference after loading weights).
    """
    # Load the pretrained ConvNeXt-Tiny backbone
    model = timm.create_model("convnext_tiny", pretrained=True)
    in_features = model.head.fc.in_features

    # Replace the default head with our custom head
    model.head = CustomHead(in_features, num_ingredients)

    # Load checkpoint if provided (after head is replaced)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Load state dict, with strict=False to handle potential mismatches
        # when loading weights from a checkpoint saved with a different head config
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific parts based on the selected stage
    if unfreeze_stage == "FE":
        # Only the head is trainable
        for param in model.head.parameters():
            param.requires_grad = True

    elif unfreeze_stage == "Disc_FT":
        # Head and the last two stages of the backbone are trainable
        for name, param in model.named_parameters():
            if name.startswith("head.") or "stages.2" in name or "stages.3" in name:
                param.requires_grad = True

    elif unfreeze_stage == "Staged_unfreezing":
        # Only the last two stages of the backbone are trainable (head stays frozen)
        for name, param in model.named_parameters():
            if "stages.2" in name or "stages.3" in name:
                param.requires_grad = True

    # Note: If unfreeze_stage is None, all layers remain frozen (useful for inference only)

    return model