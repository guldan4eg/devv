"""Model definition and Grad-CAM for the WikiArt style classifier.

Provides helpers to build a ResNet50-based classifier with optional layer
freezing, plus a Grad-CAM implementation for interpretability.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from config import Config


def get_model(
    cfg: Config,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a ResNet50 model with a custom classification head.

    Args:
        cfg: Global configuration object.
        freeze_backbone: If ``True``, freeze all backbone parameters so that
            only the final classifier is trainable (Stage 1).

    Returns:
        A ``torch.nn.Module`` ready for training.
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, cfg.num_classes),
    )

    return model


def unfreeze_for_finetuning(model: nn.Module) -> None:
    """Unfreeze ``layer3``, ``layer4``, and ``fc`` for Stage-2 fine-tuning.

    All other parameters remain frozen.

    Args:
        model: The ResNet50 model returned by :func:`get_model`.
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith(("layer3.", "layer4.", "fc.")):
            param.requires_grad = True


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Return a list of parameters that have ``requires_grad=True``.

    Args:
        model: Any ``torch.nn.Module``.

    Returns:
        List of trainable parameters.
    """
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for ResNet-like models.

    Usage::

        cam = GradCAM(model, target_layer_name="layer4")
        heatmap = cam(input_tensor)           # predicted class
        heatmap = cam(input_tensor, class_idx=3)  # specific class

    Args:
        model: A ResNet model (or similar with named children).
        target_layer_name: Name of the convolutional layer to hook into.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "layer4") -> None:
        self.model = model
        self.model.eval()

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        target_layer = dict(model.named_children())[target_layer_name]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        _module: nn.Module,
        _input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _backward_hook(
        self,
        _module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for *input_tensor*.

        Args:
            input_tensor: A single image tensor of shape ``(1, C, H, W)``.
            class_idx: Target class index.  If ``None``, the predicted class
                is used.

        Returns:
            A 2-D numpy array (H, W) with values in ``[0, 1]``.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        gradients = self._gradients[0]   # (C, h, w)
        activations = self._activations[0]  # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # global average pooling
        cam = (weights[:, None, None] * activations).sum(dim=0)  # weighted sum
        cam = torch.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()
