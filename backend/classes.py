"""
This module contains classes and utilities for working with neural networks
using PyTorch's nn module.
"""

from torch import nn
from timm.data.transforms_factory import create_transform
from transformers import AutoModel
from constants import INPUT_RESOLUTION
from pydantic import BaseModel


# Block MambaClassifier
class MambaClassifier(nn.Module):
    """
    A PyTorch module for a classification model that uses a backbone network and
    optional fully connected (fc) layers before the final classifier.

    Attributes:
        backbone (AutoModel): The backbone model used for feature extraction.
        config (Config): Configuration object from the backbone model.
        fc_layer (int): Number of fully connected layers before the classifier.
                        If None, no additional layers are added.
        fc_layers (nn.ModuleList): A list of fully connected layers with ReLU
                                   activation and dropout (if fc_layer > 0).
        classifier (nn.Linear): The final linear layer for classification.

    Methods:
        create_transform(training: bool, auto_augment=None) -> Callable:
            Creates a data transformation pipeline for preprocessing input images.

        forward(x: Tensor) -> Tensor:
            Defines the forward pass of the model, applying the backbone,
            optional fully connected layers, and the final classifier.
    """

    def __init__(
        self,
        backbone: AutoModel,
        num_classes: int,
        hidden_dim: int,
        fc_layer: int = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = self.backbone.config
        self.fc_layer = fc_layer
        if fc_layer:
            self.fc_layers = nn.ModuleList()
            for i in range(fc_layer):
                if i == 0:
                    self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
                else:
                    self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(0.1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def create_transform(self, training: bool, auto_augment=None):
        """
        Creates a data transformation pipeline for image preprocessing.

        Args:
            training (bool): Indicates whether the transformation is for training or evaluation.
                             If True, additional augmentations like ColorJitter may be applied.
            auto_augment (str, optional): Specifies the auto-augmentation policy to use.
                                           For example, "rand-m9-mstd0.5-inc1". Defaults to None.

        Returns:
            Callable: A transformation function that can be applied to image data.
        """
        transform = create_transform(
            input_size=INPUT_RESOLUTION,
            is_training=training,  # Add a ColorJitter augmentation during training
            mean=self.config.mean,
            std=self.config.std,
            crop_mode=self.config.crop_mode,
            crop_pct=self.config.crop_pct,
            auto_augment=auto_augment,  # "rand-m9-mstd0.5-inc1"
        )
        return transform

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output logits after passing through the backbone,
                          optional fully connected layers, and the classifier.
        """
        out_avg_pool, _ = self.backbone(x)
        if hasattr(self, "fc_layers"):
            for layer in self.fc_layers:
                out_avg_pool = layer(out_avg_pool)
        logits = self.classifier(out_avg_pool)
        return logits


# BaseModel for prediction response
class PredictionResponse(BaseModel):
    received_label: str
    predicted_label: str

# End of code