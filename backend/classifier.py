"""
PyTorch equivalent of the Keras MNIST classifier.

Architecture (matches train_classifier.py):
  Conv2d(1→32) → ReLU → Conv2d(32→32) → ReLU → MaxPool(2)
  Conv2d(32→64) → ReLU → Conv2d(64→64) → ReLU → MaxPool(2)
  Flatten → Linear(3136→256) → ReLU → Linear(256→10) → Softmax
"""

import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1)
        x = self.head(x)
        return torch.softmax(x, dim=1)
