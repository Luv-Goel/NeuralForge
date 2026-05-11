"""
NeuralForge — Data Augmentations
=================================

Common data augmentation techniques used during architecture search.
Includes Cutout (DeVries & Taylor, 2017) which is standard in DARTS.

References:
    - Improved Regularization of Convolutional Neural Networks with Cutout
      (DeVries & Taylor, 2017)
    - DARTS: Differentiable Architecture Search (Liu et al., 2019)
"""

import numpy as np
import torch


class Cutout:
    """Cutout augmentation — randomly masks a square region of the input.

    Applies a square mask of size `length` to a random location in the
    image. This acts as a strong regularizer for convolutional networks.

    Args:
        length: Side length of the square mask in pixels.
        fill_value: Pixel value to fill the mask with (default: 0).

    Reference:
        Improved Regularization of Convolutional Neural Networks
        with Cutout (DeVries & Taylor, 2017)
    """

    def __init__(self, length: int, fill_value: int = 0):
        self.length = length
        self.fill_value = fill_value

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout to a single image tensor (C, H, W)."""
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = self.fill_value
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask
