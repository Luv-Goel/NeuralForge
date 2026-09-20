import torch
import numpy as np

class Cutout:
    """Randomly mask out one or more patches from an image.
    
    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with a cutout region.
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
