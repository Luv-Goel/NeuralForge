"""
NeuralForge — PyTorch Lightning Integration
=============================================

Optional integration with PyTorch Lightning for easier training.
Provides LightningModule wrappers around NAS search and evaluation.

Requires: pytorch-lightning >= 1.8
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

    # Create placeholder
    class pl:
        class LightningModule:
            pass


if LIGHTNING_AVAILABLE:

    class DARTSLightningModule(pl.LightningModule):
        """PyTorch Lightning wrapper for DARTS search."""

        def __init__(
            self,
            model: nn.Module,
            arch_parameters: list,
            learning_rate: float = 0.025,
            arch_learning_rate: float = 3e-4,
            weight_decay: float = 3e-4,
            arch_weight_decay: float = 1e-3,
        ):
            super().__init__()
            self.model = model
            self.arch_parameters = arch_parameters
            self.lr = learning_rate
            self.arch_lr = arch_learning_rate
            self.wd = weight_decay
            self.arch_wd = arch_weight_decay

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx, optimizer_idx=0):
            x, y = batch
            logits = self(x)
            loss = nn.functional.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log("train_loss", loss)
            self.log("train_acc", acc)
            return loss

        def configure_optimizers(self):
            w_optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.wd,
            )
            a_optim = torch.optim.Adam(
                self.arch_parameters,
                lr=self.arch_lr,
                betas=(0.5, 0.999),
                weight_decay=self.arch_wd,
            )
            return [w_optim, a_optim]

else:

    class DARTSLightningModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pytorch-lightning is required. "
                "Install with: pip install pytorch-lightning"
            )
