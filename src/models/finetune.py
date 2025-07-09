# File: src/models/finetune.py
"""
Finetuning ViT-Tiny with SSL weights on Shenzhen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchmetrics
import pytorch_lightning as pl


class LitTBFinetune(pl.LightningModule):
    """
    Lightning module for finetuning SSL pre-trained ViT on Shenzhen.
    """
    def __init__(self, ckpt_path, freeze_epochs=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = timm.create_model(
            "vit_tiny_patch16_224", num_classes=0, global_pool="token"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        self.encoder.load_state_dict(
            {k.replace("student.", ""): v for k, v in ckpt.items()
             if k.startswith("student.")},
            strict=False,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.encoder(dummy)
            if feat.ndim == 3:
                feat = feat[:, 0]
            self.feat_dim = feat.shape[-1]

        self.head = nn.Linear(self.feat_dim, 1)
        self.freeze_epochs, self.lr = freeze_epochs, lr
        self.auc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        z = self.encoder(x)
        if z.ndim == 3:
            z = z[:, 0]
        return self.head(z).squeeze(1)

    def training_step(self, batch, _):
        x, y = batch
        loss = F.binary_cross_entropy_with_logits(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        preds = torch.sigmoid(self(x))
        self.auc.update(preds, y.int())

    def on_validation_epoch_end(self):
        val_auc = self.auc.compute(); self.auc.reset()
        self.log("val_auc", val_auc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        freeze = self.current_epoch < self.freeze_epochs
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
