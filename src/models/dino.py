# File: src/models/dino.py
"""
DINO SSL model and loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import timm


class DinoLoss(nn.Module):
    """
    DINO loss with center update to prevent collapse.
    """
    def __init__(self, out_dim=256, teacher_temp=0.04, student_temp=0.1, center_m=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_m = center_m
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        t_out = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)
        s_out = F.log_softmax(student_out / self.student_temp, dim=-1)
        loss = -(t_out * s_out).sum(dim=-1).mean()

        with torch.no_grad():
            batch_center = t_out.mean(dim=0, keepdim=True)
            self.center.mul_(self.center_m).add_(batch_center, alpha=1 - self.center_m)

        return loss


class LitDINO(pl.LightningModule):
    """
    PyTorch Lightning module for DINO pretraining.
    """
    def __init__(self, lr=3e-4, out_dim=256, warmup_epochs=5, max_epochs=40):
        super().__init__()
        self.student = timm.create_model('vit_tiny_patch16_224', num_classes=0)
        self.teacher = timm.create_model('vit_tiny_patch16_224', num_classes=0)

        embed_dim = self.student.num_features  # 192
        self.student_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim),
        )
        self.teacher_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim),
        )

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.criterion = DinoLoss(out_dim)
        self.lr = lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

    def training_step(self, batch, _):
        g1, g2 = batch
        feat1 = self.student(g1)
        feat2 = self.student(g2)
        s1 = self.student_head(feat1)
        s2 = self.student_head(feat2)

        with torch.no_grad():
            t_feat1 = self.teacher(g1)
            t_feat2 = self.teacher(g2)
            t1 = self.teacher_head(t_feat1)
            t2 = self.teacher_head(t_feat2)

        loss = 0.5 * (self.criterion(s1, t2) + self.criterion(s2, t1))
        self.log("train_loss", loss, prog_bar=True)

        # momentum update teacher
        m = 0.996
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1 - m)
            for hs, ht in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                ht.data.mul_(m).add_(hs.data, alpha=1 - m)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]
