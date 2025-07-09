# File: src/models/continual.py
"""
Continual learning via replay + EWC.
"""
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset


class BalancedDataset(Dataset):
    """
    Mix current dataset with replay buffer and optional oversampling.
    """
    def __init__(self, cur_ds, rep_x, rep_y, oversample=False):
        self.cur, self.rep, self.repy = cur_ds, rep_x, rep_y
        self.oversample = oversample and hasattr(cur_ds, 'pos')
        self.pos = cur_ds.pos if self.oversample else None
        self.neg = cur_ds.neg if self.oversample else None
        self.n = max(len(self.cur), len(self.rep)) * 2 if self.rep else len(self.cur)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.rep and idx % 2 == 1:
            r = (idx // 2) % len(self.rep)
            return self.rep[r], self.repy[r]
        if self.oversample:
            idx_cur = random.choice(self.pos) if random.random() < 0.67 else random.choice(self.neg)
        else:
            idx_cur = (idx // 2) % len(self.cur)
        return self.cur[idx_cur]


class CLModel(pl.LightningModule):
    """
    Continual learning model with EWC and replay.
    """
    def __init__(self, ckpt_path, init_lr, wd=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['ckpt_path'])

        # SSL encoder
        self.encoder = timm.create_model(
            'vit_tiny_patch16_224', num_classes=0, global_pool='token'
        )
        sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
        self.encoder.load_state_dict(
            {k.replace('student.', ''): v for k, v in sd.items() if k.startswith('student.')},
            strict=False
        )

        with torch.no_grad():
            z = self.encoder(torch.zeros(1,3,224,224))
            if z.ndim == 3: z = z[:,0]
        self.head = nn.Linear(z.shape[-1], 1)

        self.base_lr = init_lr
        self.wd = wd
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.mem_x, self.mem_y = [], []
        self.ewc_mu, self.ewc_f = None, None

    def set_pos_weight(self, ds):
        pos = sum(ds.targets) if hasattr(ds, 'targets') else sum([y for _, y in ds])
        neg = len(ds) - pos
        pw = torch.tensor([neg / pos]) if pos > 0 else torch.tensor([1.0])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw.to(self.device))

    @torch.no_grad()
    def add_to_memory(self, dataset):
        sel = random.sample(range(len(dataset)), k=min(len(dataset), 200))
        for xs, ys in DataLoader(Subset(dataset, sel), batch_size=64, num_workers=0):
            for xi, yi in zip(xs, ys):
                self.mem_x.append(xi.cpu()); self.mem_y.append(yi.cpu())
        self.mem_x, self.mem_y = self.mem_x[-200:], self.mem_y[-200:]

    def compute_fisher(self, dataset, samples=600):
        fisher = {n: torch.zeros_like(p) for n,p in self.named_parameters()}
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        cnt = 0
        for x,y in loader:
            x,y = x.to(self.device), y.to(self.device)
            loss = self.loss_fn(self(x).squeeze(1), y.squeeze(1))
            self.zero_grad(); loss.backward()
            for n,p in self.named_parameters():
                fisher[n] += p.grad.detach()**2
            cnt += 1
            if cnt*32 >= samples: break
        for n in fisher: fisher[n] /= cnt
        self.ewc_mu = [p.detach().clone() for p in self.parameters()]
        self.ewc_f = [fisher[n] for n,_ in self.named_parameters()]

    def forward(self, x):
        z = self.encoder(x)
        if z.ndim == 3: z = z[:,0]
        return self.head(z)

    def _total_loss(self, logits, y):
        loss = self.loss_fn(logits, y)
        if self.ewc_mu is not None:
            loss += 0.05 * sum((f*(p-m).pow(2)).sum() for p,m,f in zip(self.parameters(), self.ewc_mu, self.ewc_f))
        return loss

    def training_step(self, batch, _):
        x,y = batch
        loss = self._total_loss(self(x).squeeze(1), y.squeeze(1))
        self.log('loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return [opt], [sched]
