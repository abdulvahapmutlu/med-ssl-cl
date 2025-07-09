# File: src/utils.py
"""
Utilities: progress callback & AUC computation.
"""
import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC


class BatchProgressCallback(pl.callbacks.Callback):
    def on_train_epoch_start(self, *_):
        self.start = time.time()

    def on_train_batch_end(self, trainer, *_):
        if trainer.global_step % 50 == 0:
            mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            ex_s = (trainer.accumulate_grad_batches * trainer.train_dataloader.batch_size * 50) \
                   / max(1e-3, time.time() - self.start)
            trainer.progress_bar_callback.progress.print(
                f"🖥️ GPU {mb:.0f} MB  ⚡ {ex_s:.1f} img/s"
            )
            self.start = time.time()


def compute_auc(model, dataset, device):
    model.eval().to(device)
    auc = BinaryAUROC().to(device)
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=64):
            preds = torch.sigmoid(model(x.to(device)).squeeze(1))
            auc.update(preds, y.to(device).int().squeeze(1))
    return auc.compute().item()
