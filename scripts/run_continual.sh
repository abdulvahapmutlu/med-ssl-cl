# File: scripts/run_continual.sh
#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/continual_mias.yaml}
echo "📝 Continual Learning pipeline with config $CONFIG"

python - <<EOF
import yaml
import pandas as pd, numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.datasets import CXRCSV, MIASCSV
from src.models.continual import CLModel, BalancedDataset
from src.utils import compute_auc

conf = yaml.safe_load(open("$CONFIG"))
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_cxr_ds(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    labels = df["findings"].apply(lambda s: 0 if s.lower().strip()=="normal" else 1)
    tr_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=0.2,
        stratify=labels, random_state=conf["dataloader"]["seed"]
    )
    return (
        CXRCSV(df.iloc[tr_idx], Path(img_dir), train=True),
        CXRCSV(df.iloc[val_idx], Path(img_dir), train=False)
    )

# Phase 0: Shenzhen
shen_tr, shen_val = get_cxr_ds(
    conf["data"]["shen_csv_path"], conf["data"]["shen_img_dir"]
)
model = CLModel(
    ckpt_path=conf["model"]["ssl_ckpt_path"],
    init_lr=conf["training"]["lr"]["shen"],
    wd=conf["model"]["ewc_lambda"]
)

def run_phase(dataset, epochs, lr, t_max):
    model.set_pos_weight(dataset)
    # dynamic optimizer/scheduler
    def _cfg(self):
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        return [opt], [sched]
    model.configure_optimizers = _cfg.__get__(model, model.__class__)
    pl.Trainer(
        max_epochs=epochs,
        accelerator=conf["trainer"]["accelerator"],
        devices=conf["trainer"]["devices"],
        log_every_n_steps=conf["trainer"]["log_every_n_steps"],
        gradient_clip_val=conf["trainer"]["gradient_clip_val"]
    ).fit(
        model,
        DataLoader(dataset, batch_size=conf["dataloader"]["batch_size"],
                   shuffle=conf["dataloader"]["shuffle"],
                   num_workers=conf["dataloader"]["num_workers"])
    )

# Run Shenzhen
run_phase(shen_tr,
          conf["training"]["epochs"]["shen"],
          conf["training"]["lr"]["shen"],
          conf["training"]["epochs"]["shen"]*2)
auc0 = compute_auc(model, shen_val, device)
print(f"Shenzhen AUC after Task 0: {auc0:.3f}")

model.add_to_memory(shen_tr); model.compute_fisher(shen_tr)

# Run Montgomery
mont_tr, mont_val = get_cxr_ds(
    conf["data"]["mont_csv_path"], conf["data"]["mont_img_dir"]
)
run_phase(
    BalancedDataset(mont_tr, model.mem_x, model.mem_y),
    conf["training"]["epochs"]["mont"],
    conf["training"]["lr"]["mont"],
    conf["training"]["epochs"]["mont"]*2
)
mont_auc = compute_auc(model, mont_val, device)
print(f"Montgomery AUC: {mont_auc:.3f}")

# Optional MIAS phases
if "mias_csv_path" in conf["data"]:
    df_mias = pd.read_csv(conf["data"]["mias_csv_path"])
    sev = df_mias["SEVERITY"].fillna("B").astype(str).str.upper()
    y = sev.map(lambda s: 1 if s.startswith("M") else 0)
    tr2, val2 = train_test_split(
        np.arange(len(df_mias)), test_size=0.2,
        stratify=y, random_state=conf["dataloader"]["seed"]
    )
    mias_tr = MIASCSV(df_mias.iloc[tr2], conf["data"]["mias_img_dir"], train=True)
    mias_val = MIASCSV(df_mias.iloc[val2], conf["data"]["mias_img_dir"], train=False)

    # Phase MIAS head
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    run_phase(
        BalancedDataset(mias_tr, model.mem_x, model.mem_y, oversample=True),
        conf["training"]["epochs"]["mias_phase1"],
        conf["training"]["lr"]["mias_head"],
        conf["training"]["epochs"]["mias_phase1"]*2
    )
    # Phase MIAS mid
    for blk in model.encoder.blocks[6:]:
        for p in blk.parameters(): p.requires_grad = True
    run_phase(
        BalancedDataset(mias_tr, model.mem_x, model.mem_y, oversample=True),
        conf["training"]["epochs"]["mias_phase2"],
        conf["training"]["lr"]["mias_mid"],
        conf["training"]["epochs"]["mias_phase2"]*2
    )
    mias_auc = compute_auc(model, mias_val, device)
    print(f"MIAS AUC: {mias_auc:.3f}")
    final_aucs = compute_auc(model, shen_val, device)
    print(f"Backward Transfer Δ (Shenzhen): {final_aucs - auc0:+.3f}")
EOF
