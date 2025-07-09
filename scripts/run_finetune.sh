# File: scripts/run_finetune.sh
#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/finetune_shenzhen.yaml}
echo "📝 Finetuning Shenzhen with config $CONFIG"

python - <<EOF
import yaml
import pytorch_lightning as pl
from src.datamodules import ShenzhenDM
from src.models.finetune import LitTBFinetune

conf = yaml.safe_load(open("$CONFIG"))
dm = ShenzhenDM(
    csv_path=conf["data"]["csv_path"],
    img_dir=conf["data"]["img_dir"],
    batch=conf["dataloader"]["batch_size"],
    workers=conf["dataloader"]["num_workers"],
    seed=conf["dataloader"]["seed"]
)
model = LitTBFinetune(
    ckpt_path=conf["model"]["ckpt_path"],
    freeze_epochs=conf["model"]["freeze_epochs"],
    lr=conf["model"]["lr"]
)
trainer = pl.Trainer(
    accelerator=conf["trainer"]["accelerator"],
    devices=conf["trainer"]["devices"],
    precision=conf["trainer"]["precision"],
    max_epochs=conf["trainer"]["max_epochs"],
    log_every_n_steps=conf["trainer"]["log_every_n_steps"],
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath=conf["trainer"]["callbacks"][0]["dirpath"],
            filename=conf["trainer"]["callbacks"][0]["filename"],
            monitor=conf["trainer"]["callbacks"][0]["monitor"],
            mode=conf["trainer"]["callbacks"][0]["mode"],
            save_top_k=conf["trainer"]["callbacks"][0]["save_top_k"]
        ),
        pl.callbacks.EarlyStopping(
            monitor=conf["trainer"]["callbacks"][1]["monitor"],
            mode=conf["trainer"]["callbacks"][1]["mode"],
            patience=conf["trainer"]["callbacks"][1]["patience"]
        ),
        pl.callbacks.RichProgressBar()
    ]
)
trainer.fit(model, dm)
EOF
