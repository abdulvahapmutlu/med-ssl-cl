# File: scripts/run_pretrain.sh
#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/pretrain.yaml}
echo "📝 SSL Pretraining with config $CONFIG"

python - <<EOF
import yaml
import pytorch_lightning as pl
from src.datamodules import ChestSSLDataModule
from src.models.dino import LitDINO

conf = yaml.safe_load(open("$CONFIG"))
dm = ChestSSLDataModule(
    npz_path=conf["data"]["npz_path"],
    batch=conf["dataloader"]["batch_size"],
    workers=conf["dataloader"]["num_workers"]
)
model = LitDINO(
    lr=conf["model"]["lr"],
    out_dim=conf["model"]["out_dim"],
    warmup_epochs=conf["model"]["warmup_epochs"],
    max_epochs=conf["model"]["max_epochs"]
)
trainer = pl.Trainer(
    accelerator=conf["trainer"]["accelerator"],
    devices=conf["trainer"]["devices"],
    precision=conf["trainer"]["precision"],
    max_epochs=conf["trainer"]["max_epochs"],
    accumulate_grad_batches=conf["trainer"]["accumulate_grad_batches"],
    log_every_n_steps=conf["trainer"]["log_every_n_steps"],
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath=conf["trainer"]["callbacks"][0]["dirpath"],
            save_top_k=conf["trainer"]["callbacks"][0]["save_top_k"],
            every_n_epochs=conf["trainer"]["callbacks"][0]["every_n_epochs"]
        ),
        pl.callbacks.LearningRateMonitor(
            logging_interval=conf["trainer"]["callbacks"][1]["logging_interval"]
        ),
        pl.callbacks.RichProgressBar(
            theme=pl.callbacks.progress.rich_progress.RichProgressBarTheme(
                description=conf["trainer"]["callbacks"][2]["theme"]["description"],
                progress_bar=conf["trainer"]["callbacks"][2]["theme"]["progress_bar"]
            )
        )
    ]
)
trainer.fit(model, dm)
EOF
