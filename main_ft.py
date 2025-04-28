"""
Copyright (c) 2024 - Institute of Chemical Research of Catalonia (ICIQ)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

# Pytorch Lightning
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.data.lightning import LightningDataset

# Datasets
from vibraclip.datasets.qm9_dataset import LmdbQM9Dataset, get_dataset_splits

# Model and Callbacks
from vibraclip.models.vibraclip_ir_raman import VibraCLIP
from vibraclip.callbacks.callbacks import RetrievalAccIRRaman

# Config
import hydra
from omegaconf import DictConfig

# Warnings
import warnings

warnings.filterwarnings("ignore")

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("medium")

# Device
device = (
    torch.device("cuda:0")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# Trainer
@hydra.main(config_path="./configs", config_name="config")
def train_model(cfg: DictConfig) -> Trainer:
    # Root dir
    root_dir = cfg.paths.root_dir
    # Dataset
    dataset = LmdbQM9Dataset(
        root=f"{cfg.paths.root_dir}/data",
        db_path=f"{root_dir}{cfg.paths.db_path}",
        transform=cfg.dataset.transform,
    )
    # Split dataset into train/val/test
    train_data, val_data, test_data = get_dataset_splits(
        dataset,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
    )
    # Datamodule
    graph_module = LightningDataset(
        train_data,
        val_data,
        test_data,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )
    # Logger
    logger = WandbLogger(
        name=cfg.experiment.id,
        save_dir=f"{root_dir}",
        project=cfg.logging.wandb_project,
        offline=cfg.logging.offline,
    )
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(
            f"{root_dir}{cfg.paths.checkpoint_path}", cfg.experiment.id
        ),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        inference_mode=False,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=cfg.training.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(
                    f"{root_dir}{cfg.paths.checkpoint_path}", cfg.experiment.id
                ),
                save_weights_only=True,
                mode="min",
                monitor="val_loss",
                save_top_k=1,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            RetrievalAccIRRaman(filename=f"{cfg.experiment.id}"),
        ],
        logger=logger,
    )
    # Setting the seed
    pl.seed_everything(42)

    # Loading Model from checkpoint
    model = VibraCLIP.load_from_checkpoint(
        checkpoint_path=f"{cfg.paths.root_dir}{cfg.paths.pre_trained_path}"
    )
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the last layers of the projection heads
    for project_head in [
        model.graph_projection,
        model.ir_projection,
        model.raman_projection,
    ]:
        # Access to the last layer
        last_layer = project_head.fc
        # Unfreeze
        for param in last_layer.parameters():
            param.requires_grad = True
    # Fine-Tune!
    trainer.fit(model, datamodule=graph_module)
    model = VibraCLIP.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
    )
    # Test
    trainer.test(model, datamodule=graph_module)
    logger.experiment.finish()
    return trainer


# Run!
if __name__ == "__main__":
    trainer = train_model()
