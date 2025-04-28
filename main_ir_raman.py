"""
Copyright (c) 2025 - Institute of Chemical Research of Catalonia (ICIQ)

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
    EarlyStopping,
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
            RichProgressBar(refresh_rate=1),
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                mode="min",
            ),
            RetrievalAccIRRaman(filename=f"{cfg.experiment.id}"),
        ],
        logger=logger,
    )
    # Training from scratch
    pl.seed_everything(42)
    # Model
    model = VibraCLIP(
        # Graph Encoder
        g_encoder_hidden_channels=cfg.model.g_encoder.hidden_channels,
        g_encoder_out_channels=cfg.model.g_encoder.out_channels,
        g_encoder_num_blocks=cfg.model.g_encoder.num_blocks,
        g_encoder_int_emb_size=cfg.model.g_encoder.int_emb_size,
        g_encoder_basis_emb_size=cfg.model.g_encoder.basis_emb_size,
        g_encoder_out_emb_channels=cfg.model.g_encoder.out_emb_channels,
        g_encoder_num_spherical=cfg.model.g_encoder.num_spherical,
        g_encoder_num_radial=cfg.model.g_encoder.num_radial,
        g_encoder_cutoff=cfg.model.g_encoder.cutoff,
        g_encoder_max_num_neighbors=cfg.model.g_encoder.max_num_neighbors,
        g_encoder_envelope_exponent=cfg.model.g_encoder.envelope_exponent,
        g_encoder_num_before_skip=cfg.model.g_encoder.num_before_skip,
        g_encoder_num_after_skip=cfg.model.g_encoder.num_after_skip,
        g_encoder_num_output_layers=cfg.model.g_encoder.num_output_layers,
        # Spectra Encoders
        spectra_encoder_input_dim=cfg.model.spectra_encoder.input_dim,
        spectra_encoder_hidden_dim=cfg.model.spectra_encoder.hidden_dim,
        spectra_encoder_n_layers=cfg.model.spectra_encoder.n_layers,
        spectra_encoder_out_features=cfg.model.spectra_encoder.out_features,
        spectra_encoder_act_fun=cfg.model.spectra_encoder.act_fun,
        spectra_encoder_batch_norm=cfg.model.spectra_encoder.batch_norm,
        # Projection Heads
        projection_latent_dim=cfg.model.projection.latent_dim,
        projection_dropout=cfg.model.projection.dropout,
        projection_p_dropout=cfg.model.projection.p_dropout,
        projection_layer_norm=cfg.model.projection.layer_norm,
        projection_bias=cfg.model.projection.bias,
        # Training
        molecular_mass=cfg.training.molecular_mass,
        loss_allpairs=cfg.training.loss_allpairs,
        temperature=cfg.training.temperature,
        weight_decay=cfg.training.weight_decay,
        head_lr=cfg.training.head_lr,
        gnn_encoder_lr=cfg.training.gnn_encoder_lr,
        spectra_lr=cfg.training.spectra_lr,
        lr_scheduler_patience=cfg.training.lr_scheduler_patience,
        lr_scheduler_factor=cfg.training.lr_scheduler_factor,
    )
    # Training
    trainer.fit(model, datamodule=graph_module)
    model = VibraCLIP.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    # Test
    trainer.test(model, datamodule=graph_module)
    logger.experiment.finish()
    return trainer


# Run!
if __name__ == "__main__":
    trainer = train_model()
