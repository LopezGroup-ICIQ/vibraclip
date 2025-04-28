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
    EarlyStopping,
)
from torch_geometric.data.lightning import LightningDataset

# Datasets
from vibraclip.datasets.qm9_dataset import LmdbQM9Dataset, get_dataset_splits

# Model and Callbacks
from vibraclip.models.vibraclip_ir import VibraCLIP

# Optuna
import optuna

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

# GLOBAL Variables
ROOT_DIR = "<<YOUR_ROOT_DIR>>"
DB_PATH = "/qm9s_ir_raman.lmdb"
PRE_TRAINED_PATH = "/pre_trained/"
MAX_EPOCHS = 50


def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function to be minimized"""

    # Graph Neural Network Encoder
    g_encoder_out_channels = trial.suggest_int(
        "g_encoder_out_channels", 64, 1024
    )

    # Vibrational Spectra
    spectra_encoder_hidden_dim = trial.suggest_int(
        "spectra_encoder_hidden_dim", 64, 3000
    )
    spectra_encoder_n_layers = trial.suggest_int(
        "spectra_encoder_n_layers", 1, 12
    )
    spectra_encoder_out_features = trial.suggest_int(
        "spectra_encoder_out_features", 64, 3000
    )
    spectra_encoder_act_fun = trial.suggest_categorical(
        "spectra_encoder_act_fun",
        ["relu", "elu", "leakyrelu", "softplus", "tanh"],
    )
    spectra_encoder_batch_norm = trial.suggest_categorical(
        "spectra_encoder_batch_norm", [True, False]
    )

    # Projection Heads
    projection_latent_dim = trial.suggest_int(
        "projection_latent_dim", 100, 1024
    )
    projection_dropout = trial.suggest_categorical(
        "projection_dropout", [True, False]
    )
    projection_p_dropout = trial.suggest_float(
        "projection_p_dropout", 0.0, 1.0
    )
    projection_layer_norm = trial.suggest_categorical(
        "projection_layer_norm", [True, False]
    )
    projection_bias = trial.suggest_categorical(
        "projection_bias", [True, False]
    )

    # Training
    molecular_mass = trial.suggest_categorical("molecular_mass", [True, False])
    batch_size = trial.suggest_int("batch_size", 16, 256)
    temperature = trial.suggest_int("temperature", 100, 300)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
    head_lr = trial.suggest_loguniform("head_lr", 1e-5, 1e-2)
    gnn_encoder_lr = trial.suggest_loguniform("gnn_encoder_lr", 1e-5, 1e-2)
    spectra_lr = trial.suggest_loguniform("spectra_lr", 1e-5, 1e-2)

    # Dataset
    dataset = LmdbQM9Dataset(
        root=ROOT_DIR,
        db_path=ROOT_DIR + DB_PATH,
    )
    train_data, val_data, test_data = get_dataset_splits(
        dataset,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    graph_module = LightningDataset(
        train_data,
        val_data,
        test_data,
        batch_size=batch_size,
        num_workers=0,
    )

    # Model
    model = VibraCLIP(
        # Graph Encoder
        g_encoder_hidden_channels=128,
        g_encoder_out_channels=g_encoder_out_channels,
        g_encoder_num_blocks=4,
        g_encoder_int_emb_size=64,
        g_encoder_basis_emb_size=8,
        g_encoder_out_emb_channels=256,
        g_encoder_num_spherical=7,
        g_encoder_num_radial=6,
        g_encoder_cutoff=5.0,
        g_encoder_max_num_neighbors=32,
        g_encoder_envelope_exponent=5,
        g_encoder_num_before_skip=1,
        g_encoder_num_after_skip=2,
        g_encoder_num_output_layers=3,
        # Spectra Encoder
        spectra_encoder_input_dim=1750,
        spectra_encoder_hidden_dim=spectra_encoder_hidden_dim,
        spectra_encoder_n_layers=spectra_encoder_n_layers,
        spectra_encoder_out_features=spectra_encoder_out_features,
        spectra_encoder_act_fun=spectra_encoder_act_fun,
        spectra_encoder_batch_norm=spectra_encoder_batch_norm,
        # Projection Heads
        projection_latent_dim=projection_latent_dim,
        projection_dropout=projection_dropout,
        projection_p_dropout=projection_p_dropout,
        projection_layer_norm=projection_layer_norm,
        projection_bias=projection_bias,
        # Training
        molecular_mass=molecular_mass,
        temperature=temperature,
        weight_decay=weight_decay,
        head_lr=head_lr,
        gnn_encoder_lr=gnn_encoder_lr,
        spectra_lr=spectra_lr,
        lr_scheduler_patience=5,
        lr_scheduler_factor=0.1,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                mode="min",
            ),
        ],
    )

    # Hparams Dict
    hparams_dict = dict(
        batch_size=batch_size,
        g_encoder_out_channels=g_encoder_out_channels,
        spectra_encoder_hidden_dim=spectra_encoder_hidden_dim,
        spectra_encoder_n_layers=spectra_encoder_n_layers,
        spectra_encoder_out_features=spectra_encoder_out_features,
        spectra_encoder_act_fun=spectra_encoder_act_fun,
        spectra_encoder_batch_norm=spectra_encoder_batch_norm,
        projection_latent_dim=projection_latent_dim,
        projection_dropout=projection_dropout,
        projection_p_dropout=projection_p_dropout,
        projection_layer_norm=projection_layer_norm,
        projection_bias=projection_bias,
        molecular_mass=molecular_mass,
        temperature=temperature,
        weight_decay=weight_decay,
        head_lr=head_lr,
        gnn_encoder_lr=gnn_encoder_lr,
        spectra_lr=spectra_lr,
    )
    print(hparams_dict)

    # Fit
    trainer.logger.log_hyperparams(hparams_dict)
    trainer.fit(model, datamodule=graph_module)

    return (
        trainer.logged_metrics["val_g_loss"].item(),
        trainer.logged_metrics["val_s_loss"].item(),
    )


# Run!
if __name__ == "__main__":

    # Create optuna study
    study = optuna.create_study(
        study_name="vibraclip",
        storage="sqlite:///vibraclip_ir.db",
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )

    # Run Optimization
    study.optimize(objective, n_trials=200)

    # Get best parameters
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
    trial_with_low_gnn = min(study.best_trials, key=lambda t: t.values[0])
    trial_with_low_nn = min(study.best_trials, key=lambda t: t.values[1])

    # GNN
    print("Trial with min GNN val loss:")
    print(f"\tNumber: {trial_with_low_gnn.number}")
    print(f"\tParams: {trial_with_low_gnn.params}")
    print(f"\tValues: {trial_with_low_gnn.values}")

    # NN
    print("Trial with min NN val loss:")
    print(f"\tNumber: {trial_with_low_nn.number}")
    print(f"\tParams: {trial_with_low_nn.params}")
    print(f"\tValues: {trial_with_low_nn.values}")
