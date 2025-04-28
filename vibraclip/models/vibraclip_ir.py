"""
Copyright (c) 2025 - Institute of Chemical Research of Catalonia (ICIQ)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import itertools
import numpy as np
import lightning.pytorch as pl

import torch_geometric.nn as tgnn
from torch_geometric.nn import DimeNetPlusPlus


# Activation functions dict
act_fun_dict = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(),
    "softplus": nn.Softplus(),
    "tanh": nn.Tanh(),
}


# Graph Neural Network
class GNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        out_channels=1,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        max_num_neighbors=32,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    ):
        super(GNNEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.int_emb_size = int_emb_size
        self.basis_emb_size = basis_emb_size
        self.out_emb_channels = out_emb_channels
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.envelope_exponent = envelope_exponent
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.num_output_layers = num_output_layers

        # DimeNet++
        self.graph_conv = DimeNetPlusPlus(
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_blocks=self.num_blocks,
            int_emb_size=self.int_emb_size,
            basis_emb_size=self.basis_emb_size,
            out_emb_channels=self.out_emb_channels,
            num_spherical=self.num_spherical,
            num_radial=self.num_radial,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            envelope_exponent=self.envelope_exponent,
            num_before_skip=self.num_before_skip,
            num_after_skip=self.num_after_skip,
            num_output_layers=self.num_output_layers,
        )

    def forward(self, data):
        out = self.graph_conv(data.z, data.pos, data.batch)
        return out


# Vibrational Spectra Encoder
class SpectraEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        out_features,
        act_fun,
        batch_norm,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_features = out_features
        self.act_fun = act_fun_dict[act_fun]
        self.batch_norm = batch_norm

        self.hidden = nn.Sequential()
        inter_dim = np.linspace(
            self.hidden_dim, self.out_features, self.n_layers + 1, dtype=int
        )[0:-1]
        inter_dim = np.concatenate(([self.input_dim], inter_dim))
        for i, (in_size, out_size) in enumerate(
            zip(inter_dim[:-1], inter_dim[1:])
        ):
            self.hidden.add_module(
                name=f"Linear_{str(i)}", module=nn.Linear(in_size, out_size)
            )
            if self.batch_norm:
                self.hidden.add_module(
                    name=f"BatchNorm_{str(i)}", module=nn.BatchNorm1d(out_size)
                )
            self.hidden.add_module(name=f"Act_{str(i)}", module=self.act_fun)
        self.lin_out = nn.Linear(inter_dim[-1], self.out_features)

    def forward(self, x):
        x = self.hidden(x)
        out = self.lin_out(x)
        return out


# Projection Head
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
        p_dropout,
        layer_norm,
        bias,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.p_dropout = p_dropout
        self.layer_norm = layer_norm
        self.bias = bias

        self.projection = nn.Linear(
            self.embedding_dim, self.projection_dim, bias=self.bias
        )
        self.gelu = nn.GELU()
        self.fc = nn.Linear(
            self.projection_dim, projection_dim, bias=self.bias
        )

        if self.dropout:
            self.dropout_layer = nn.Dropout(self.p_dropout)

        if self.layer_norm:
            self.layer_norm_out = nn.LayerNorm(self.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        # Dropout
        if self.dropout:
            x = self.dropout_layer(x)
        # Residual Connection
        x += projected
        # Layer Norm
        if self.layer_norm:
            x = self.layer_norm_out(x)
        return x


# VibraCLIP (Graph + IR) Lightning
class VibraCLIP(pl.LightningModule):
    def __init__(
        self,
        g_encoder_hidden_channels,
        g_encoder_out_channels,
        g_encoder_num_blocks,
        g_encoder_int_emb_size,
        g_encoder_basis_emb_size,
        g_encoder_out_emb_channels,
        g_encoder_num_spherical,
        g_encoder_num_radial,
        g_encoder_cutoff,
        g_encoder_max_num_neighbors,
        g_encoder_envelope_exponent,
        g_encoder_num_before_skip,
        g_encoder_num_after_skip,
        g_encoder_num_output_layers,
        spectra_encoder_input_dim,
        spectra_encoder_hidden_dim,
        spectra_encoder_n_layers,
        spectra_encoder_out_features,
        spectra_encoder_act_fun,
        spectra_encoder_batch_norm,
        projection_latent_dim,
        projection_dropout,
        projection_p_dropout,
        projection_layer_norm,
        projection_bias,
        molecular_mass,
        temperature,
        weight_decay,
        head_lr,
        gnn_encoder_lr,
        spectra_lr,
        lr_scheduler_patience,
        lr_scheduler_factor,
    ):
        super().__init__()

        # Molecular Mass as Anchoring Feature
        self.molecular_mass = molecular_mass
        # Add additional feature to allocate molecular mass
        if self.molecular_mass:
            g_encoder_out_channels_mass = g_encoder_out_channels + 1
            spectra_encoder_out_features_mass = (
                spectra_encoder_out_features + 1
            )
        else:
            g_encoder_out_channels_mass = g_encoder_out_channels
            spectra_encoder_out_features_mass = spectra_encoder_out_features

        # Graph Neural Network Encoder
        self.gnn_encoder = GNNEncoder(
            hidden_channels=g_encoder_hidden_channels,
            out_channels=g_encoder_out_channels,
            num_blocks=g_encoder_num_blocks,
            int_emb_size=g_encoder_int_emb_size,
            basis_emb_size=g_encoder_basis_emb_size,
            out_emb_channels=g_encoder_out_emb_channels,
            num_spherical=g_encoder_num_spherical,
            num_radial=g_encoder_num_radial,
            cutoff=g_encoder_cutoff,
            max_num_neighbors=g_encoder_max_num_neighbors,
            envelope_exponent=g_encoder_envelope_exponent,
            num_before_skip=g_encoder_num_before_skip,
            num_after_skip=g_encoder_num_after_skip,
            num_output_layers=g_encoder_num_output_layers,
        )

        # Spectra Encoders
        self.ir_encoder = SpectraEncoder(
            input_dim=spectra_encoder_input_dim,
            hidden_dim=spectra_encoder_hidden_dim,
            n_layers=spectra_encoder_n_layers,
            out_features=spectra_encoder_out_features,
            act_fun=spectra_encoder_act_fun,
            batch_norm=spectra_encoder_batch_norm,
        )

        # Projection Heads
        self.graph_projection = ProjectionHead(
            embedding_dim=g_encoder_out_channels_mass,
            projection_dim=projection_latent_dim,
            dropout=projection_dropout,
            p_dropout=projection_p_dropout,
            layer_norm=projection_layer_norm,
            bias=projection_bias,
        )

        self.ir_projection = ProjectionHead(
            embedding_dim=spectra_encoder_out_features_mass,
            projection_dim=projection_latent_dim,
            dropout=projection_dropout,
            p_dropout=projection_p_dropout,
            layer_norm=projection_layer_norm,
            bias=projection_bias,
        )

        # Other
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.gnn_encoder_lr = gnn_encoder_lr
        self.spectra_lr = spectra_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        # To save hyperparameters inside the model
        self.save_hyperparameters()

    def forward(self, data):
        # GNN Encoder features
        gnn_features = self.gnn_encoder(data)
        # Reshape vibrational spectra
        ir_spectra_input = data.ir_spectra.view(
            data.batch_size, self.hparams.spectra_encoder_input_dim
        )
        # IR Spectra Encoder
        ir_features = self.ir_encoder(ir_spectra_input)
        # Adding the Anchoring feature
        if self.molecular_mass:
            mol_mass = data.mol_mass.view(data.batch_size, 1)
            gnn_features = torch.cat([gnn_features, mol_mass], dim=1)
            ir_features = torch.cat([ir_features, mol_mass], dim=1)
        # Projection Heads
        graph_embeddings = self.graph_projection(gnn_features)
        ir_spectra_embeddings = self.ir_projection(ir_features)
        return graph_embeddings, ir_spectra_embeddings

    def configure_optimizers(self):
        # Model parameters
        parameters = [
            {
                "params": self.gnn_encoder.parameters(),
                "lr": self.gnn_encoder_lr,
            },
            {"params": self.ir_encoder.parameters(), "lr": self.spectra_lr},
            {
                "params": itertools.chain(
                    self.graph_projection.parameters(),
                    self.ir_projection.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        # Optimizer algorithm
        optimizer = optim.AdamW(parameters, weight_decay=self.weight_decay)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def _loss_function(self, graph_embeddings, spectra_embeddings):
        """Loss function that aligns graph embeddings and IR embeddings"""
        # Compute logits
        logits = (
            spectra_embeddings @ graph_embeddings.T
        ) / self.hparams.temperature
        # Similarities
        graph_similarities = graph_embeddings @ graph_embeddings.T
        spectra_similarities = spectra_embeddings @ spectra_embeddings.T
        # Targets
        targets = F.softmax(
            (graph_similarities + spectra_similarities)
            / 2.0
            * self.hparams.temperature,
            dim=-1,
        )
        # Losses
        graph_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        spectra_loss = (-targets * self.log_softmax(logits)).sum(1)
        avg_loss = (graph_loss + spectra_loss) / 2.0
        return avg_loss, graph_loss, spectra_loss

    def training_step(self, data, batch_idx):
        # Forward pass
        graph_embeddings, ir_spectra_embeddings = self.forward(data)
        # Loss
        loss, graph_loss, spectra_loss = self._loss_function(
            graph_embeddings, ir_spectra_embeddings
        )
        train_loss = self.all_gather(loss.mean())
        # Logging
        self.log(
            "train_loss",
            train_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_g_loss",
            graph_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_s_loss",
            spectra_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, data, batch_idx):
        # Forward pass
        graph_embeddings, ir_spectra_embeddings = self.forward(data)
        # Loss
        loss, graph_loss, spectra_loss = self._loss_function(
            graph_embeddings, ir_spectra_embeddings
        )
        val_loss = self.all_gather(loss.mean())
        # Logging
        self.log(
            "val_loss",
            val_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_g_loss",
            graph_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_s_loss",
            spectra_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return val_loss

    def test_step(self, data, batch_idx):
        # Forward pass
        graph_embeddings, ir_spectra_embeddings = self.forward(data)
        # Loss
        loss, graph_loss, spectra_loss = self._loss_function(
            graph_embeddings, ir_spectra_embeddings
        )
        test_loss = self.all_gather(loss.mean())
        # Logging
        self.log(
            "test_loss",
            test_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_g_loss",
            graph_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_s_loss",
            spectra_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return test_loss
