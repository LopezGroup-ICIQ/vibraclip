"""
Copyright (c) 2025 - Institute of Chemical Research of Catalonia (ICIQ)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle

import numpy as np

import torch
import lightning.pytorch as pl

from tqdm import tqdm


class RetrievalAccIR(pl.Callback):
    """
    Callback that runs at the end of the testing to easily
    evaluate the model's performance by exporting a pickle
    file with the following structure:

    {"id": {"SMILE":, "Embeddings":, ...}}

    Args:
        filename: Using the WANDB run id as identifier.

    Returns:
        Exports pickle file with test dataset and performance.
    """

    def __init__(self, filename):
        self.filename = filename

    def _get_embeddings(self, dataset, pl_module):
        """Method to generate all embeddings"""
        # Inference on test loader
        graph_embeddings_dict = {}
        spectra_embeddings_dict = {}
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                # Move batch to GPU
                batch = batch.cuda()
                # Get SMILES
                smiles = batch.smiles
                # Embedding [Batch, N]
                graph_embeddings, spectra_embeddings = pl_module.forward(batch)
                for smile, g_emb, s_emb in zip(
                    smiles, graph_embeddings, spectra_embeddings
                ):
                    # Move to CPU and as array
                    smile = smile
                    g_emb = g_emb.cpu().numpy()
                    s_emb = s_emb.cpu().numpy()
                    # Append to dicts
                    graph_embeddings_dict.update({str(smile): g_emb})
                    spectra_embeddings_dict.update({str(smile): s_emb})
        return graph_embeddings_dict, spectra_embeddings_dict

    def _get_similarities(self, graph_embs_dict, spectra_embs_dict):
        similarities_dict = {}

        # Convert embeddings to matrices
        graph_matrix = np.array(list(graph_embs_dict.values()))
        spectra_matrix = np.array(list(spectra_embs_dict.values()))

        # Pre-compute norms
        graph_norms = np.linalg.norm(graph_matrix, axis=1)
        spectra_norms = np.linalg.norm(spectra_matrix, axis=1)

        # Progress Bar
        pbar = tqdm(
            total=len(spectra_embs_dict.keys()),
            desc="Similarity Scores",
        )

        # Compute cosine similarity using matrix operations
        pbar_idx = 0
        for idx, (smile_target, s_emb_vec) in enumerate(
            spectra_embs_dict.items()
        ):
            # Calc similarities
            similarities = np.dot(graph_matrix, s_emb_vec) / (
                graph_norms * spectra_norms[idx]
            )

            # Create a dict with similarities
            candidates_dict = {
                str(key): float(sim)
                for key, sim in zip(graph_embs_dict.keys(), similarities)
            }

            # Sorting
            candidates_dict_sort = dict(
                sorted(
                    candidates_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            # Append candidates dict
            similarities_dict.update({str(smile_target): candidates_dict_sort})

            # Update pbar
            pbar_idx += 1
            pbar.update(1)

        return similarities_dict

    def on_test_end(self, trainer, pl_module):
        """Method that evaluates the TopK matches given a molecule and a IR"""
        # Switch model to eval
        model = pl_module.eval()

        # Generate embeddings lists
        graph_embeddings_dict, spectra_embeddings_dict = self._get_embeddings(
            dataset=trainer.test_dataloaders,
            pl_module=model,
        )
        # Get Similarities
        similarities_dict = self._get_similarities(
            graph_embs_dict=graph_embeddings_dict,
            spectra_embs_dict=spectra_embeddings_dict,
        )

        # Export pickle file with similarities dict
        print("\nExporting Pickle file...")
        with open(f"{self.filename}.pkl", "wb") as p_file:
            pickle.dump(similarities_dict, p_file)
        p_file.close()
        return


class RetrievalAccIRRaman(pl.Callback):
    """
    Callback that runs at the end of the testing to easily
    evaluate the model's performance by exporting pickle.

    To be used with the model that incorporates both
    the IR and Raman spectra.

    Args:
        filename: Using the WANDB run id as identifier

    Returns:
        Export pickle file with test dataset and performance.
    """

    def __init__(self, filename):
        self.filename = filename

    def _get_embeddings(self, dataset, pl_module):
        """Method to generate all embeddings"""
        graph_embeddings_dict = {}
        ir_embeddings_dict = {}
        raman_embeddings_dict = {}
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                # Move batch to GPU
                batch = batch.cuda()
                # Get Smiles
                smiles = batch.smiles
                # Embedding [Batch, N]
                (
                    graph_embeddings,
                    ir_embeddings,
                    raman_embeddings,
                ) = pl_module.forward(batch)
                # Loop over embeddings
                for smile, g_emb, ir_emb, raman_emb in zip(
                    smiles, graph_embeddings, ir_embeddings, raman_embeddings
                ):
                    # Move to CPU and as array
                    smile = smile
                    g_emb = g_emb.cpu().numpy()
                    ir_emb = ir_emb.cpu().numpy()
                    raman_emb = raman_emb.cpu().numpy()
                    # Append to dicts
                    graph_embeddings_dict.update({str(smile): g_emb})
                    ir_embeddings_dict.update({str(smile): ir_emb})
                    raman_embeddings_dict.update({str(smile): raman_emb})
        return graph_embeddings_dict, ir_embeddings_dict, raman_embeddings_dict

    def _get_similarities(
        self, graph_embs_dict, ir_embs_dict, raman_embs_dict
    ):
        similarities_dict = {}

        # Convert embeddings to matrices
        graph_matrix = np.array(list(graph_embs_dict.values()))
        ir_matrix = np.array(list(ir_embs_dict.values()))
        raman_matrix = np.array(list(raman_embs_dict.values()))

        # Pre-compute norms for each modality
        graph_norms = np.linalg.norm(graph_matrix, axis=1)
        ir_norms = np.linalg.norm(ir_matrix, axis=1)
        raman_norms = np.linalg.norm(raman_matrix, axis=1)

        # Progress Bar
        pbar = tqdm(
            total=len(graph_embs_dict.keys()),
            desc="Similarity Scores",
        )

        # Compute 3 modalities cosine similarity
        pbar_idx = 0
        for idx, (smile_target, ir_emb_vec) in enumerate(ir_embs_dict.items()):
            # Retrieve the corresponding Graph and Raman embeddings for the same key
            graph_emb_vec = graph_embs_dict[smile_target]
            raman_emb_vec = raman_embs_dict[smile_target]

            # Calculate vectorized pairwise cosine similarity
            sim_graph_ir = np.dot(graph_matrix, ir_emb_vec) / (
                graph_norms * ir_norms[idx]
            )
            sim_graph_raman = np.dot(graph_matrix, raman_emb_vec) / (
                graph_norms * raman_norms[idx]
            )
            sim_ir_raman = np.dot(ir_matrix, raman_emb_vec) / (
                ir_norms * raman_norms[idx]
            )

            # Geometric mean of similarities
            similarities = np.cbrt(
                sim_graph_ir * sim_graph_raman * sim_ir_raman
            )

            # Create a dict with geometric mean
            candidates_dict = {
                str(key): float(sim)
                for key, sim in zip(graph_embs_dict.keys(), similarities)
            }

            # Sort the dictionary
            candidates_dict_sort = dict(
                sorted(
                    candidates_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            # Append sorted candidates to the result dict
            similarities_dict.update({str(smile_target): candidates_dict_sort})

            # update pbar
            pbar_idx += 1
            pbar.update(1)
        return similarities_dict

    def on_test_end(self, trainer, pl_module):
        """Method that evaluates TopK matches given a molecular, and IR and Raman"""
        # Switching model to eval
        model = pl_module.eval()

        # Generate embeddings dicts
        (
            graph_embeddings_dict,
            ir_embeddings_dict,
            raman_embeddings_dict,
        ) = self._get_embeddings(
            dataset=trainer.test_dataloaders,
            pl_module=model,
        )

        # Get Similarities
        similarity_dict = self._get_similarities(
            graph_embs_dict=graph_embeddings_dict,
            ir_embs_dict=ir_embeddings_dict,
            raman_embs_dict=raman_embeddings_dict,
        )

        # Export pickle file with similarities dict
        print("\nExporting Pickle file...")
        with open(f"{self.filename}.pkl", "wb") as p_file:
            pickle.dump(similarity_dict, p_file)
        p_file.close()
        return
