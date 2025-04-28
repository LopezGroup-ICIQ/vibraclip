"""
Copyright (c) 2025 - Institute of Chemical Research of Catalonia (ICIQ)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle
import lmdb
from tqdm import tqdm

import numpy as np
import scipy.interpolate as interp
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem import Descriptors

import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_networkx,
    one_hot,
    scatter,
)
import matplotlib.pyplot as plt


types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


class QM9Spectra:
    """
    General method to build QM9 molecular graph representations with
    spectra data (e.g., IR, Raman and UV) as target.

    Args:
        data_path     : Path directing to the pickle file with raw data
        db_path       : Path to store the generated lmdb file
        spectra_dim   : To interpolate the IR spectra to a given dimension

    Returns:
        LMDB dataset with graph representations
    """

    def __init__(
        self,
        data_path="",
        db_path="",
        spectra_dim=3501,
    ):
        self.data_path = data_path
        self.db_path = db_path
        self.spectra_dim = spectra_dim

        # Raw Data
        self.raw_data = self._load_pickle()
        self.length = len(list(self.raw_data.keys()))

        # Molecular Mass
        self.mol_mass_dict = self._get_mol_mass_dict()
        self.mol_mass_dict_norm, self.mol_mass_state_dict = self._norm_mass()

        # Transform to Graphs
        self.graph_data, self.filtered_ids = self._transform()
        self.length = len(self.graph_data)
        self.length_filtered = len(self.filtered_ids)
        print(f"Filtered Graphs: {self.length_filtered}")

    def _load_pickle(self):
        """Method to load pickle file"""
        with open(self.data_path, "rb") as p_file:
            data = pickle.load(p_file)
        p_file.close()
        return data

    def _interpolate(self, arr):
        """Method to interpolate the targets into a given dimension"""
        # Initialize the interpolation method
        y_interp = interp.interp1d(np.arange(arr.size), arr)
        # Interpolate to a new dimension
        y_new = y_interp(np.linspace(0, arr.size - 1, self.spectra_dim))
        return y_new

    def _normalize(self, arr):
        """Method to normalize spectral data using MinMax method"""
        # Initialize scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Transform
        arr_scaled = scaler.fit_transform(arr.reshape(-1, 1))
        return arr_scaled.reshape(-1)

    def _get_mol_mass_dict(self):
        """Method to get a dict with standardize molecular mass"""
        mol_mass_dict = {}
        # Progress Bar
        pbar = tqdm(
            total=self.length,
            desc="Extracting Molecular Mass",
        )
        # Loop over raw data
        pbar_idx = 0
        for idx, data in self.raw_data.items():
            # Get rdkit mol object
            mol = Chem.MolFromSmiles(data["smile"])
            # Saturate SMILE with Hs
            mol_hs = Chem.AddHs(mol)
            # Get mass
            mol_mass = Descriptors.MolWt(mol_hs)
            # Append to dict
            mol_mass_dict.update({idx: mol_mass})
            # Progress bar
            pbar_idx += 1
            pbar.update(1)
        return mol_mass_dict

    def _norm_mass(self):
        """Method to normalize with std the molecular mass"""
        # Get masses
        mass_arr = np.array(list(self.mol_mass_dict.values())).reshape(-1, 1)
        # scaler
        scaler = StandardScaler()
        # Fit and transform
        std_mass = scaler.fit_transform(mass_arr).flatten()
        # rebuild dict
        std_mol_mass_dict = {
            key: std_mass[i] for i, key in enumerate(self.mol_mass_dict.keys())
        }
        # State dict
        state_dict = {
            "mean": scaler.mean_.item(),
            "scale": scaler.scale_.item(),
            "n_features": scaler.n_features_in_,
        }
        return (
            std_mol_mass_dict,
            state_dict,
        )

    def _get_graph(self, index, smiles):
        """Method that returns PyG Data object as graph representation"""
        # Convert SMILE representation to Molecule
        mol = Chem.MolFromSmiles(smiles)
        # Saturate SMILE with Hs
        mol_hs = Chem.AddHs(mol)
        # Embed Molecule
        has_conf = AllChem.EmbedMolecule(mol_hs, AllChem.ETKDG())

        # Get Conformers and XYZ Positions
        if has_conf == 0:
            conf = mol_hs.GetConformers()
            pos = mol_hs.GetConformer().GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            # Get Number of atoms
            n_atoms = mol_hs.GetNumAtoms()

            # Loop over atoms
            type_idx = []
            atomic_number = []
            aromatic = []
            sp, sp2, sp3 = [], [], []
            num_hs = []
            for atom in mol_hs.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            # Loop over Bonds
            row, col, edge_type = [], [], []
            for bond in mol_hs.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * n_atoms + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            # Include distance in edge_attr
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
            edge_attr_dist = torch.cat(
                [edge_attr, dist.type_as(edge_attr)], dim=-1
            )

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(
                hs[row], col, dim_size=n_atoms, reduce="sum"
            ).tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2, sp3, num_hs],
                    dtype=torch.float,
                )
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            # Build the graph object
            graph = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr_dist,
                idx=index,
                smiles=smiles,
            )

            return graph

        else:
            return None

    def _plot_graph(self, graph):
        """Helper method to plot a graph representation"""
        g = to_networkx(graph)
        nx.draw(g, pos=nx.kamada_kawai_layout(g), with_labels=True)
        plt.show()
        return

    def _transform(self):
        """Method that loops over raw data to transform into graph representations"""
        graph_data_list = []
        filtered_ids = []

        # Progress Bar
        pbar = tqdm(
            total=self.length,
            desc="Transforming Structures to Graphs",
        )
        # Loop over raw data to transform structures into graphs
        pbar_idx = 0
        for idx, data in self.raw_data.items():
            # Build the graph
            graph = self._get_graph(index=idx, smiles=data["smile"])
            # Check if there are nans inside the spectra
            has_nans = np.isnan(self.raw_data[idx]["ir_spectra"][1, :]).any()
            # Add vibrational spectra to the graph
            if graph and not has_nans:
                # Interpolate IR spectra
                ir_interp = self._interpolate(arr=data["ir_spectra"][1, :])
                ir_range_interp = self._interpolate(
                    arr=data["ir_spectra"][0, :]
                )
                # Interpolate Raman spectra
                raman_interp = self._interpolate(
                    arr=data["raman_spectra"][1, :]
                )
                raman_range_interp = self._interpolate(
                    arr=data["raman_spectra"][0, :]
                )
                # Normalize the IR and Raman spectra between 0 and 1.
                ir_interp_norm = self._normalize(arr=ir_interp)
                raman_interp_norm = self._normalize(arr=raman_interp)
                # Convert norm spectra to tensors
                ir_interp_norm = torch.Tensor(ir_interp_norm)
                raman_interp_norm = torch.Tensor(raman_interp_norm)
                # Append spectra to PyG Data object
                graph.ir_spectra = ir_interp_norm
                graph.ir_range = torch.Tensor(ir_range_interp)
                graph.raman_spectra = raman_interp_norm
                graph.raman_range = torch.Tensor(raman_range_interp)
                # Add molecular mass
                graph.mol_mass = torch.Tensor([self.mol_mass_dict_norm[idx]])
                # Append to lists
                graph_data_list.append(graph)
            else:
                filtered_ids.append(idx)
            # Progress Bar
            pbar_idx += 1
            pbar.update(1)
        return graph_data_list, filtered_ids

    def get_lmdb(self):
        """Method to create the LMDB file with graph data objects"""
        # Initialize DB
        db = lmdb.open(
            self.db_path,
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )

        # Progress Bar
        pbar = tqdm(
            total=self.length,
            desc="Transferring Graph data into LMDB",
        )

        # Loop over graph data
        pbar_idx = 0
        for idx, graph in enumerate(self.graph_data):
            # Write in the lmdb file
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(graph, protocol=-1))
            txn.commit()

            # Update pbar
            pbar_idx += 1
            pbar.update(1)

        # Save metadata inside LMDB
        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
        txn.put(
            "filtered".encode("ascii"),
            pickle.dumps(self.filtered_ids, protocol=-1),
        )
        txn.put(
            "mol_mass_scaler".encode("ascii"),
            pickle.dumps(self.mol_mass_state_dict, protocol=-1),
        )
        txn.commit()

        # Close DB
        db.sync()
        db.close()
        return

    def get_pickle(self):
        """Method to create pickle file with same structure as the LMDB file"""
        with open(f"{self.db_path}.pkl", "wb") as p_file:
            pickle.dump(self.graph_data, p_file)
        p_file.close()
        return
