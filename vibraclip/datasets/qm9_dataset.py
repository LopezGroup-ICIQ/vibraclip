"""
Copyright (c) 2025 - Institute of Chemical Research of Catalonia (ICIQ)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle
import lmdb

import numpy as np

import torch
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops


class LmdbQM9Dataset(InMemoryDataset):
    """
    PyG Dataset class to load LMDB file containing graph representation
    and spectra data.

    Args:
        root     : Path where the processed folder is going to be created
        db_path  : Path where the lmdb file is stored
        transform: If transforming the graph representation is required

    Returns:
        PyG Dataset object to be converted into DataLoaders.
    """

    def __init__(self, root, db_path, transform=False):
        self.root = root
        self.db_path = db_path
        self.transform = transform
        self.db = self._connect_db()
        self.length = self._get_additional_info(info="length")
        self.mol_mass_scaler = self._get_additional_info(
            info="mol_mass_scaler"
        )
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _connect_db(self):
        db = lmdb.open(
            str(self.db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return db

    def _get_additional_info(self, info):
        """To retrieve specific information from the lmdb file"""
        return pickle.loads(self.db.begin().get(str(info).encode("ascii")))

    @property
    def raw_file_names(self):
        return self.db_path

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        """Method to build a list of data objects, collate and save"""
        # Get data objects from lmdb
        data_list = []
        for idx in range(self.length):
            with self.db.begin() as txn:
                data_compress = txn.get(f"{idx}".encode("ascii"))
                # De-serialize PyG Data object
                graph = pickle.loads(data_compress)
                # Transform
                if self.transform:
                    row = torch.arange(graph.num_nodes, dtype=torch.long)
                    col = torch.arange(graph.num_nodes, dtype=torch.long)

                    row = row.view(-1, 1).repeat(1, graph.num_nodes).view(-1)
                    col = col.repeat(graph.num_nodes)
                    edge_index = torch.stack([row, col], dim=0)

                    if graph.edge_attr is not None:
                        index = (
                            graph.edge_index[0] * graph.num_nodes
                            + graph.edge_index[1]
                        )
                        size = list(graph.edge_attr.size())
                        size[0] = graph.num_nodes * graph.num_nodes
                        edge_attr = graph.edge_attr.new_zeros(size)
                        edge_attr[index] = graph.edge_attr

                    edge_index, edge_attr = remove_self_loops(
                        edge_index, edge_attr
                    )
                    graph.edge_index = edge_index
                    graph.edge_attr = edge_attr
                # Append graph to list
                data_list.append(graph)

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_dataset_splits(
    dataset, val_ratio=0.1, test_ratio=0.1, random_state=42
):
    """Method function to split PyG dataset into train/val/testing"""
    dataset_length = len(dataset)
    val_split = int(np.ceil(val_ratio * dataset_length))
    test_split = int(np.ceil(test_ratio * dataset_length))
    train_split = int(np.floor(dataset_length - (val_split + test_split)))
    train_data, val_data, test_data = random_split(
        dataset,
        [train_split, val_split, test_split],
        generator=torch.Generator().manual_seed(random_state),
    )
    return train_data, val_data, test_data
