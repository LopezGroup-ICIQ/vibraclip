"""
Copyright (c) 2024 - Institute of Chemical Research of Catalonia (ICIQ)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from vibraclip.preprocessing.graph import QM9Spectra

# Paths
data_path = "../vibraclip/data/qm9s_ir_raman.pkl"
db_path = "../vibraclip/data/qm9s_ir_raman.lmdb"

# LMDB Generator
extractor = QM9Spectra(
    data_path=data_path,
    db_path=db_path,
    spectra_dim=1750,
)

# Run
extractor.get_lmdb()
# extractor.get_pickle()
