"""
Copyright (c) 2025 - Institute of Chemical Research of Catalonia (ICIQ)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import glob
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# List of Molecular CSV
csv_list = list(glob.glob("./qm9s_csv/*.csv"))

# Load IR csv file
df_ir = pd.read_csv("./ir_boraden.csv")
df_raman = pd.read_csv("./raman_boraden.csv")

# Loop over Molecular CSV files
dataset_dict = {}
for idx, csv_file in tqdm(enumerate(csv_list)):
    # Load CSV file with pandas
    df = pd.read_csv(csv_file)
    # Locate 1st CSV file row with Number, original number, SMILE, n atoms
    first_row = list(df.iloc[0])
    # Get IR data from df_ir
    y_axis_ir = np.array(list(df_ir.iloc[idx]))[1:]
    x_axis_ir = np.linspace(500, 4000, 3501)
    ir_spectra = np.vstack([x_axis_ir, y_axis_ir])
    # Get Raman data from df_raman
    y_axis_raman = np.array(list(df_raman.iloc[idx]))[1:]
    x_axis_raman = np.linspace(500, 4000, 3501)
    raman_spectra = np.vstack([x_axis_raman, y_axis_raman])
    # Molecular Info dict
    mol_dict = {"idx_qm9s": first_row[1],
                "idx_qm9": first_row[2],
                "smile": first_row[3],
                "n_atoms": first_row[4],
                "ir_spectra": ir_spectra,
                "raman_spectra": raman_spectra,
    }
    # Append all to dataset_dict
    dataset_dict.update({idx: mol_dict})


# Export RAW data as pickle file
with open("./qm9s_ir_raman.pkl", "wb") as p_file:
    pickle.dump(dataset_dict, p_file)
p_file.close()
