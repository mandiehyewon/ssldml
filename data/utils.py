import os
import h5py
import hdf5plugin
from typing import List
import pandas as pd
import numpy as np

import torch

ALL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']   

def load_ecg(
    hd5: h5py.File, date: str,
    target_length: int = 2500, 
    leads: List[str] = ALL_LEADS,
):
    out = np.empty((target_length, len(leads)))
    for i, lead in enumerate(leads):
        lead_array = hd5["ecg"][date][lead][()]
        out[:, i] = np.interp(
            np.linspace(0, 1, target_length),
            np.linspace(0, 1, lead_array.shape[0]),
            lead_array,
        )
    return out