from pathlib import Path

import numpy as np
from xsw_scripts.xps_datagen.DatasetObjects import SpectrumDataset
from xsw_scripts.xsw_profile.predict_modulation import predict_modulation

dataset = SpectrumDataset("gen", num_spectra=1, num_peaks=4, seed=6)
print(dataset.inputs.shape)
result = predict_modulation(
    data_dir=Path("src/xsw_scripts/fpfpp/"),
    coherent_fraction=0.8,
    coherent_position=0,
    theta=18,
    plotter=False,
    atom_type=8,
    principal_qn=1,
    azimuthal_qn=0,
    spin_qn=0.5,
    alphaB=0,
    hkl_index=(1, 1, 1),
    lps0=np.array([3.6149, 3.6149, 3.6149, 90, 90, 90]),
    xyzs=np.array(
        [
            [29, 0.0, 0.0, 0.0, 1],
            [29, 0.5, 0.5, 0.0, 1],
            [29, 0.5, 0.0, 0.5, 1],
            [29, 0.0, 0.5, 0.5, 1],
        ]
    ),
    width=0.2,
)
print(result[0].shape)
