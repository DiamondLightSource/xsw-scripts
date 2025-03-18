from pathlib import Path

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from xsw_scripts.xps_datagen.DatasetObjects import SpectrumDataset
from xsw_scripts.xps_datagen.NNDataBuilderPeter import *
from xsw_scripts.xsw_profile.predict_modulation import predict_modulation
from xps_extension import extend_cs
NUM_PEAKS = 1

#dataset = SpectrumDataset("gen", num_spectra=1, num_peaks=NUM_PEAKS, seed=6)
cs = rescale_cs(gen_cs(num_spectra=1, num_peaks=NUM_PEAKS, seed = 2444666668888888))
result = predict_modulation(
    data_dir=Path("src/xsw_scripts/fpfpp/"),
    coherent_fraction=1,
    coherent_position=0,
    theta=18,
    plotter=0,
    atom_type=6,
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
    )
)

#result2 = predict_modulation(
#    data_dir=Path("src/xsw_scripts/fpfpp/"),
#    coherent_fraction=1,
#    coherent_position=0.5,
#    theta=18,
#    plotter=0,
#    atom_type=6,
#    principal_qn=1,
#    azimuthal_qn=0,
#    spin_qn=0.5,
#    alphaB=0,
#    hkl_index=(1, 1, 1),
#    lps0=np.array([3.6149, 3.6149, 3.6149, 90, 90, 90]),
#    xyzs=np.array(
#        [
#            [29, 0.0, 0.0, 0.0, 1],
#            [29, 0.5, 0.5, 0.0, 1],
#            [29, 0.5, 0.0, 0.5, 1],
#            [29, 0.0, 0.5, 0.5, 1],
#        ]
#    )
#)

xsw = np.array(extend_cs(cs, 1, [result], lower_bound = -2, upper_bound = 4, num_plots_x = 16, num_plots_y = 9, plot = 1))