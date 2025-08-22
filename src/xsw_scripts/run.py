from pathlib import Path

import numpy as np
import pandas as pd 
import random
#import matplotlib.pyplot as plt
#from xsw_scripts.xps_datagen.DatasetObjects import SpectrumDataset
from xsw_scripts.xps_datagen.NNDataBuilderPeter import *
from xsw_scripts.xsw_profile.predict_modulation import predict_modulation
from xps_extension import extend_cs
NUM_PEAKS = 1

while True:
    #dataset = SpectrumDataset("gen", num_spectra=1, num_peaks=NUM_PEAKS, seed=6)
    modulation_params = [random.uniform(0,1), random.uniform(0,1), random.gauss(0.3, 0.067)]
    cs = rescale_cs(gen_cs(num_spectra = 1, 
                        num_peaks = NUM_PEAKS, 
                        edge_marg = 0.1,
                        cutoff_higher=1))
    
    result = predict_modulation(
        data_dir=Path("src/xsw_scripts/fpfpp/"),
        coherent_fraction=modulation_params[0],
        coherent_position=modulation_params[1],
        theta=18,
        plotter=0,
        atom_type=5,
        principal_qn=1,
        azimuthal_qn=0,
        spin_qn=0.5,
        alphaB=0,
        width = modulation_params[2],
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
    #
    xsw = np.array(extend_cs(
        cs = cs, 
        num_peaks = NUM_PEAKS, 
        modulations = [result], 
        lower_bound = -5, 
        upper_bound = 5, 
        num_plots_x = 4, 
        num_plots_y = 4, 
        plot = 1
        )
    )
    resized_spectra, e_range = resize_spectra(xsw)
    #frac_peak_pos = (cs[10] - e_range[0][0])/e_range[0][1]
    e_scale = np.linspace(e_range[0][0], e_range[0][1], resized_spectra.shape[1])
    sum_spectra = [0] * resized_spectra.shape[1]
    min_val, max_val = 0, 0
    for i in range(resized_spectra.shape[0]):
        sum_spectra = sum_spectra + resized_spectra[i][:]
        print(min(resized_spectra[i][:]))
        max_val += max(resized_spectra[i][:])

    rescaled = minmax_scale(sum_spectra)
    
    print('gaussian width {}\nlorentzian width {}\nmodulation width {}'.format(cs[11], cs[12], modulation_params[2]))
    
    plt.plot(rescaled)
    #plt.vlines(cs[10], plt.ylim()[0], plt.ylim()[1])
    #plt.hlines([plt.ylim()[1]*0.33, plt.ylim()[1]*.5, plt.ylim()[1]*.67],
    #           [cs[10]-(cs[11]/2), cs[10]-(cs[12]/2), cs[10]-(modulation_params[2]/2)],
    #           [cs[10]+(cs[11]/2), cs[10]+(cs[12]/2), cs[10]+(modulation_params[2]/2)])
    plt.gca().invert_xaxis()
    plt.show()