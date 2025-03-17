from pathlib import Path

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from xsw_scripts.xps_datagen.DatasetObjects import SpectrumDataset
from xsw_scripts.xps_datagen.NNDataBuilderPeter import *
from xsw_scripts.xsw_profile.predict_modulation import predict_modulation

NUM_PEAKS = 1

#dataset = SpectrumDataset("gen", num_spectra=1, num_peaks=NUM_PEAKS, seed=6)
cs = rescale_cs(gen_cs(num_spectra=1, num_peaks=NUM_PEAKS, seed = 11111111111111111111111))
result = predict_modulation(
    data_dir=Path("src/xsw_scripts/fpfpp/"),
    coherent_fraction=0.8,
    coherent_position=0,
    theta=18,
    plotter=0,
    atom_type=7,
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

result2 = predict_modulation(
    data_dir=Path("src/xsw_scripts/fpfpp/"),
    coherent_fraction=0.3,
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

def extend_cs(cs, num_peaks, modulations, lower_bound = -2, upper_bound = 2, plot = True, num_plots_x = 4, num_plots_y = 5):
    #cs a 46 parameter array, where the 15th, 21st, 27th, 33rd, 39th, 45th correspond to the intensities of individual peaks
    if len(modulations) != num_peaks:
        print('number of modulations does not equal number of peaks.')
        return
    
    curve_samples = []
    for i in range(len(modulations)):
        current_sample = []
        current_curve = modulations[i][0] #result of predict modulation: two arrays, one containing e-e_bragg vs xsw profile, the other e-e_bragg vs reflection
        for tuple in current_curve:
            if tuple[0] >= lower_bound and tuple[0] <= upper_bound:
                current_sample.append(tuple)
        curve_samples.append(current_sample)

    ##are predict_modulation outputs consistent? will they have the same number of values - naively assume yes for now
    current_slice = np.copy(cs)
    print(len(current_sample))
    for i in range(len(current_sample)):
        for j in range(num_peaks):
            current_sample = curve_samples[j]
            current_slice[15 + 6*j] = cs[15 + 6*j] * current_sample[i][1]
        if i == 0:
            stacked_slices = np.copy(current_slice)
        else:
            stacked_slices = np.column_stack([stacked_slices, current_slice])

    obtained_spectra = build_spectra(stacked_slices)
    
    
    
    
    if plot == True:
        num_plots = num_plots_x * num_plots_y
        fig, ax = plt.subplots(num_plots_y, num_plots_x, sharey = True)
        count = 0
        current_max_y = 0
        if num_plots == 1:
            ax.plot(obtained_spectra[(len(current_sample)//num_plots)*count][0], obtained_spectra[(len(current_sample)//num_plots)*count][1])
        elif num_plots_x == 1:
            for i in range(num_plots_y):
                ax[i].plot(obtained_spectra[(len(current_sample)//num_plots)*count][0], obtained_spectra[(len(current_sample)//num_plots)*count][1])
        elif num_plots_y == 1:
            for i in range(num_plots_x):
                ax[i].plot(obtained_spectra[(len(current_sample)//num_plots)*count][0], obtained_spectra[(len(current_sample)//num_plots)*count][1])

        else:
            for i in range(num_plots_y):
                for j in range(num_plots_x):
                    count += 1
                    #print(count, (len(current_sample)//num_plots)*count)
                    ax[i, j].plot(obtained_spectra[(len(current_sample)//num_plots)*count][0], obtained_spectra[(len(current_sample)//num_plots)*count][1])
                    #ax[i, j].set_ylim(bottom = 0)

                    if ax[i, j].get_ylim()[1] > current_max_y:
                        current_max_y = ax[i,j].get_ylim()[1]

            #for i in range(num_plots_y):
            #    for j in range(num_plots_x):
            #        ax[i, j].set_ylim(top = current_max_y)

        plt.show()
    
    
    return obtained_spectra

xsw = np.array(extend_cs(cs, 1, [result,], lower_bound = -2, upper_bound = 2, num_plots_x = 6, num_plots_y = 12 , plot = 1))
#202 steps in predict_modulation
#2 by 1679 - 
#num_plots_x = 4
#num_plots_y = 5
#num_plots = num_plots_y * num_plots_x

#ig, ax = plt.subplots(1, 2)
#ax[0].plot((1,2,3,4,5), (1,2,3,4,5))
#ax[1].plot((1,2,3,4,5), (5,4,3,2,1))
#plt.show()

#fig, ax = plt.subplots(4, 5)
#count = 0
#current_max_y = 0
#for i in range(num_plots_x):
#    for j in range(num_plots_y):
#        count += 1
#        print((202//num_plots)*count)
#        #print(num_plots_x * j + i)
#        ax[i, j].plot(xsw[(202//num_plots)*count][0], xsw[(202//num_plots)*count][1])
#        ax[i, j].set_ylim(bottom = 0)
#
#        if ax[i, j].get_ylim()[1] > current_max_y:
#            current_max_y = ax[i,j].get_ylim()[1]

#for i in range(num_plots_x):
#    for j in range(num_plots_y):
#        ax[i, j].set_ylim(top = current_max_y)

    
#plt.show()