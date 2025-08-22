import numpy as np
import matplotlib.pyplot as plt
from xps_datagen.NNDataBuilderPeter import *

def extend_cs(cs, num_peaks, modulations, lower_bound = -2, upper_bound = 2, plot = False, num_plots_x = 4, num_plots_y = 5):
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
        interval = len(current_sample)//num_plots
        for ax in fig.get_axes():
            ax.plot(obtained_spectra[interval*count][0], obtained_spectra[interval * count][1])
            ax.xaxis.set_inverted(True)
            count += 1

        plt.show()
    
    
    #obtained_spectra an n * 2 * k array
    #n the number of elements of predict_modulation with e_bragg in [lower_bound, upper_bound]
    #2* k array containing the energy scale and intensity scale
    return obtained_spectra

def get_xsw_profile(xsw, cs, plot = False):
    resized_spectra, e_range = resize_spectra(xsw)
    e_scale = np.linspace(e_range[0][0], e_range[0][1], resized_spectra.shape[1])
    sum_spectra = [0] * resized_spectra.shape[1]
    for i in range(resized_spectra.shape[0]):
        sum_spectra = sum_spectra + resized_spectra[i][:]

    if plot:
        plt.plot(e_scale, sum_spectra)
        plt.vlines(cs[10], plt.ylim()[0], plt.ylim()[1])
        plt.gca().invert_xaxis()
        plt.show()

    return(resized_spectra, e_scale)