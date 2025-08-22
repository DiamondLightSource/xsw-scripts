import random
import pathlib as Path
import numpy as np
from sys import argv

# takes one input, number of spectra to generate

from xsw_profile.predict_modulation import *
from xps_extension import extend_cs
from xps_datagen.NNDataBuilderPeter import gen_cs, rescale_cs, resize_spectra, minmax_scale

import tensorflow as tf

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


##sys args - number of datasets already created, current dataset being created, number of spectra to generate
num_created_datasets = int(argv[1])
current_dataset_num = int(argv[2])
num_spectra = int(argv[3])
NUM_PEAKS = int(argv[4])


save_path = argv[5]
#random.seed = 37
xsw_dataset = {
    'xsw_curve': [],
    #'e_range': [],
    'cs': [],
    'cs_norm': [],
    #'modulation': [],
    'coh_params': [],
    'mod_width': [],
}
for i in range(num_spectra):
    print(i)
    coh_params = []
    modulations = []
    mod_width = random.gauss(0.3, 0.066)
    while mod_width <= 0:
        mod_width = random.gauss(0.3, 0.066)
    for i in range(NUM_PEAKS):
        c_frac, c_pos = random.uniform(0,1), random.uniform(0,1)

        modulation_i = predict_modulation(
            data_dir=("/home/vol01/scarf1391/xsw_scripts/fpfpp"),
            coherent_fraction=c_frac,
            coherent_position=c_pos,
            theta=18,
            plotter=0,
            atom_type=5,
            principal_qn=1,
            azimuthal_qn=0,
            spin_qn=0.5,
            alphaB=0,
            hkl_index=(1, 1, 1),
            lps0=np.array([3.6149, 3.6149, 3.6149, 90, 90, 90]),
            width = mod_width,
            xyzs=np.array(
                [
                    [29, 0.0, 0.0, 0.0, 1],
                    [29, 0.5, 0.5, 0.0, 1],
                    [29, 0.5, 0.0, 0.5, 1],
                    [29, 0.0, 0.5, 0.5, 1],
                ]
            )
        )
        coh_params.append(c_frac)
        coh_params.append(c_pos)
        modulations.append(modulation_i)
  
    #print(mod_width, coh_params)

    cs_norm = gen_cs(num_spectra = 1, 
                       num_peaks = NUM_PEAKS, 
                       edge_marg = 0.1,
                       cutoff_higher=1)
    cs = rescale_cs(cs_norm)
    
    xsw = (extend_cs( #un-numpy arrayed
    cs = cs, 
    num_peaks = NUM_PEAKS,
    modulations = modulations, 
    lower_bound = -5, 
    upper_bound = 5, 
    num_plots_x = 4, 
    num_plots_y = 4, 
    plot = 0
    )
    )

    #print(max(modulations[0, :]))

    #print(np.array(xsw).shape, cs[0])
    #get_xsw_profile(xsw, cs, True)

    resized_spectra, e_range = resize_spectra(xsw)
    
    #print(resized_spectra.shape)

    #e_scale = np.linspace(e_range[0][0], e_range[0][1], resized_spectra.shape[1])
    #sum_spectra = [0] * resized_spectra.shape[1]
    #min_val, max_val = 0, 0
    #for i in range(resized_spectra.shape[0]):
    #    sum_spectra = sum_spectra + resized_spectra[i][:]

    
    normalised_spectra = minmax_scale(resized_spectra, True)
    #print(normalised_spectra)
    flattened_spectra = normalised_spectra.flatten()


    xsw_dataset['xsw_curve'].append(flattened_spectra)
    xsw_dataset['cs_norm'].append(cs_norm)
    xsw_dataset['cs'].append(cs)
    #xsw_dataset['modulation'].append(modulations)
    xsw_dataset['mod_width'].append([mod_width])
    xsw_dataset['coh_params'].append(coh_params)


with tf.io.TFRecordWriter(path = '{}/{}records{}.tfrecords'.format(save_path, num_spectra, str(num_created_datasets + current_dataset_num))) as writer:
    for i in range(num_spectra):
        #print(i)
        features = {
            'xsw_curve': _float_feature(xsw_dataset['xsw_curve'][i]),
            'cs': _float_feature(xsw_dataset['cs'][i][:, 0]),
            'cs_norm': _float_feature(xsw_dataset['cs_norm'][i][:, 0]),
            #'modulations': _float_feature(xsw_dataset['modulation'][0]),
            'mod_width': _float_feature(xsw_dataset['mod_width'][i]),
            'coh_params': _float_feature(xsw_dataset['coh_params'][i])
        }
        example = tf.train.Example(features = tf.train.Features(feature = features))
        serialized_example = example.SerializeToString()
        writer.write(serialized_example)