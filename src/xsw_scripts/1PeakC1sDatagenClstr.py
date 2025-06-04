import random
import pathlib as Path
import numpy as np
from sys import argv

# takes one input, number of spectra to generate

from xsw_profile.predict_modulation import *
from xps_extension import extend_cs
from xps_datagen.NNDataBuilderPeter import gen_cs, rescale_cs, resize_spectra, minmax_scale

import tensorflow as tf

##sys args - number of datasets already created, current dataset being created, number of spectra to generate

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

NUM_PEAKS = 1

#random.seed = 37
num_spectra = int(argv[3])
xsw_dataset = {
    'xsw_curve': [],
    #'e_range': [],
    'cs': [],
    'cs_norm': [],
    'modulation': [],
    'mod_params': []
}
count = 0

while count < num_spectra:
    try:
        mod_params = [random.uniform(0,1), random.uniform(0,1), random.gauss(0.3, 0.066)]
        #while mod_params[2] <= 0:
        #    mod_params[2] = random.gauss(0.3, 0.066)
        modulation = predict_modulation(
            data_dir=("/home/vol01/scarf1391/xsw_scripts/fpfpp"),
            coherent_fraction=mod_params[0],
            coherent_position=mod_params[1],
            theta=18,
            plotter=0,
            atom_type=5,
            principal_qn=1,
            azimuthal_qn=0,
            spin_qn=0.5,
            alphaB=0,
            hkl_index=(1, 1, 1),
            lps0=np.array([3.6149, 3.6149, 3.6149, 90, 90, 90]),
            width = mod_params[2],
            xyzs=np.array(
                [
                    [29, 0.0, 0.0, 0.0, 1],
                    [29, 0.5, 0.5, 0.0, 1],
                    [29, 0.5, 0.0, 0.5, 1],
                    [29, 0.0, 0.5, 0.5, 1],
                ]
            )
        )
        
        cs_norm = gen_cs(num_spectra = 1, 
                        num_peaks = NUM_PEAKS, 
                        edge_marg = 0.1,
                        cutoff_higher=1)
        cs = rescale_cs(cs_norm)
        
        xsw = (extend_cs( #un-numpy arrayed
        cs = cs, 
        num_peaks = NUM_PEAKS, 
        modulations = [modulation], 
        lower_bound = -5, 
        upper_bound = 5, 
        num_plots_x = 8, 
        num_plots_y = 12, 
        plot = 0
        )
        )

        resized_spectra, e_range = resize_spectra(xsw)

        #sum_spectra = [0] * resized_spectra.shape[1]
        #for i in range(resized_spectra.shape[0]):
        #    sum_spectra = sum_spectra + resized_spectra[i][:]

        normalised_spectra = minmax_scale(resized_spectra, two_d = True)
        flattened_spectra = normalised_spectra.flatten()

        xsw_dataset['xsw_curve'].append(flattened_spectra)
        xsw_dataset['cs_norm'].append(cs_norm)
        xsw_dataset['cs'].append(cs)
        xsw_dataset['modulation'].append(modulation)
        xsw_dataset['mod_params'].append(mod_params)

        count += 1
    except ValueError as e:
        print(e)
        print("errored with following message: '{}'\nmodulation parameters:\ncoherent fraction:{}\ncoherent position:{}\nmodulation width:{}".format(e, mod_params[0], mod_params[1], mod_params[2]))



with tf.io.TFRecordWriter(path = '/work4/dls/scarf1391/1PeakC1sData/{}records{}.tfrecords'.format(num_spectra, str(int(argv[2]) + int(argv[1])))) as writer:
    for i in range(num_spectra):
        #print(i)
        features = {
            'xsw_curve': _float_feature(xsw_dataset['xsw_curve'][i]),
            'cs': _float_feature(xsw_dataset['cs'][i][:, 0]),
            'cs_norm': _float_feature(xsw_dataset['cs_norm'][i][:, 0]),
            'modulation': _float_feature(xsw_dataset['modulation'][i][0][:, 1]),
            'mod_params': _float_feature(xsw_dataset['mod_params'][i])
        }
        example = tf.train.Example(features = tf.train.Features(feature = features))
        serialized_example = example.SerializeToString()
        writer.write(serialized_example)