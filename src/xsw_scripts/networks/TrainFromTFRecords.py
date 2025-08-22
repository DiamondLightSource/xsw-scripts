import sys
import copy
# from tkinter.filedialog import askopenfilenames as aof  ### COMMENTED OUT FOR CLUSTER
# from tkinter import Tk  ### COMMENTED OUT FOR CLUSTER
import numpy as np
# import matplotlib.pyplot as plt  ### COMMENTED OUT FOR CLUSTER
import pickle
import tensorflow as tf

# These are you friends:
#   https://www.tensorflow.org/tutorials/load_data/tf_records
#   https://www.tensorflow.org/guide/datasets
#   https://www.tensorflow.org/guide/performance/datasets
#
# Introduction to TensorFlow Records:
#
# TFRecords are TensorFlow's standard binary file format for storing training data. They are especially useful when the
# datasets are too large to fit inside memory, e.g. in numpy arrays. The TensorFlow data API also has several features
# which allow increased efficiency and speed during training.
#
# A filled TFRecord is comprised of so-called 'feature columns'. In our case the features will be the 'inputs', the
# one-hot vectors for numbers of peaks, the parameter arrays etc. Each datapoint (or 'example', in TensorFlow parlance)
# is therefore distributed across the different feature columns in a similar manner to how they are in my
# SpectrumDataset objects.
#
# There are 3 (TensorFlow-defined) types of data that can be stored in TFRecords: floats, int64s and bytes. All the
# features of each spectrum must be converted to one of these 3 types, in a TensorFlow 'Feature' object, before the
# spectrum can be written to a TFRecord as a TensorFlow Example. This is what the following 3 helper functions do:


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
# Converts the inputted array of values into a TensorFlow 'FloatList', which is then wrapped into a TensorFlow
# 'Feature' object. Appropriate for floating point types (all precisions).


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# Converts the inputted array of values into a TensorFlow 'Int64List', which is then wrapped into a TensorFlow
# 'Feature' object. Appropriate for all integer precisions, unsigned integers and booleans.


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
# Converts the inputted array of values into a TensorFlow 'BytesList', which is then wrapped into a TensorFlow
# 'Feature' object. Appropriate for strings and bytes values.


# my_dataset_to_tfrecord
#
# Takes the data contained in a SpectrumDataset object and saves it in binary format to a TFRecords file. This format is
# the best way to store very large datasets, such as ours, for fast and efficient training.
#
# Arguments:
#
#   dataset:
#       The SpectrumDataset object containing the data to be written to the TFRecord. Must contain generated data.
#
#   file_name:
#       The name of the TFRecord binary file to be created.
#
#   save_dir (= ''):
#       The directory in which to create the file.
#
#   report: (= None)
#       If not NoneType, a printout is produced after the TFRecord has been completed. This is useful mostly if the
#       function is called inside a loop, producing several files, and you want to make sure it's working.

def my_dataset_to_tfrecord(dataset, file_name, save_dir='', report=None):

    with tf.io.TFRecordWriter(path='{}/{}.tfrecords'.format(save_dir, file_name)) as writer:
        # This line creates a writer object which will create the TFRecord and write binary data to it. Using a 'with'
        # statement means that the writer will automatically close once the following block finishes executing.

        for i in range(len(dataset.inputs)):
            # Iterate over all the spectra in the SpectrumDataset dataset.

            features = {'inputs':               _float_feature(dataset.inputs[i]),
                        'peak_count_oh':        _int64_feature(dataset.peak_count_OH[i]),
                        #'peak_pos_mh':          _int64_feature(dataset.peak_pos_MH[i]),
                        # 'peak_pos_oh_concat':   _int64_feature(dataset.peak_pos_OH_concat[i]),
                        'cs_norm':              _float_feature(dataset.cs_norm[:, i]),
                        'cs':                   _float_feature(dataset.cs[:, i])}
            # For each spectrum ('example'), retrieve all the features of that spectrum that you want to save to the
            # TFRecord, convert them each to the appropriate TensorFlow data types (in a 'Feature' object), and append
            # them to a dictionary with their feature names as keys. This defines what feature columns the TFRecord will
            # contain, and their names.

            example = tf.train.Example(features=tf.train.Features(feature=features))
            # Create a TensorFlow 'Example' object from the features dictionary

            serialized_example = example.SerializeToString()
            # Serialize this Example object into a binary string.

            writer.write(serialized_example)
            # Have the writer object write the serialized Example to the file.

    sys.stdout.flush()
    # Ensure that the system's standard output buffer is empty.

    if report:
        print('TFRecord Complete')
        # Print a completion message if 'report' was not NoneType.


# create_tfr_dataset
#
# Assembles a TensorFlow Dataset object from 1 or more TFRecords stored on a hard drive, for training a neural network.
# The dataset can be made to return different labels for different training tasks. This function is called inside
# 'train_tfr' and generallly shouldn't need to be called outside this context.
#
# I would recommend also referring to the official documentation for the tf.data API on tensorflow.org - just in case
# I've been at all inaccurate in my explanations. It's generally quite difficult to understand and also to explain.

# Arguments:
#
#   filenames: (= None)
#       A list containing the names of the TFRecords to be used in building the training dataset. If not a list, a
#       dialogue is opened allowing you to hand pick files (default behaviour).
#
#   file_dir: (= '')
#       The directory containing the TFRecord files.
#
#   train_type: (= 'count')
#       Specifies what kind of labels the dataset should provide (i.e. what kind of network is to be trained). Details
#       explained within the function, options are as follows:
#
#       peak counting - 'count', 'c', 0 (default)
#       peak point-position finding - 'pos', 'p', 1
#       peak multi-hot position finding - 'posMH', 'pmh', 2
#       peak one-hot position finding (single concatenated output layer) - 'posOHconcat', 'pohc', 3
#       peak one-hot position finding (multiple one-hot outputs) - 'posOHmultiout', 'pohmo', 4
#
#       Note that the last option does NOT work at present, as the keras Model API doesn't appear to support being fed
#       multiple output targets from a TensorFlow dataset object. Instead it expects single 'input' and 'label' tensors.
#       It is possible, however, to train a Keras Model with multiple outputs using numpy arrays to feed data - although
#       you are limited by your RAM as the whole dataset needs to be loaded in at once. It should become possible to
#       train multiple outputs on a Keras Model using the tf.data API in the near future, since TensorFlow are
#       interested in pushing both APIs as standard tools for training models. Hence I have left the code in here.
#
#   num_peaks: (= 6)
#       The number of peaks in the examples to be trained on. This is only relevant for point-position and concatenated
#       one-hot position training in which a single number of peaks is trained on by a given network, and in each case
#       the output size depends on the number of peaks. In other cases, the value should be left as 6.
#
#   max_num_peaks: (= 6)
#       This is the maximum number of peaks that the spectra contained in the TFRecord can hold (see NNDataBuilder).
#       This value is needed for retrieving the parameter arrays, 'cs' and 'cs_norm', because we need to know their
#       sizes in order to correctly parse each example into memory. Also for when we retrieve the 'one-hot' positions
#       feature column - these have a length /pos_bins/*/max_num_peaks/. This value will always be 6 unless you made
#       your own datasets with a different maximum number of peaks.
#
#   pos_bins: (= 200)
#       For one-hot and mutli-hot position labels, this parameter specifies the lengths of these vectors. This is needed
#       both for parsing these features in and for correctly slicing them up to get the labels we want for training on
#       different numbers of peaks.
#
#   input_shape: (= (200, 1))
#       Simply the shape of the 'inputs' tensor - needed for parsing it in. As mentioned before, the seemingly redundant
#       dimension is necessary for the 1D convolution operation in the networks.
#
#   batch_size: (= 3000)
#       During training, this is the number of spectra that the model is exposed to before a weight update is carried
#       out.
#
#   block_length: (= 10)
#       Controls how finely examples from each file are interleaved in assembling the dataset onject. Let's say we have
#       6 files, one containing each number of peaks from 1 to 6. With a block length of 5, the dataset object assembled
#       from these files would contain examples arranged as follows:
#
#       [ 1 1 1 1 1 | 2 2 2 2 2 | 3 3 3 3 3 | 4 4 4 4 4 | 5 5 5 5 5 | 6 6 6 6 6 | 1 1 1 1 1 | 2 2 2 2 2 | etc. ]
#
#   shuffle_buffer: (= 100000)
#       To get good randomisation of the order of the examples from each file, shuffling is also necessary. This
#       parameter controls how many examples are taken to be shuffled at once (we can't shuffle the entire dataset as
#       shuffling involves loading into memory).
#
#   num_parallel_calls: (= 4)
#       At various stages of the data pipeline, transformations are carried out on the elements of the data. This
#       variable controls how many CPU threads are used for these - depending on baseline efficiency, the processes are
#       sped up by splitting them over multiple threads.
#
# Outputs:
#
#   dataset:
#       A tf.data Dataset object which returns tensors of training inputs and labels from TFRecords binary data stored
#       on disk.
#
#   num_files:
#       The number of TFRecords files used to create the dataset. This is returned purely so that it can be passed to
#       'train_tfr' and written to a text file containing all the model and training details.

def create_tfr_dataset(filenames=None,
                       file_dir='',
                       train_type='count',
                       num_peaks=6,
                       max_num_peaks=6,
                       pos_bins=200,
                       input_shape=(200, 1),
                       batch_size=3000,
                       block_length=10,
                       shuffle_buffer=100000,
                       num_parallel_calls=4):

    if not isinstance(filenames, list):
        print("no files")
        #root = Tk()
        #root.withdraw()
        #file_paths = list(aof(initialdir=file_dir, title='Select Files:',
        #                      filetypes=(('tfrecords files', '*.tfrecords'), ('all files', '*.*'))))
        #root.destroy()
        # If a list wasn't provided for the 'filenames' variable, opens a dialogue that lets you hand pick the TFRecord
        # yourself.

    else:
        file_paths = ['{0}/{1}'.format(file_dir, filename) for filename in filenames]
        # If a list was provided, get a list of the full file paths for the TFRecords files.

    num_files = len(file_paths)
    # Get the number of files from the length of the list containing the path names.

    files = tf.data.Dataset.from_tensor_slices(file_paths)
    # Bundle the file path strings into a tf.data Dataset object. Note this does not yet contain our actual data, rather
    # it simply contains the strings of the file paths.

    dataset = files.interleave(map_func=lambda filename: tf.data.TFRecordDataset(filename)
                               .prefetch(num_files*block_length),
                               cycle_length=num_files,
                               block_length=block_length,)
    # num_parallel_calls=num_parallel_calls)

    # This line creates a tf.data Dataset object directly from the TFRecord files (using the path names in 'files'). The
    # 'interleave' method applies the function 'map_func' across all the elements of the 'files' dataset and then
    # interleaves the result, returning a new Dataset object. 'map_func' is converting filename strings into tf.data
    # Datasets containing the binary data from the TFRecords on disk. 'cycle_length' controls how many input elements
    # (file names) are processed concurrently into Datasets, while 'block_length' controls how many elements (of the
    # binary data) are taken from each of these datasets per cycle. The prefetch method preloads the next elements of
    # those datasets, which makes the whole process more effient. See https://www.tensorflow.org/guide/datasets for a
    # more thorough explanation and tutorial for the tf.data API.

    features_dict = {'inputs': tf.io.FixedLenFeature(shape=input_shape, dtype=tf.float32)}
    # Initialize the dictionary containing the names of the features we want to retrieve from the dataset, together with
    # their shapes and intended data types. 'inputs' will always be required for obvious reasons, but the training
    # labels will change depending on the type of network being trained.

    if train_type in ('count', 'c', 0):
        # Make the dataset produce data for training a peak counting network.

        features_dict['peak_count_oh'] = tf.io.FixedLenFeature(shape=(6,), dtype=tf.int64)
        # Update the features dictionary to contain the name of the feature column containing the one-hot vectors for
        # number of peaks, together with their shape and data type.

        dataset = dataset.map(map_func=lambda serialized: tf.io.parse_single_example(serialized=serialized,
                                                                                  features=features_dict),
                              num_parallel_calls=num_parallel_calls)
        # Deserialise Examples from the dataset into FixedLenFeature objects containing a specific data type according
        # to the contents of 'feature_dict'. Feature columns not in 'feature_dict' are ignored.

        dataset = dataset.map(map_func=lambda features:

                              (features['inputs'],
                               features['peak_count_oh']),

                              num_parallel_calls=num_parallel_calls)
        # Use a final mapping function to get the 2 data tensors. The dataset will return this pair of tensors when the
        # training loop calls for it. In this case these features can be taken as is without any need for further
        # slicing etc.

    elif train_type in ('pos', 'p', 1):
        # Make the dataset produce data for training a peak point-position finding network.

        e_ind = np.array(range(10, 10 + 6 * max_num_peaks, 6))
        # Assign the peak energy/position indices so the fractional peak positions can be retrieved from the normalised
        # parameter array.

        features_dict['cs_norm'] = tf.io.FixedLenFeature(shape=(max_num_peaks*6+10,), dtype=tf.float32)
        # Update the features dictionary to contain the name of the feature column containing the normalised parameter
        # arrays (which contains within it the fractional peak positions), together with their shape and intended data
        # type.

        dataset = dataset.map(map_func=lambda serialized: tf.io.parse_single_example(serialized=serialized,
                                                                                  features=features_dict),
                              num_parallel_calls=num_parallel_calls)
        # De-serialize Examples from the dataset into FixedLenFeature objects containing a specific data type according
        # to the contents of 'feature_dict'. Feature columns not in 'feature_dict' are ignored.

        dataset = dataset.map(map_func=lambda features:

                              (features['inputs'],
                               tf.gather(features['cs_norm'], e_ind[:num_peaks])),

                              num_parallel_calls=num_parallel_calls)
        # Use a final mapping function to get the 2 data tensors. The dataset will return this pair of tensors when the
        # training loop calls for it. In this case we've used tf.gather and some or all of the indices in 'e_ind'
        # (depending on how many peaks we're going to train on) to collect the fractional positions from the normalised
        # parameter array together into a single tensor.

    elif train_type in ('width', 'w', 5):
        # Make the dataset produce data for training a width finding network.

        #e_ind = np.array([range(10, 10 + 6 * max_num_peaks, 6),range(11,11 +6 * max_num_peaks, 6])
        e_ind = np.array([nn*6+11+ll for ll in range(2) for nn in range(max_num_peaks)])
        i_ind = np.array([nn*6+10 for nn in range(max_num_peaks)])
        # Assign the peak width indices so the fractional peak positions can be retrieved from the normalised
        # parameter array.
        
        features_dict['cs_norm'] = tf.io.FixedLenFeature(shape=(max_num_peaks*6+10,), dtype=tf.float32)
        #features_dict['inputs'] = (tf.io.FixedLenFeature(shape=input_shape, dtype=tf.float32),tf.gather(features_dict['cs_norm'], i_ind[:num_peaks]))
                # Update the features dictionary to contain the name of the feature column containing the normalised parameter
        # arrays (which contains within it the fractional peak positions), together with their shape and intended data
        # type.

        dataset = dataset.map(map_func=lambda serialized: tf.io.parse_single_example(serialized=serialized,
                                                                                  features=features_dict),
                              num_parallel_calls=num_parallel_calls)
        # De-serialize Examples from the dataset into FixedLenFeature objects containing a specific data type according
        # to the contents of 'feature_dict'. Feature columns not in 'feature_dict' are ignored.

        dataset = dataset.map(map_func=lambda features:
                              ((features['inputs'],tf.gather(features['cs_norm'], i_ind[:num_peaks])),
                               tf.gather(features['cs_norm'], e_ind[:num_peaks*2])),
                              num_parallel_calls=num_parallel_calls)


    

        
        # Use a final mapping function to get the 2 data tensors. The dataset will return this pair of tensors when the
        # training loop calls for it. In this case we've used tf.gather and some or all of the indices in 'e_ind'
        # (depending on how many peaks we're going to train on) to collect the fractional positions from the normalised
        # parameter array together into a single tensor.


    elif train_type in ('posMH', 'pmh', 2):
        # Make the dataset produce data for training a multi-hot peak position finding network.

        features_dict['peak_pos_mh'] = tf.io.FixedLenFeature(shape=(pos_bins,), dtype=tf.int64)
        # Update the features dictionary to contain the name of the feature column containing the multi-hot peak
        # position vectors, together with their shape and intended data type.

        dataset = dataset.map(map_func=lambda serialized: tf.io.parse_single_example(serialized=serialized,
                                                                                  features=features_dict),
                              num_parallel_calls=num_parallel_calls)
        # Deserialise Examples from the dataset into FixedLenFeature objects containing a specific data type according
        # to the contents of 'feature_dict'. Feature columns not in 'feature_dict' are ignored.

        dataset = dataset.map(map_func=lambda features:

                              (features['inputs'],
                               features['peak_pos_mh']),

                              num_parallel_calls=num_parallel_calls)
        # Use a final mapping function to get the 2 data tensors. The dataset will return this pair of tensors when the
        # training loop calls for it. In this case these features can be taken as is without any need for further
        # slicing etc.

    elif train_type in ('posOHconcat', 'pohc', 3):
        # Make the dataset produce data for training a concatenated one-hot peak position finding network.

        features_dict['peak_pos_oh_concat'] = tf.io.FixedLenFeature(shape=(max_num_peaks*pos_bins,), dtype=tf.int64)
        # Update the features dictionary to contain the name of the feature column containing the concatenated one-hot
        # peak position vectors, together with their shape and intended data type.

        dataset = dataset.map(map_func=lambda serialized: tf.io.parse_single_example(serialized=serialized,
                                                                                  features=features_dict),
                              num_parallel_calls=num_parallel_calls)
        # De-serialize Examples from the dataset into FixedLenFeature objects containing a specific data type according
        # to the contents of 'feature_dict'. Feature columns not in 'feature_dict' are ignored.

        dataset = dataset.map(map_func=lambda features:

                              (features['inputs'],
                               tf.slice(features['peak_pos_oh_concat'], [0], [num_peaks*pos_bins])),

                              num_parallel_calls=num_parallel_calls)
        # Use a final mapping function to get the 2 data tensors. The dataset will return this pair of tensors when the
        # training loop calls for it. In this case we've used tf.slice to get the appropriate slice length from the
        # concatenated one-hot vectors according to how many peaks we're going to train on.

    elif train_type in ('posOHmultiout', 'pohmo', 4):
        # NOTE: THIS MODE DOES NOT WORK - CURRENT VERSION OF TENSORFLOW.KERAS' MODELS DO NOT WORK WITH TF.DATA DATASETS
        # THAT RETURN MORE THAN 2 TENSORS (1 INPUTS, 1 LABELS). Future versions of TensorFlow should address this.

        # Make the dataset produce data for training a multiple outputs one-hot peak position finding network.

        features_dict['peak_pos_oh_concat'] = tf.io.FixedLenFeature(shape=(max_num_peaks*pos_bins,), dtype=tf.int64)
        # Update the features dictionary to contain the name of the feature column containing the concatenated one-hot
        # peak position vectors, together with their shape and intended data type.

        dataset = dataset.map(map_func=lambda serialized: tf.io.parse_single_example(serialized=serialized,
                                                                                  features=features_dict),
                              num_parallel_calls=num_parallel_calls)
        # De-serialize Examples from the dataset into FixedLenFeature objects containing a specific data type according
        # to the contents of 'feature_dict'. Feature columns not in 'feature_dict' are ignored.

        dataset = dataset.map(map_func=lambda features:

                              (features['inputs'],
                               tf.slice(features['peak_pos_oh_concat'], [0], [pos_bins]),
                               tf.slice(features['peak_pos_oh_concat'], [pos_bins], [pos_bins]),
                               tf.slice(features['peak_pos_oh_concat'], [2*pos_bins], [pos_bins]),
                               tf.slice(features['peak_pos_oh_concat'], [3*pos_bins], [pos_bins]),
                               tf.slice(features['peak_pos_oh_concat'], [4*pos_bins], [pos_bins]),
                               tf.slice(features['peak_pos_oh_concat'], [5*pos_bins], [pos_bins])),

                              num_parallel_calls=4)
        # Use a final mapping function to get the 7 data tensors - one input and 6 outputs (1 for each peak). The
        # dataset will return these tensors when the training loop calls for it. In this case we've used tf.slice to get
        # the appropriate sections from the concatenated one-hot vectors for each of the 6 outputs. This training mode
        # assumes that you're looking for the positions of 6 peaks or all numbers of peaks (1 - 6) simultaneously. In
        # the latter case, for examples with less than 6 peaks, some of the labels - those for absent peaks - will
        # contain only 0s. You could also write similar code as above to get tensors for another specific number of
        # peaks less than 6.

    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Shuffle the order in which Examples are returned from the dataset. 'shuffle_buffer' determines how many examples
    # are loaded into memory to be shuffled at once (we can't shuffle the whole dataset, default is 100000).

    dataset = dataset.repeat()
    # This ensures that the dataset will keep providing tensors for as long as the training routine calls upon it - i.e.
    # it resets to the start after each training epoch.

    dataset = dataset.batch(batch_size=batch_size)
    # This batches the data into groups of /batch_size/ elements, according to the batch size you want to train your
    # model with.

    dataset = dataset.prefetch(1)
    # This final method call has the dataset preload the next batch of data while the model trains on the current batch.
    # Reduces CPU downtime.

    return dataset, num_files


# train_tfr
#
# Trains a neural network model using a tf.data dataset assembled from TFRecords files. The type of training (e.g. for
# peak counting, point-position finding etc.) can be selected. This function wraps 'create_tfr_dataset' and calls on it
# to create the training dataset. Hence, several of the arguments below are the same as those in 'create_tfr_dataset'.
#
# Arguments:
#
#   model:
#       A pre-initialised tf.keras Model object. It can be totally new and untrained, or be a previously trained model
#       which is now to be trained further.
#
#   num_spectra:
#       The total number of spectra to be trained on (across all the TFRrecords used to build the training dataset).
#
# ################### The following argument descriptions are copy/pasted from 'create_tfr_dataset' ####################
#
#   filenames: (= None)
#       A list containing the names of the TFRecords to be used in building the training dataset. If not a list, a
#       dialogue is opened allowing you to hand pick files.
#
#   file_dir: (= '')
#       The directory containing the TFRecord files.
#
#   train_type: (= 'count')
#       Specifies what kind of labels the dataset should provide (i.e. what kind of network is to be trained). Details
#       explained within the function, options are as follows:
#
#       peak counting - 'count', 'c', 0 (default)
#       peak point-position finding - 'pos', 'p', 1
#       peak multi-hot position finding - 'posMH', 'pmh', 2
#       peak one-hot position finding (single concatenated output layer) - 'posOHconcat', 'pohc', 3
#       peak one-hot position finding (multiple one-hot outputs) - 'posOHmultiout', 'pohmo', 4
#
#       Note that the last option does NOT work at present, as the keras Model API doesn't appear to support being fed
#       multiple output targets from a tf.data dataset object. Instead it expects single 'input' and 'label' tensors.
#
#   num_peaks: (= 6)
#       The number of peaks in the spectra to be trained on. This is only relevant for point-position and concatenated
#       one-hot position training in which a single number of peaks is trained on by a given network, and in each case
#       the output size depends on the number of peaks. In other cases, the value should be left as 6.
#
#   pos_bins: (= 200)
#       For one-hot and mutli-hot position labels, this parameter specifies the lengths of these vectors. This is needed
#       both for parsing these features in and for correctly slicing them up to get the labels we want for the different
#       training scenarios.
#
########################################################################################################################
#
#   val_data: (= None)
#       A spectrumDataset object (i.e. mine, not tf.data's) containing the data the model is to be validated on after
#       each epoch as it is trained. Default is no validation data.
#
#   val_steps: (= 10)
#       The number of batches to split the validation data into when validating the model at the end of each epoch.
#       Only important if using a very large validation set that might not fit into GPU memory all at once.
#
#   epochs: (= 10)
#       The number of epochs to train the model for. An epoch is declared complete once the model has been exposed to
#       all the data in the training set once. The total number of weight updates depends on the batch size, the total
#       size of the training dataset and the number of epochs.
#
#   batch_size: (= 3000)
#       The number of spectra that the network is exposed to before a weight update is carried out.
#
#   verbose: (= 2)
#       The detail of the progress and training metrics printouts during training. 2 - prints an update after each
#       epoch, 1 - prints an update after each iteration (weight update), 0 - silent.
#
#   initial_epoch: (= 0)
#       Determines what value the epochs are counted from. If you've loaded in a model that was previously trained for,
#       say, 20 epochs, and now want to train it some more, you could set this variable to 20 and it would resume
#       counting as if from where it left off. Not terribly important.
#
#   save_prefix: (= 'test')
#       Common name element for all the outputted files (e.g. the model hdf5, pickled history, plots etc.).
#
#   save_dir: (= '')
#       The directory where the training outputs, including the model itself, are to be saved.
#
#   hyp_par: (= None)
#       The dictionary containing all the hyper-parameters of the model (should have been produced by 'build_cnn'
#       alongside the model itself). This information, if provided, is stored together with the model summary and
#       training hyper-paramters in a text file.
#
#   plots: (= True)
#       If True, create plots of the training history with respect to each of the metrics (training and validation) for
#       each of the model's outputs (if more than one). Note that this doesn't work on the cluster, and should be set to
#       False when training there.
#
#   save_history: (= True)
#       If True, save the history dictionary produced by the training procedure using pickle as a pk1 file. This object
#       contains the same data that would be plotted (see above).
#
#   save_model: (= True)
#       If True, save the trained model as an hdf5 file. This can easily be loaded back into python using keras'
#       'load_model' method of the 'Model' class.

def train_tfr(model,
              num_spectra,
              filenames=None,
              file_dir='',
              train_type='count',
              num_peaks=6,
              pos_bins=200,
              val_data=None,
              val_steps=10,
              epochs=10,
              batch_size=3000,
              verbose=2,
              initial_epoch=0,
              save_prefix='test',
              save_dir='',
              hyp_par=None,
              plots=True,
              save_history=True,
              save_model=True):

    if batch_size > num_spectra:
        batch_size = num_spectra
        # Checks if the requested batch size is larger than the dataset to be trained on. If so, then the batch size is
        # reduced to the dataset size.

    if val_data:
        # The following lines prepare the validation data if any was provided.
        if train_type in ('pos', 'p', 1):
            e_ind = np.array(range(10, len(val_data.cs_norm), 6))
        elif train_type in ('width','w',5): 
            e_ind = np.array([nn*6+11+ll for ll in range(2) for nn in range(num_peaks)])
            i_ind = np.array([nn*6+10 for nn in range(num_peaks)])

        # Assign peak energy (position) indices within the parameter arrays.

        val_x = np.expand_dims(val_data.inputs, axis=2)
        # Get the validation dataset's inputs (resized and normalised intensity scales). The dimensions are expanded due
        # to shape requirements by the convolution operations in TensorFlow/Keras.

        val_y_true = None
        # Initialise the validation labels as a NoneType.

        if train_type in ('count', 'c', 0):
            val_y_true = val_data.peak_count_OH
            # If training is for peak counting, set the validation labels to be the one-hot vectors for number of peaks.

        elif train_type in ('pos', 'p', 1):
            val_y_true = val_data.cs_norm[e_ind[:num_peaks]].transpose()
            # If training is for peak point-position finding, set the validation labels to be the fractional peak
            # positions within the normalised parameter array. 'num_peaks' determines how many positions to retrieve for
            # each spectrum - i.e. what number of peaks is being trained on.
        elif train_type in ('width', 'w', 5):
            val_x = (np.expand_dims(val_data.inputs, axis=2),np.expand_dims(val_data.cs_norm[i_ind[:num_peaks]].transpose(), axis=2))
            val_y_true = val_data.cs_norm[e_ind[:num_peaks*2]].transpose()
            # If training is for peak and width point-position finding, set the validation labels to be the fractional peak
            # positions within the normalised parameter array. 'num_peaks' determines how many positions to retrieve for
            # each spectrum - i.e. what number of peaks and widths is being trained on.

        elif train_type in ('posMH', 'pmh', 2):
            val_y_true = val_data.peak_pos_MH
            # If training is for peak multi-hot position finding, set the validation labels to be the multi-hot vectors
            # for peak positons.

        elif train_type in ('posOHconcat', 'pohc', 3):
            val_y_true = val_data.peak_pos_OH_concat[:, :num_peaks*pos_bins]
            # If training is for peak one-hot position finding (single concatenated output layer), set the validation
            # labels to be the concatenated one-hot vectors for peak positions.

        elif train_type in ('posOHmultiout', 'pohmo', 4):
            val_y_true = [val_data.peak_pos_OH_concat[: j*pos_bins:(j+1)*pos_bins] for j in range(num_peaks)]
            # If training is for peak one-hot position finding (separate outputs for each peak), set the validation
            # labels to be a list of up to 6 arrays (depending on number of peaks, one for each), each containing the
            # appropriate slice of the concatenated one-hot vector. The order of the list is used to assign each label
            # array to each output (first to first etc.). As noted in 'get_tfr_dataset', this mode does not work due to
            # the way tf.data Datasets interface with Keras Models, although this part here actually works fine.

        if isinstance(val_y_true, np.ndarray):
            val_data = (val_x, val_y_true)
            # If a valid training type was selected, and hence 'val_y_true' is not NoneType, set 'val_data' to be a
            # tuple containing the inputs array and the labels array (or list of arrays). This format is required by
            # Keras for the validation data.

        else:
            val_data = None
            # Otherwise, set 'val_data' to NoneType, which Keras understands to mean that it shouldn't do any
            # validation.

    tfr_dataset, num_files = create_tfr_dataset(train_type=train_type, num_peaks=num_peaks,
                                                batch_size=batch_size, file_dir=file_dir, filenames=filenames)
    # Create and return the training dataset (and number of files) based on the requested training type, number of
    # peaks, training batch size and filenames. The data contained in the chosen files should of course be consistent
    # with the number of peaks selected if training on point or one-hot peak positions.

    history = model.fit(tfr_dataset, validation_data=val_data, epochs=epochs, steps_per_epoch=num_spectra//batch_size,
                        validation_steps=val_steps, verbose=verbose, initial_epoch=initial_epoch)
    # Call the model's fit method to begin training. At the end a 'History' object is produced, whose 'history'
    # attribute is a dictionary of the training history (loss and other metrics with respect to epochs).
    # 'steps_per_epoch' is the number of weight updates to be carried out each epoch - i.e. the number of batches it
    # will be fed. This needs to be declared explicitly since we're using a tf.data Dataset to provide the data - the
    # routine can't see the full length of the dataset (as it's loaded interatively from disk) meaning it can't
    # calculate this itself. If we were using numpy arrays, we'd only have to specify the 'batch_size' argument instead.
    # 'validation_steps' is the number of batches to divide the validation data into for validating the model (at the
    # end of each epoch). If you have a large validation set, it may be necessary to batch it in order for it to fit
    # into GPU memory.

    if isinstance(hyp_par, dict):
        if train_type not in ('width', 'w', 5):
            _write_model_summary(model, num_files, num_spectra, batch_size, epochs, hyp_par, save_dir, save_prefix)
            # If a dictionary of hyper-parameters for the model was provided, write a summary of the model and training
            # hyper-parameters to a text file.
        else:
            _write_model_summary_multiInput(model, num_files, num_spectra, batch_size, epochs, hyp_par, save_dir, save_prefix)

    if save_history:
        with open('{}/{}_history.pk1'.format(save_dir, save_prefix), 'wb') as file:
            pickle.dump(history.history, file)
            # Save the model's history dictionary as a pk1 file using pickle.

    if save_model:
        model.save('{}/{}_model.h5'.format(save_dir, save_prefix))
        # Save the model itself as an hdf5 file.

    if plots:
        _history_plotter(model, history.history, val_data is not None, save_dir, save_prefix)
        # Draw plots of the training history for the loss and other metrics (both training and validation, and for each
        # of the model's outputs if there is more than one).

    return history


# _write_model_summary
#
# Writes details and hyper-parameters of a model and its training into a text file. Should only be called inside
# 'train_tfr'. Note that the layout of the text file assumes quite rigidly that the network has an initial series of
# convolutional layers followed by a series of dense layers, and that each layer has an associtaed dropout layer. It
# also assumes that the Model is linear, and doesn't have branching outputs.
#
# Arguments:
#
#   All are passed from previous functions (see above) and are explained there. In any case, their meaning isn't
#   important here, only the fact that they contain information about the model and training that we want to store.
#
#   save_dir:
#       The directory into which to save the text file.
#
#   save_prefix:
#       The common name fragment given to all the files outputted by the given training run.

def _write_model_summary(model, num_files, num_spectra, batch_size, epochs, hyp_par, save_dir, save_prefix):

    with open('{}/{}_hyp_param_summary.txt'.format(save_dir, save_prefix), 'w') as text_file:
        text_file.write('# TFRecords: {0}\n# Spectra: {1}\nBatch Size: {2}\n# Epochs: {3}\n# Iterations: {4}\n\n'
                        'Input Shape: {5}\n# Conv. Filters: {6}\nConv. Filter Shapes: {7}\nConv. Dropouts: {8}\n'
                        'Dense Shapes: {9}\nDense Dropouts: {10}\nActivations: {11}\n\nOptimiser: {12}\n'
                        'Learning Rate: {13}\nLoss Function: {14}\nOther Metrics: {15}\n'
                        .format(num_files, num_spectra, batch_size, epochs,
                                int(np.ceil(num_spectra//batch_size))*epochs, hyp_par['input_size'],
                                hyp_par['filters'], hyp_par['kernel_size'], hyp_par['conv_drop'],
                                hyp_par['dense_units'], hyp_par['dense_drop'], hyp_par['activations'],
                                hyp_par['optimizer'], hyp_par['learning_rate'], hyp_par['loss'], hyp_par['metrics']))
        # Write the values for 'num_files', 'num_spectra', 'batch_size', 'epochs', 'hyp_par', 'save_dir', 'save_prefix'
        # and all the hyper-parameters contained in 'hyp_par' to the opened text file under appropriate headings.

        model.summary(print_fn=lambda summary: text_file.write(summary + '\n'))
        # Calls the Model's summary method which prints out a table diagram summarising the network architecture.
        # 'print_fn' allows us to specify that the output of the summary method should be written to the open text file.

def _write_model_summary_multiInput(model, num_files, num_spectra, batch_size, epochs, hyp_par, save_dir, save_prefix):

    with open('{}/{}_hyp_param_summary.txt'.format(save_dir, save_prefix), 'w') as text_file:
        text_file.write('# TFRecords: {0}\n# Spectra: {1}\nBatch Size: {2}\n# Epochs: {3}\n# Iterations: {4}\n\n'
                        '# of Inputs: {5}\nInput names: {6}\nInput Shapes: {7}\n# Input Conv. Filters: {8}\n'
                        'Input Conv. Filter Shapes: {9}\nInput Conv. Dropouts: {10}\n'
                        'Input Dense Shapes: {11}\nInput Dense Dropouts: {12}\nInput Activations: {13}\n\n'
                        '# Conv. Filters: {14}\nConv. Filter Shapes: {15}\nConv. Dropouts: {16}\n'
                        'Dense Shapes: {17}\nDense Dropouts: {18}\nActivations: {19}\n\n'
                        'Optimiser: {20}\n'
                        'Learning Rate: {21}\nLoss Function: {22}\nOther Metrics: {23}\n'
                        .format(num_files, num_spectra, batch_size, epochs,
                                int(np.ceil(num_spectra//batch_size))*epochs, hyp_par['number of inputs'],
                                hyp_par['input names'],hyp_par['input_sizes'],hyp_par['input filters'],
                                hyp_par['input filter size'],hyp_par['input filter drop'],hyp_par['input dense units'],
                                hyp_par['input dense drop'],hyp_par['input activation'],
                                hyp_par['combined filters'], hyp_par['combined filter size'], hyp_par['combined filter drop'],
                                hyp_par['combined dense units'], hyp_par['combined dense drop'], hyp_par['activations'],
                                hyp_par['optimizer'], hyp_par['learning_rate'], hyp_par['loss'], hyp_par['metrics']))
        # Write the values for 'num_files', 'num_spectra', 'batch_size', 'epochs', 'hyp_par', 'save_dir', 'save_prefix'
        # and all the hyper-parameters contained in 'hyp_par' to the opened text file under appropriate headings.

        model.summary(print_fn=lambda summary: text_file.write(summary + '\n'))
        # Calls the Model's summary method which prints out a table diagram summarising the network architecture.
        # 'print_fn' allows us to specify that the output of the summary method should be written to the open text file.
# _history_plotter
#
# Creates plots of a trained model's training history for the loss as well as any other metrics that were recorded.
# Plots are drawn for both training and validation - for each of the model's outputs (if more than one). Doesn't work on
# the cluster, and importing matplotlib there seems to mess up the virtual environment. So don't do that.
#
# Arguments:
#
#   'model' and 'history' are passed from previous functions (see above) and are explained there.
#
#   validated:
#       True if validation data was provided during training. If True, plots are created for validation history as well
#       as training history.
#
#   save_dir:
#       The directory into which to save the plots.
#
#   save_prefix:
#       The common name fragment given to all the files outputted by the given training run.
#
#   close: (= True)
#       If True, closes plots as soon as they are generated (default behaviour).

def _history_plotter(model, history, validated, save_dir, save_prefix, close=True):

    metrics_list = list(history.keys())
    # Get a list of the names from the history dictionary of all the metrics used to monitor the model as it trained.
    # 'Metric' here refers to metrics recorded at different outputs, as well as just different metric types.

    new_metrics_list = copy.deepcopy(metrics_list)
    # Create a copy of this list. This allows us to edit one of them while keeping the other for reference.

    for item in metrics_list:
        # Iterate over each of the metrics in 'metrics_list'.

        if 'val_' in item:
            new_metrics_list.remove(item)
            # If the metric is a validation metric, delete it from 'new_metric_list'. This is done because when it comes
            # to plotting, we want to plot both training and validation for a given metric on the same graph - and this
            # gives us a list of just one of the two to iterate over for making plot figures.

    #x = range(1, len(history.get('loss')) + 1)
    # Get the number of epochs (number of bins for the x-axis of the plots) from the length of one of the metrics lists.

    #for metric in new_metrics_list:
        # Iterate over each of the metrics in 'new_metrics_list'.

    #    plt.figure(figsize=(10, 8))
        # Initialise a plot figure of size 10 by 8 inches.

    #    plt.title(metric + ' training history')
    #    plt.xlabel('Epoch')
        # Set the plot title and x-axis label.

    #    if 'loss' in metric:
    #        plt.ylabel(metric + ' - ' + model.loss)
            # If the metric is a 'loss', append also the name of the specfic loss function used by the model to the
            # y-axis label.

    #    else:
    #        plt.ylabel(metric)
            # Otherwise simply use the metric name as the y-axis label.

    #    plt.plot(x, history.get(metric))
        # Get the metric from the history dictionary and plot it.

    #    if validated:
    #        plt.plot(x, history.get('val_' + metric))
            # If the model was validated throughout training, plot the validation history for this metric as well.

    #    plt.grid(which='both')
        # Add a grid to the plot on both major and minor gridlines.

    #    plt.savefig('{}/{}_{}_plot.svg'.format(save_dir, save_prefix, metric))
        # Save the completed figure as a scalable vector graphic, including the metric name in the file name.

    #    if close:
    #        plt.close()
            # Immediately close the plots after they've been rendered and saved.

    # Keep the following here for the time being in case you messed it up with your new version:
    #
    # dataset = dataset.map(
    #
    #     map_func=lambda serialized: tf.io.parse_single_example(serialized=serialized, features={
    #
    #         'inputs':               tf.io.FixedLenFeature(shape=input_shape, dtype=tf.float32),
    #         'peak_count_oh':        tf.io.FixedLenFeature(shape=(6,), dtype=tf.int64),
    #         'peak_pos_mh':          tf.io.FixedLenFeature(shape=(pos_bins,), dtype=tf.int64),
    #         'peak_pos_oh_concat':   tf.io.FixedLenFeature(shape=(max_num_peaks*pos_bins,), dtype=tf.int64),
    #         'cs_norm':              tf.io.FixedLenFeature(shape=(max_num_peaks*6+10,), dtype=tf.float32),
    #         'cs':                   tf.io.FixedLenFeature(shape=(max_num_peaks*6+10,), dtype=tf.float32)}),
    #
    #     num_parallel_calls=4)

    # This line is deserialising the data into familiar TensorFlow data types that can be used for training. The 'map'
    # method of the dataset applies the given 'map_func' ('parse_single_example') to all the dataset's elements. We pass
    # a dictionary as the 'features' argument, whose keys correspond to the feature columns we want to retrieve from the
    # dataset and whose elements are TensorFlow 'FixedLenFeature' objects with the specified shapes and types of the
    # data we expect. The keys need to match those we used when writing to the TFRecords (in 'my_dataset_to_tfrecord').
    # However, we don't actually need to parse all the feature columns if we don't want to - it's possible to pass a
    # dictionary containing, say, only 'inputs' and 'peak_count_oh' etc. I parse everything here and then select the
    # specific features I want afterwards.
