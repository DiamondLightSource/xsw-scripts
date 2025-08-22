import tensorflow as tf
import tensorflow.keras as keras
import pickle
from networks.NNModels import build_2D_cnn
import numpy as np
from sys import argv
from datetime import datetime

##file inputs - num_files, validation_percentage

#NUM_PEAKS = int(argv[4])
batch_size = 1000

filename_format = '{}records{}.tfrecords'
records_per_file = 10000
num_files = int(argv[1])
validation_percentage = int(argv[2])/100
filenames = [filename_format.format(records_per_file, i+1) for i in range(num_files)]

file_dir = "/work4/dls/scarf1391/MultiPeakC1sDataOneHot"
save_path = "/home/vol01/scarf1391/training/models"

file_paths = ['{}/{}'.format(file_dir, filename) for filename in filenames]
train_file_paths, val_file_paths = np.split(file_paths, [int(num_files*(1-validation_percentage))])

train_size = len(train_file_paths) * records_per_file
val_size = len(val_file_paths) * records_per_file

features_dict = {
    'xsw_curve': tf.io.FixedLenFeature(shape = (504, 200, 1), dtype = tf.float32),
    #'cs_norm': tf.io.FixedLenFeature(shape = (46, 1), dtype = tf.float32),
    'one_hot_count': tf.io.FixedLenFeature(shape = (6, 1), dtype=tf.float32)
}

train_files = tf.data.Dataset.from_tensor_slices(train_file_paths)
val_files = tf.data.Dataset.from_tensor_slices(val_file_paths)
block_length = 10
train_dataset = train_files.interleave(lambda filename: tf.data.TFRecordDataset(filename).prefetch(num_files * block_length))
val_dataset = val_files.interleave(lambda filename: tf.data.TFRecordDataset(filename))
train_dataset = train_dataset.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))
val_dataset = val_dataset.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))


train_dataset = train_dataset.map(map_func = lambda features:
                                  
                                  (features['xsw_curve'],
                                   features['one_hot_count'])

                                   , num_parallel_calls= tf.data.AUTOTUNE
                                   )
                                   

val_dataset = val_dataset.map(map_func = lambda features:
                                  (features['xsw_curve'],
                                   features['one_hot_count'])
                                   , num_parallel_calls= tf.data.AUTOTUNE
                                   )

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

train_dataset = train_dataset.repeat()
val_dataset = val_dataset.repeat()

model, hyper_params = build_2D_cnn(dense_neurons = [512, 128, 6],
                                   loss = 'categorical_crossentropy',
                                   metrics = ['accuracy'], 
                                   activations = ['relu', 'softmax'],
                                   dropout_probabilites = [0.5, 0.5])


currenttime = datetime.now()
model_suffix = '-' + str(currenttime.strftime("%d%m%H"))
model_name = '{}/MultiPeakC1sCountModel{}.keras'.format(save_path, model_suffix)
history_filename = '{}/CountTrainHistory{}.pkl'.format(save_path, model_suffix)

#tf.profiler.experimental.start('logdir')
history = model.fit(train_dataset, epochs = 10, steps_per_epoch=train_size//batch_size, validation_data = val_dataset, validation_steps = val_size//batch_size)
#tf.profiler.experimental.stop()
model.save(model_name)

with open(history_filename, 'wb') as file_pi:
  pickle.dump(history.history, file_pi)
