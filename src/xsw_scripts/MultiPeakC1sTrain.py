import tensorflow as tf
import tensorflow.keras as keras
import pickle
from networks.NNModels import build_2D_cnn, build_2D_cnn_functional
import numpy as np
from sys import argv
from datetime import datetime

##file inputs - train_type, num_files, validation_percentage, num_peaks

train_type = argv[1]
num_peaks = int(argv[4])
batch_size = 1000
print(tf.__version__)

filename_format = '{}records{}.tfrecords'
records_per_file = 10000
num_files = int(argv[2])
validation_percentage = int(argv[3])/100
filenames = [filename_format.format(records_per_file, i+1) for i in range(num_files)]

file_dir = "/work4/dls/scarf1391/{}PeakC1sData".format(num_peaks)
save_path = "/home/vol01/scarf1391/training/models"

file_paths = ['{}/{}'.format(file_dir, filename) for filename in filenames]
train_file_paths, val_file_paths = np.split(file_paths, [int(num_files*(1-validation_percentage))])

train_size = len(train_file_paths) * records_per_file
val_size = len(val_file_paths) * records_per_file

features_dict = {
    'xsw_curve': tf.io.FixedLenFeature(shape = (504, 200, 1), dtype = tf.float32),
    'cs_norm': tf.io.FixedLenFeature(shape = (46, 1), dtype = tf.float32),
}

train_files = tf.data.Dataset.from_tensor_slices(train_file_paths)
val_files = tf.data.Dataset.from_tensor_slices(val_file_paths)
block_length = 10
train_dataset = train_files.interleave(lambda filename: tf.data.TFRecordDataset(filename).prefetch(num_files * block_length))
val_dataset = val_files.interleave(lambda filename: tf.data.TFRecordDataset(filename))
train_dataset = train_dataset.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))
val_dataset = val_dataset.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))


feature_indices = []
if train_type == "Pos":
    for i in range(num_peaks):
        feature_indices.append(10 + 6*i)
elif train_type == "Gauss":
    for i in range(num_peaks):
        feature_indices.append(11 + 6*i)
elif train_type == "Lorentz":
    for i in range(num_peaks):
        feature_indices.append(12 + 6*i)
elif train_type == "Asymm":
    for i in range(num_peaks):
        feature_indices.append(13 + 6*i)
elif train_type == "Step":
    for i in range(num_peaks):
        feature_indices.append(14 + 6*i)
elif train_type == "Intensity":
    for i in range(num_peaks):
        feature_indices.append(15 + 6*i)
elif train_type == "All":
    for i in range(num_peaks):
        feature_indices += 10 + 6*i, 11 + 6*i, 12 + 6*i, 14 + 6*i, 14 + 6*i, 15 + 6*i

train_dataset = train_dataset.map(map_func = lambda features:
                                  (features['xsw_curve'],
                                   (tf.gather(features['cs_norm'], feature_indices[0]),
                                   tf.gather(features['cs_norm'], feature_indices[1])),
                                   #tf.gather(features['cs_norm'], feature_indices[1])
                                   ), num_parallel_calls= tf.data.AUTOTUNE
                                   )
#train_dataset_y = train_dataset.map(map_func = lambda features:
#                                  (features['xsw_curve'],
#                                   tf.gather(features['cs_norm'], feature_indices[0]),
#                                   tf.gather(features['cs_norm'], feature_indices[1])
#                                   ), num_parallel_calls= tf.data.AUTOTUNE
#                                   )

val_dataset = val_dataset.map(map_func = lambda features:
                                  (features['xsw_curve'],
                                   (tf.gather(features['cs_norm'], feature_indices[0]),
                                   tf.gather(features['cs_norm'], feature_indices[1])),
                                   #tf.gather(features['cs_norm'], feature_indices[1])
                                   ), num_parallel_calls= tf.data.AUTOTUNE
                                   )

train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
val_dataset = val_dataset.batch(batch_size)

model, hyper_params = build_2D_cnn_functional(dense_neurons = [512, 128, num_peaks])
model.summary()
currenttime = datetime.now()
model_suffix = '-' + str(currenttime.strftime("%d%m%H"))
model_name = '{}/{}PeakC1s{}Model{}.keras'.format(save_path, num_peaks, train_type, model_suffix)
history_filename = '{}/{}TrainHistory{}.pkl'.format(save_path, train_type, model_suffix)

#tf.profiler.experimental.start('logdir')
history = model.fit(train_dataset, epochs = 10, steps_per_epoch=train_size//batch_size, validation_data = val_dataset, validation_steps = val_size//batch_size)
#tf.profiler.experimental.stop()
model.save(model_name)

with open(history_filename, 'wb') as file_pi:
  pickle.dump(history.history, file_pi)
