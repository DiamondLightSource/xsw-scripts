import tensorflow as tf
import tensorflow.keras as keras
import pickle
import numpy as np
from sys import argv
from datetime import datetime

##file inputs - train_type, num_files, validation_percentage

print(tf.config.list_logical_devices(), tf.config.list_physical_devices(), flush=True)
train_type = argv[1]

NUM_PEAKS = 1
batch_size = 250

filename_format = '{}records{}.tfrecords'
records_per_file = 10000
num_files = int(argv[2])
validation_percentage = int(argv[3])/100
filenames = [filename_format.format(records_per_file, i+1) for i in range(num_files)]

file_dir = "/work4/dls/scarf1391/1PeakC1sData"
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
if train_type == 'Pos':
  train_dataset = train_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],
                        tf.gather(features['cs_norm'], 10)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  val_dataset = val_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],                             
                        tf.gather(features['cs_norm'], 10)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )

if train_type == 'Gauss':
  train_dataset = train_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],
                        tf.gather(features['cs_norm'], 11)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  val_dataset = val_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],                             
                        tf.gather(features['cs_norm'], 11)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )

if train_type == 'Lorentz':
  train_dataset = train_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],
                        tf.gather(features['cs_norm'], 12)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  val_dataset = val_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],                             
                        tf.gather(features['cs_norm'], 12)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  
if train_type == 'Asymm':
  train_dataset = train_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],
                        tf.gather(features['cs_norm'], 13)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  val_dataset = val_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],                             
                        tf.gather(features['cs_norm'], 13)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  
if train_type == 'Step':
  train_dataset = train_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],
                        tf.gather(features['cs_norm'], 14)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  val_dataset = val_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],                             
                        tf.gather(features['cs_norm'], 14)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )

if train_type == 'Intensity':
  train_dataset = train_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],
                        tf.gather(features['cs_norm'], 15)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )
  val_dataset = val_dataset.map(map_func = lambda features:
                        (features['xsw_curve'],                             
                        tf.gather(features['cs_norm'], 15)
                        ), num_parallel_calls=tf.data.AUTOTUNE
                      )

train_dataset = train_dataset.shuffle(10000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
train_dataset = train_dataset.repeat()

val_dataset = val_dataset.shuffle(10000)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.repeat()

model = keras.Sequential()
model.add(keras.Input(dtype='float32', shape = (504, 200, 1))) #(batch_size, height, width, channels)
model.add(keras.layers.Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation='relu', strides = (2, 2)))
#model.add(keras.layers.MaxPooling2D(pool_size=2, padding = 'same'))
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation='relu', strides = (2, 2)))
#model.add(keras.layers.MaxPooling2D(pool_size=2, padding = 'same'))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (5,5), padding = 'same', activation='relu', strides = (2, 2)))
#model.add(keras.layers.MaxPooling2D(pool_size=2, padding = 'same'))
#model.add(keras.layers.Conv2D(filters = 80, kernel_size = (5,5), padding = 'same', activation='relu', strides = (2, 2)))
#model.add(keras.layers.MaxPooling2D(pool_size=2, padding = 'same'))
#model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation='relu'))
##model.add(keras.layers.MaxPooling2D(pool_size=2, padding = 'same'))
#model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation='relu'))
#model.add(keras.layers.MaxPooling2D(pool_size=2, padding = 'same'))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(512, activation='relu'))
#model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
              loss = 'mae',
              metrics = ['mse'])

model.summary()


currenttime = datetime.now()
model_suffix = str(currenttime.day) + '-' + str(currenttime.strftime("%H:%M:%S"))
model_name = '{}/1PeakC1s{}Model{}.keras'.format(save_path, train_type, model_suffix)
history_filename = '{}/{}TrainHistory{}'.format(save_path, train_type, model_suffix)


#tf.profiler.experimental.start('logdir')
history = model.fit(train_dataset, epochs = 10, steps_per_epoch=train_size//batch_size, validation_data = val_dataset, validation_steps = val_size//batch_size)
#tf.profiler.experimental.stop()
model.save(model_name)

with open(history_filename, 'wb') as file_pi:
  pickle.dump(history.history, file_pi)