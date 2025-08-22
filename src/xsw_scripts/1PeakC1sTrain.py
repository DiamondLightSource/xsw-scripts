import tensorflow as tf
import pickle

from networks.NNModels import build_cnn
from sys import argv

NUM_PEAKS = 1

num_records = int(argv[1]) ##1st input is the number of records in each file
num_files = int(argv[2]) ##2nd input is number of files
num_files_created = int(argv[3]) ##3rd input is number of files created
data_file_dir = argv[4] ##4th input is path to directory containg the tfrecord files
train_type = argv[5] ##5th input is type of training
save_path = argv[6] ##6th input is the location to save the model

filename_format = '{}records{}of{}.tfrecords'

filenames = [filename_format.format(num_records, i+1, num_files) for i in range(num_files_created)]
file_paths = [str('{}/{}'.format(data_file_dir, filename)) for filename in filenames]

#test_file_paths = file_paths[-1]
val_file_paths = file_paths[-1]
train_file_paths = file_paths[0:-1]

train_size = num_records * len(train_file_paths)
#test_size = num_records * len(test_file_paths)
val_size = num_records * len(val_file_paths)

features_dict = {
    'xsw_curve': tf.io.FixedLenFeature(shape = (200,1), dtype = tf.float32),
    'cs_norm': tf.io.FixedLenFeature(shape = (46,1), dtype = tf.float32),
    'mod_params': tf.io.FixedLenFeature(shape = (3,1), dtype = tf.float32),
    #'cs': tf.io.FixedLenFeature(shape=(46,1), dtype=tf.float32),
    #'modulation': tf.io.FixedLenFeature(shape=(?, 1), dtype=tf.float32)
}

train_files = tf.data.Dataset.from_tensor_slices(train_file_paths)
block_length = 10
train_dataset = train_files.interleave(lambda filename: tf.data.TFRecordDataset(filename).prefetch(num_files * block_length))

train_dataset = train_dataset.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))


val_dataset = tf.data.TFRecordDataset(val_file_paths)
val_dataset = val_dataset.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))

#test_dataset = tf.data.TFRecordDataset(test_file_paths)
#test_dataset = test_dataset.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))

if train_type == 'p' or train_type == 'pos':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 10)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],                             
                       tf.gather(features['cs_norm'], 10)
                      )
                    )
  val_dataset = val_dataset.batch(2500)

  #test_dataset = test_dataset.map(map_func = lambda features:
  #                    (features['xsw_curve'],
  #                     tf.gather(features['cs_norm'], 10)
  #                    )
  #                  )
  #test_dataset = test_dataset.batch(2500)

  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )

  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sPosModel.keras".format(save_path))

if train_type == 'gw' or train_type == 'gaussian':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 11)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 11)
                      )
                    )
  val_dataset = val_dataset.batch(2500)

  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )

  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sGaussWidthModel.keras".format(save_path))

if train_type == 'lw' or train_type == 'lorentzian':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 12)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 12)
                      )
                    )
  val_dataset = val_dataset.batch(2500)

  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )

  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sLorentzWidthModel.keras".format(save_path))

if train_type == 'a' or train_type == 'asymmetry':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 13)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 13)
                      )
                    )
  val_dataset = val_dataset.batch(2500)

  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )

  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sAsymModel.keras".format(save_path))

if train_type == 's' or train_type == 'step':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 14)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 14)
                      )
                    )
  val_dataset = val_dataset.batch(2500)

  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )

  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sStepModel.keras".format(save_path))

if train_type == 'i' or train_type == 'intensity':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 15)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['cs_norm'], 15)
                      )
                    )
  val_dataset = val_dataset.batch(2500)


  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )
  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sIntensityModel.keras".format(save_path))

if train_type == 'mw' or train_type == 'modwidth':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['mod_params'], 2)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['mod_params'], 2)
                      )
                    )
  val_dataset = val_dataset.batch(2500)


  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )
  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sModWidthModel.keras".format(save_path))

if train_type == 'cf' or train_type == 'cfrac':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['mod_params'], 0)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['mod_params'], 0)
                      )
                    )
  val_dataset = val_dataset.batch(2500)


  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )
  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sCohFracModel.keras".format(save_path))

if train_type == 'cp' or train_type == 'cpos':
  train_dataset = train_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['mod_params'], 1)
                      )
                    )

  train_dataset = train_dataset.shuffle(buffer_size = 100000)
  train_dataset = train_dataset.batch(2500)
  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.prefetch(1)

  val_dataset = val_dataset.map(map_func = lambda features:
                      (features['xsw_curve'],
                       tf.gather(features['mod_params'], 1)
                      )
                    )
  val_dataset = val_dataset.batch(2500)


  model, hyper_param = build_cnn(
      input_size=200,
      num_kernels=(16, 16, 16),  # 3 convolutionals for position finding.
      kernel_size=(16, 32, 48),
      conv_drop=(0, 0, 0),  # No dropout on convolutionals
      dense_units=(500, 300, NUM_PEAKS),
      dense_drop=(0.2, 0.2),  # Lighter dropout for position finding.
      loss="mae",  # Regression tasks should use mean absolute or quadratic loss.
      metrics="mse",  # Another regression metric to keep track of.
      activations=(
          "relu",
          "sigmoid",
      ),  # Sigmoid rescales outputs to (0,1), but doesn't enforce summation to unity.
      learning_rate=0.001,
      name=None,
  )
  history = model.fit(train_dataset, epochs = 10, steps_per_epoch = train_size//2500, validation_data = val_dataset)

  model.save("{}/1PeakC1sCohPosModel.keras".format(save_path))

with open('{}/{}trainHistoryDict'.format(save_path, train_type), 'wb') as file_pi:
  pickle.dump(history.history, file_pi)