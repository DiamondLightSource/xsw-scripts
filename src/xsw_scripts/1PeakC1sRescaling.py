import tensorflow as tf
from xps_datagen.NNDataBuilderPeter import minmax_scale

unscaled = tf.data.TFRecordDataset("C:/Users/kxl61517/Documents/xsw_data/1PeakC1s/1PeakC1s100000recordsUNSCALED.tfrecords")

features_dict = {
    'xsw_curve': tf.io.FixedLenFeature(shape = (200,1), dtype = tf.float32),
    'cs_norm': tf.io.FixedLenFeature(shape = (46,1), dtype = tf.float32),
    'mod_params': tf.io.FixedLenFeature(shape = (3,1), dtype = tf.float32),
    'cs': tf.io.FixedLenFeature(shape=(46,1), dtype=tf.float32),
    'modulation': tf.io.FixedLenFeature(shape=(2171, 1), dtype=tf.float32)
}

unscaled = unscaled.map(map_func = lambda serialized: tf.io.parse_single_example(serialized=serialized, features = features_dict))
#unscaled = unscaled.map(map_func = lambda features:
#                      (features['xsw_curve'],
#                       tf.gather(features['cs_norm'], 10)
#                      )
#                    ) 

iterator = unscaled.as_numpy_iterator()

#scaled_datastet = {
#    'xsw_curve': [],
#    
#}

record = next(iterator)
record['xsw_curve'] = minmax_scale(record['xsw_curve'])

#try:
#    while True:
#        record = next(iterator)
#        curve = minmax_scale(curve)
#except tf.errors.OutOfRangeError:
#    pass