from enum import Enum
import functools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class SessionMode(Enum):
    TRAIN = 1
    PREDICT = 2
    
def load_data(data_pattern, mode, batch_size):

    def _parse_tfexample_fn(example_proto, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        feature_to_type = {
            "ink": tf.VarLenFeature(dtype=tf.float32),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64)
        }
        if mode == SessionMode.TRAIN:
            # The labels won't be available at inference time, so don't add them
            # to the list of feature_columns to be read.
            feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto, feature_to_type)
        labels = None
        if mode == SessionMode.TRAIN:
            labels = parsed_features["class_index"]
    
        parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])
        return parsed_features, labels
    
    dataset = tf.data.TFRecordDataset.list_files(data_pattern)
    #todo: why shuffle here.
    if mode == SessionMode.TRAIN:
        dataset = dataset.shuffle(buffer_size=10)
    
    dataset = dataset.repeat()
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10, block_length=1)
    dataset = dataset.map(functools.partial(_parse_tfexample_fn, mode=mode), num_parallel_calls=10)
    #dataset = dataset.prefetch(10000)

    if mode == SessionMode.TRAIN:
        dataset = dataset.shuffle(buffer_size=1000000)
    # Our inputs are variable length, so pad them.
    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

def plot(ink):
    ink = np.reshape(ink, (-1, 3))
    ink = np.concatenate(([[0, 0, 0]], ink))
    ink = np.transpose(ink)

    for i in range(ink.shape[1] - 1):
        ink[0][i + 1] = ink[0][i + 1] + ink[0][i]
        ink[1][i + 1] = ink[1][i + 1] + ink[1][i]

    pre_i = 0
    for i in range(ink.shape[1]):
        if ink[2][i] > 0.:
            plt.plot(ink[0][pre_i:(i+1)], -ink[1][pre_i:(i+1)])
            pre_i = i+1

    plt.show()
