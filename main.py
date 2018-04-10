import data
import tensorflow as tf

train_data_pattern = '../tutorial_dataset/training.tfrecord-?????-of-?????'
train_batch_size = 4

features, labels = data.load_data(train_data_pattern, data.SessionMode.TRAIN, train_batch_size)

with tf.Session() as sess:
    vfeature, vlabel = sess.run([features, labels])
    print(vlabel)
    data.plot(vfeature['ink'][0])
    data.plot(vfeature['ink'][1])
    data.plot(vfeature['ink'][2])