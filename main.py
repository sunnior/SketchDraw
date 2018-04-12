import data
import model
import tensorflow as tf

train_data_pattern = '../tutorial_dataset/training.tfrecord-?????-of-?????'
class_file = '../tutorial_dataset/training.tfrecord.classes'
param = model.Param()
param.class_num = data.get_num_classes(class_file)
features, labels = data.load_data(train_data_pattern, data.SessionMode.TRAIN, param.batch_size)
model = model.Model(features, labels, param)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./nn_logs", sess.graph)
    tf.summary.scalar('cost', model.cross_entropy)
    merged = tf.summary.merge_all()
    
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        vsummary, vloss, _ = sess.run([merged, model.cross_entropy, model.train_op], feed_dict={model.if_train:True})
        print(vloss)
        writer.add_summary(vsummary, i)

    writer.close()

