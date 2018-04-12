import data
import util_model
import tensorflow as tf
import numpy as np

def train(trained_model, trained_optimizer):
    train_data_pattern = '../tutorial_dataset/training.tfrecord-?????-of-?????'
    class_file = '../tutorial_dataset/training.tfrecord.classes'
    eval_data_pattern = '../tutorial_dataset/eval.tfrecord-?????-of-?????'

    param = util_model.Param()
    param.class_num = data.get_num_classes(class_file)
    train_inks, train_lengths, train_labels = data.load_data(train_data_pattern, data.SessionMode.TRAIN, param.batch_size)
    eval_inks, eval_lengths, eval_labels = data.load_data(eval_data_pattern, data.SessionMode.PREDICT, param.batch_size * 2)

    model = util_model.Model(param)

    saver_model = tf.train.Saver(model.model_variables)
    saver_optimizer = tf.train.Saver(model.optimizer_variables)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./nn_logs", sess.graph)
        tf.summary.scalar('cost', model.cross_entropy)
        #merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        
        if trained_model and trained_optimizer:
            saver_model.restore(sess, trained_model)
            saver_optimizer.restore(sess, trained_optimizer)
        else:
            sess.run(tf.global_variables_initializer())

        idx = 0
        while True:
            train_vinks, train_vlengths, train_vlabels = sess.run([train_inks, train_lengths, train_labels])
            vloss, vacc, _ = sess.run([model.cross_entropy, model.accuracy, model.train_op], 
                feed_dict={model.if_train:True, model.input_inks:train_vinks, model.input_lengths:train_vlengths, model.input_labels:train_vlabels })
            print(vloss, vacc)
            if (idx + 1) % 10 == 0:
                eval_vinks, eval_vlengths, eval_vlabels = sess.run([eval_inks, eval_lengths, eval_labels])
                vacc = sess.run(model.accuracy, 
                    feed_dict={model.if_train:False, model.input_inks:eval_vinks, model.input_lengths:eval_vlengths, model.input_labels:eval_vlabels }) 
                print('val acc: ', vacc)

            idx = idx + 1
            #writer.add_summary(vsummary, i)

        writer.close()
        saver_model.save(sess, 'mdl/model.ckpt')
        saver_optimizer.save(sess, 'mdl/optimizer.ckpt')

#trained_optimizer = 'mdl/optimizer.ckpt'
#trained_model = 'mdl/model.ckpt'
trained_model = None
trained_optimizer = None
train(trained_model, trained_optimizer)