import pandas as pd
import numpy as np
import tensorflow as tf
import data_utils as utils
import argparse
import model

def train(batch_size, epoch_size, fold_size, learning_rate, ckpt, logdir, no_ia, regen_data):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        if ckpt is None:
            print('training')
            md = model.model()
            x_in = md['x_in']
            y_in = md['y_in']
            y_ = md['y_']
            cost = md['cost']
            accuracy = md['accuracy']
            optimizer = md['optimizer']
            learning_rate_in = md['learning_rate_in']
            trn_acc_labels_in = md['trn_acc_labels_in']
            trn_acc_logits_in = md['trn_acc_logits_in']
            trn_acc = md['trn_acc']
            tst_acc_labels_in = md['tst_acc_labels_in']
            tst_acc_logits_in = md['tst_acc_logits_in']
            tst_acc = md['tst_acc']
            trn_cost = md['trn_cost']
            tst_cost = md['tst_cost']
            summ = md['summ']
            is_training = md['is_training']
        else:
            print('training from ckpt {}'.format(ckpt))

        total_x, total_y = utils.get_train_data(regen_data=regen_data, no_ia=no_ia)
        train_x = total_x[200:]
        train_y = total_y[200:]
        test_x = total_x[0:200]
        test_y = total_y[0:200]

        with tf.Session() as sess:
            if ckpt is None:
                sess.run(tf.global_variables_initializer())
                tf.add_to_collection('cost', cost)
                tf.add_to_collection('y_', y_)
                tf.add_to_collection('x_in', x_in)
                tf.add_to_collection('y_in', y_in)
                tf.add_to_collection('optimizer', optimizer)
                tf.add_to_collection('learning_rate_in', learning_rate_in)
                tf.add_to_collection('accuracy', accuracy)
                tf.add_to_collection('trn_acc', trn_acc)
                tf.add_to_collection('trn_acc_labels_in', trn_acc_labels_in)
                tf.add_to_collection('trn_acc_logits_in', trn_acc_logits_in)
                tf.add_to_collection('tst_acc', tst_acc)
                tf.add_to_collection('tst_acc_labels_in', tst_acc_labels_in)
                tf.add_to_collection('tst_acc_logits_in', tst_acc_logits_in)
                tf.add_to_collection('trn_cost', trn_cost)
                tf.add_to_collection('tst_cost', tst_cost)
                tf.add_to_collection('is_training', is_training)

            else:
                print('loading ckpt {}'.format(ckpt))
                saver = tf.train.import_meta_graph('../models/{}.ckpt.meta'.format(ckpt))
                saver.restore(sess, '../models/{}.ckpt'.format(ckpt))
                print('ckpt loaded')
                cost = tf.get_collection('cost')[0]
                y_ = tf.get_collection('y_')[0]
                x_in = tf.get_collection('x_in')[0]
                y_in = tf.get_collection('y_in')[0]
                optimizer = tf.get_collection('optimizer')[0]
                learning_rate_in = tf.get_collection('learning_rate_in')[0]
                accuracy = tf.get_collection('accuracy')[0]
                trn_acc = tf.get_collection('trn_acc')[0]
                trn_acc_labels_in = tf.get_collection('trn_acc_labels_in')[0]
                trn_acc_logits_in = tf.get_collection('trn_acc_logits_in')[0]
                tst_acc = tf.get_collection('tst_acc')[0]
                tst_acc_labels_in = tf.get_collection('tst_acc_labels_in')[0]
                tst_acc_logits_in = tf.get_collection('tst_acc_logits_in')[0]
                trn_cost = tf.get_collection('trn_cost')[0]
                tst_cost = tf.get_collection('tst_cost')[0]
                is_training = tf.get_collection('is_training')[0]

            graph = sess.graph
            is_training_tensor = graph.get_tensor_by_name('input/is_training:0')

            writer = tf.summary.FileWriter(logdir)
            writer.add_graph(sess.graph)
            # epoch
            if epoch_size is None:
                epoch_size = int(train_x.shape[0] / 64)
            print('start training, batch size {}, epoch size {}'.format(batch_size, epoch_size))
            for epoch in range(fold_size):
                for batch in range(epoch_size):
                    x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                    y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
                    sess.run(optimizer, feed_dict={x_in: x_batch, y_in: y_batch, learning_rate_in: learning_rate, is_training: False})

                # print accuracy after each epoch
                trn_acc_labels = None
                trn_acc_logits = None
                for batch in range(epoch_size):
                    x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                    y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
                    res = y_.eval(session=sess, feed_dict={x_in:x_batch, is_training: False})
                    if trn_acc_labels is None:
                        trn_acc_labels = y_batch
                        trn_acc_logits = res
                    else:
                        np.concatenate((trn_acc_labels, y_batch))
                        np.concatenate((trn_acc_logits, res))
                tst_acc_logits = y_.eval(session=sess, feed_dict={x_in:test_x, is_training: False})
                [trn_acc_out, tst_acc_out, trn_cost_out, tst_cost_out, summ_out] = sess.run([trn_acc, tst_acc, trn_cost, tst_cost, summ],
                                                                feed_dict={trn_acc_labels_in:trn_acc_labels,
                                                                           trn_acc_logits_in:trn_acc_logits,
                                                                           tst_acc_labels_in:test_y,
                                                                           tst_acc_logits_in:tst_acc_logits,
                                                                           is_training: False})
                print('\nepoch {}'.format(epoch))
                print('train accuracy {} train cost {}'.format(trn_acc_out, trn_cost_out))
                print('test accuracy {} test cost {}'.format(tst_acc_out, tst_cost_out))
                writer.add_summary(summ_out, epoch)
                saver = tf.train.Saver()
                print('saving ckpt...')
                saved_path = saver.save(sess, '../models/{}.ckpt'.format(epoch))
                print('ckpt {} saved'.format(saved_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train cnn for sic problem')
    parser.add_argument('--epoch_size', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fold_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--logdir', type=str, default='../logs/default')
    parser.add_argument('--no_ia', type=bool, default=False)
    parser.add_argument('--regen_data', type=bool, default=False)
    args = parser.parse_args()
    train(args.batch_size, args.epoch_size, args.fold_size, args.learning_rate, args.ckpt, args.logdir, args.no_ia, args.regen_data)
