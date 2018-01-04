import pandas as pd
import numpy as np
import tensorflow as tf
import data_utils as utils
import argparse
import model

def train(batch_size, epoch_size, fold_size, learning_rate, ckpt, logdir, no_ia, regen_data, channels, test_size):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        print('training')
        total_x, total_y = utils.get_train_data(regen_data=regen_data, no_ia=no_ia)
        utils.shuffle_in_unison(total_x, total_y)
        train_x = total_x[test_size:]
        train_y = total_y[test_size:]
        test_x = total_x[0:test_size]
        test_y = total_y[0:test_size]

        with tf.Session() as sess:
            if ckpt is None:
                model.model(channels)
                sess.run(tf.global_variables_initializer())
                fold_start = 0
            else:
                print('loading ckpt {}'.format(ckpt))
                saver = tf.train.import_meta_graph('../models/{}.ckpt.meta'.format(ckpt))
                saver.restore(sess, '../models/{}.ckpt'.format(ckpt))
                print('ckpt loaded')
                fold_start = ckpt + 1

            graph = sess.graph
            is_training_t = graph.get_tensor_by_name('input/is_training:0')
            optimizer_t = graph.get_operation_by_name('output/optimizer')
            x_in_t = graph.get_tensor_by_name('input/x_in:0')
            y_in_t = graph.get_tensor_by_name('input/y_in:0')
            process_t = graph.get_tensor_by_name('input/process:0')
            logits_t = graph.get_tensor_by_name('output/logits:0')
            learning_rate_t = graph.get_tensor_by_name('input/learning_rate:0')
            accuracy_t = graph.get_tensor_by_name('output/accuracy:0')
            probability_t = graph.get_tensor_by_name('output/probability:0')
            cost_t = graph.get_tensor_by_name('output/cost:0')
            summ_t = tf.summary.merge_all()

            writer = tf.summary.FileWriter(logdir)
            writer.add_graph(sess.graph)
            # epoch
            if epoch_size is None:
                epoch_size = int(train_x.shape[0] / 64)
            print('start training, batch size {}, epoch size {}'.format(batch_size, epoch_size))
            for epoch in range(fold_start, fold_size):
                utils.shuffle_in_unison(train_x, train_y)
                for batch in range(epoch_size):
                    x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                    y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
                    sess.run(optimizer_t, feed_dict={x_in_t: x_batch, y_in_t: y_batch, learning_rate_t: learning_rate,
                                                     is_training_t: True, process_t: 'train step'})

                # print accuracy after each epoch
                train_acc = []
                train_cost = []
                for batch in range(epoch_size):
                    x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                    y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
                    [tmp_acc, tmp_cost, summ] = sess.run([accuracy_t, cost_t, summ_t],
                                                   feed_dict={x_in_t:x_batch, y_in_t:y_batch, is_training_t: False,
                                                              process_t: 'eval train'})
                    train_acc.append(tmp_acc)
                    train_cost.append(tmp_cost)
                writer.add_summary(summ, epoch)
                [test_acc, test_cost, summ] = sess.run([accuracy_t, cost_t, summ_t],
                                                 feed_dict={x_in_t:test_x, is_training_t: False, y_in_t:test_y,
                                                            process_t: 'eval test'})

                print('\nepoch {}'.format(epoch))
                print('train accuracy {} train cost {}'.format(np.mean(train_acc), np.mean(train_cost)))
                print('test accuracy {} test cost {}'.format(test_acc, test_cost))
                writer.add_summary(summ, epoch)
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
    parser.add_argument('--ckpt', type=int)
    parser.add_argument('--logdir', type=str, default='../logs/default')
    parser.add_argument('--no_ia', type=bool, default=False)
    parser.add_argument('--regen_data', type=bool, default=False)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--test_size', type=int, default=1000)
    args = parser.parse_args()
    print(args)
    train(args.batch_size, args.epoch_size, args.fold_size, args.learning_rate, args.ckpt, args.logdir, args.no_ia, args.regen_data, args.channels, args.test_size)
