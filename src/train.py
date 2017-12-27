import pandas as pd
import numpy as np
import tensorflow as tf
import resnet_model
import data_utils as utils
import argparse
import random

# def get_sic_training_data(path='../data/train.json'):
#     train_data = pd.read_json(path)
#     return np.array(train_data['band_1'].tolist()), \
#            np.array(train_data['band_2'].tolist()), \
#            np.array(train_data['is_iceberg'].tolist())


def model():
    with tf.name_scope('reshape'):
        trn_acc_labels_in = tf.placeholder(tf.float32, name='trn_acc_labels_in')
        trn_acc_logits_in = tf.placeholder(tf.float32, name='trn_acc_logits_in')
        tst_acc_labels_in = tf.placeholder(tf.float32, name='tst_acc_labels_in')
        tst_acc_logits_in = tf.placeholder(tf.float32, name='tst_acc_logits_in')
        x_in = tf.placeholder(tf.float32, name='x_in')
        y_in = tf.placeholder(tf.float32, name='y_in')
        learning_rate_in = tf.placeholder(tf.float32, name='learning_rate_in')
        x = tf.image.resize_images(tf.reshape(x_in, [-1, 75, 75, 2]), [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        y = tf.reshape(y_in, [-1, 2])
    with tf.name_scope('resnet'):
        y_generator = resnet_model.imagenet_resnet_v2(18, 2)
        y_ = y_generator(x, True)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    with tf.name_scope('trn_cost'):
        trn_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=trn_acc_logits_in, labels=trn_acc_labels_in))
        tf.summary.scalar('trn_cost', trn_cost)
    with tf.name_scope('tst_cost'):
        tst_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tst_acc_logits_in, labels=tst_acc_labels_in))
        tf.summary.scalar('tst_cost', tst_cost)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    with tf.name_scope('trn_acc'):
        trn_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(trn_acc_labels_in, 1), tf.argmax(trn_acc_logits_in, 1)), tf.float32))
        tf.summary.scalar("trn_acc", trn_acc)
    with tf.name_scope('tst_acc'):
        tst_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tst_acc_labels_in, 1), tf.argmax(tst_acc_logits_in, 1)), tf.float32))
        tf.summary.scalar("tst_acc", tst_acc)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate_in).minimize(cost)
    summ = tf.summary.merge_all()
    ret = {'x_in': x_in, 'y_in': y_in, 'y_': y_, 'cost': cost, 'accuracy': accuracy, 'optimizer': optimizer,
           'learning_rate_in': learning_rate_in, 'trn_acc_labels_in': trn_acc_labels_in,
           'trn_acc_logits_in': trn_acc_logits_in, 'trn_acc': trn_acc,
           'tst_acc_labels_in': tst_acc_labels_in, 'tst_acc_logits_in': tst_acc_logits_in,
           'tst_acc': tst_acc, 'trn_cost': trn_cost, 'tst_cost': tst_cost, 'summ': summ}
    return ret


# def ia_simple(start_index, end_index, output):
#     print('getting training data...')
#     c1, c2, y = get_sic_training_data()
#     c1 = c1[start_index:end_index]
#     c2 = c2[start_index:end_index]
#     y = y[start_index:end_index]
#     print('concatenate channels...')
#     c12 = np.concatenate((c1, c2), axis=1)
#     print('one-hot labels...')
#     y_onehot = utils.format_label(y)
#     np.save('../data/train_x_{}.npy'.format(output), c12)
#     np.save('../data/train_y_{}.npy'.format(output), y_onehot)
#     return c12, y_onehot


# def ia(start_index, end_index, output):
#     print('getting training data...')
#     c1, c2, y = get_sic_training_data()
#     c1 = c1[start_index:end_index]
#     c2 = c2[start_index:end_index]
#     y = y[start_index:end_index]
#     print('concatenate channels...')
#     c12 = np.concatenate((c1, c2), axis=1)
#     print('formatting images...')
#     fmt_img = utils.format_img(c12)
#     print('formatting labels...')
#     fmt_lb = utils.format_label(y)
#     print('image augmentation...')
#     train_x, train_y = utils.ia(fmt_img, fmt_lb)
#     np.save('../data/train_x_{}.npy'.format(output), train_x)
#     np.save('../data/train_y_{}.npy'.format(output), train_y)
#     return train_x, train_y


# def get_ia(ia_output):
#     x = np.load('../data/train_x_{}.npy'.format(ia_output))
#     y = np.load('../data/train_y_{}.npy'.format(ia_output))
#     return x[200:], y[200:], x[0:200], y[0:200]


def train(batch_size, epoch_size, fold_size, learning_rate, ckpt, logdir, no_ia, regen_data):
    if ckpt is None:
        print('training')
        md = model()
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
                sess.run(optimizer, feed_dict={x_in: x_batch, y_in: y_batch, learning_rate_in: learning_rate})
            # print accuracy after each epoch
            trn_acc_labels = None
            trn_acc_logits = None
            for batch in range(epoch_size):
                x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
                res = y_.eval(session=sess, feed_dict={x_in:x_batch})
                if trn_acc_labels is None:
                    trn_acc_labels = y_batch
                    trn_acc_logits = res
                else:
                    np.concatenate((trn_acc_labels, y_batch))
                    np.concatenate((trn_acc_logits, res))
            tst_acc_logits = y_.eval(session=sess, feed_dict={x_in:test_x})
            [trn_acc_out, tst_acc_out, trn_cost_out, tst_cost_out, summ_out] = sess.run([trn_acc, tst_acc, trn_cost, tst_cost, summ],
                                                            feed_dict={trn_acc_labels_in:trn_acc_labels,
                                                                       trn_acc_logits_in:trn_acc_logits,
                                                                       tst_acc_labels_in:test_y,
                                                                       tst_acc_logits_in:tst_acc_logits})
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
