import pandas as pd
import numpy as np
import tensorflow as tf
import resnet_model
import data_augment_utils as utils
import argparse
import os
import math


def get_sic_training_data(path='../data/train.json'):
    train_data = pd.read_json(path)
    return np.array(train_data['band_1'].tolist()), \
           np.array(train_data['band_2'].tolist()), \
           np.array(train_data['is_iceberg'].tolist())


def model():
    with tf.name_scope('reshape'):
        x_in = tf.placeholder(tf.float32)
        y_in = tf.placeholder(tf.float32)
        learning_rate_in = tf.placeholder(tf.float32)
        x = tf.image.resize_image_with_crop_or_pad(tf.reshape(x_in, [-1, 75, 75, 3]), 256, 256)
        y = tf.reshape(y_in, [-1, 2])
    with tf.name_scope('resnet'):
        y_generator = resnet_model.imagenet_resnet_v2(18, 2, 'channels_last')
        y_ = y_generator(x, True)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate_in).minimize(cost)
    return x_in, y_in, y_, cost, accuracy, optimizer, learning_rate_in


def ia(start_index, end_index, output):
    print('getting training data...')
    c1, c2, y = get_sic_training_data()
    c1 = c1[start_index:end_index]
    c2 = c2[start_index:end_index]
    y = y[start_index:end_index]
    print('concatenate channels...')
    c12 = np.concatenate((c1, c1, c2), axis=1)
    print('formatting images...')
    fmt_img = utils.format_img(c12)
    print('formatting labels...')
    fmt_lb = utils.format_label(y)
    print('image augmentation...')
    train_x, train_y = utils.ia(fmt_img, fmt_lb)
    np.save('../data/train_x_{}.npy'.format(output), train_x)
    np.save('../data/train_y_{}.npy'.format(output), train_y)
    return train_x, train_y


def get_ia(ia_output):
    x = np.load('../data/train_x_{}.npy'.format(ia_output))
    y = np.load('../data/train_y_{}.npy'.format(ia_output))
    return x[200:], y[200:], x[0:200], y[0:200]


def combine_ia(output_list):
    x = None
    y = None
    for i in output_list:
        x1 = np.load('../data/train_x_{}.npy'.format(i))
        y1 = np.load('../data/train_y_{}.npy'.format(i))
        if x is not None:
            x = np.concatenate((x, x1))
            y = np.concatenate((y, y1))
        else:
            x = x1
            y = y1
    np.save('../data/train_x.npy', x)
    np.save('../data/train_y.npy', y)


def train(ia_output, batch_size, epoch_size, fold_size, learning_rate, ckpt):
    if ckpt is None:
        print('training')
        x_in, y_in, y_, cost, accuracy, optimizer, learning_rate_in = model()
    else:
        print('training from ckpt {}'.format(ckpt))

    train_x, train_y, test_x, test_y = get_ia(ia_output)

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

        # epoch
        if epoch_size is None:
            epoch_size = int(train_x.shape[0] / 64)
        print('start training, batch size {}, epoch size {}'.format(batch_size, epoch_size))
        for epoch in range(fold_size):
            for batch in range(epoch_size):
                x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
                sess.run(optimizer, feed_dict={x_in: x_batch, y_in: y_batch, learning_rate_in: learning_rate})
                # print('epoch {} batch {}'.format(i, j))
            # print accuracy after each epoch
            train_accuracy = 0
            for batch in range(epoch_size):
                x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
                train_accuracy += accuracy.eval(session=sess,
                                               feed_dict={
                                                   x_in: x_batch, y_in: y_batch
                                               })
            train_accuracy /= int(epoch_size)
            print('epoch {}, training accuracy {}'.format(epoch, train_accuracy))
            print('test accuracy {}'.format(accuracy.eval(session=sess, feed_dict={x_in: test_x, y_in: test_y})))
            saver = tf.train.Saver()
            print('saving ckpt...')
            saved_path = saver.save(sess, '../models/{}.ckpt'.format(epoch))
            print('ckpt {} saved'.format(saved_path))


def keep_train(ckpt, ia_output):
    print('keep training...')
    train_x, train_y, test_x, test_y = get_ia(ia_output)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../models/{}.ckpt.meta'.format(ckpt))
        saver.restore(sess, '../models/{}.ckpt'.format(ckpt))
        accuracy = tf.get_collection('accuracy')[0]
        x_in = tf.get_collection('x_in')[0]
        y_in = tf.get_collection('y_in')[0]
        cost = tf.get_collection('cost')[0]
        optimizer = tf.get_collection('optimizer')[0]
        # epoch
        batch_size = 64
        for i in range(ckpt+1, 1001):
            for j in range(int(train_x.shape[0] / 64)):
                x_batch = train_x[j * batch_size: (j + 1) * batch_size]
                y_batch = train_y[j * batch_size: (j + 1) * batch_size]
                sess.run(optimizer, feed_dict={x_in: x_batch, y_in: y_batch})
            # print accuracy after each epoch
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={
                                               x_in: x_batch, y_in: y_batch
                                           })
            print('epoch {}, training accuracy {}'.format(i, train_accuracy))
            print('test accuracy {}'.format(accuracy.eval(session=sess, feed_dict={x_in: test_x, y_in: test_y})))
            saver = tf.train.Saver()
            saved_path = saver.save(sess, '../models/{}.ckpt'.format(i))
            print('model saved to {}'.format(saved_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train cnn for sic problem')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'ia', 'combine_ia'], default='train')
    parser.add_argument('--ia_start', type=int)
    parser.add_argument('--ia_end', type=int)
    parser.add_argument('--ia_output', type=str, default='')
    parser.add_argument('-l', '--ia_output_list', nargs='+', type=str, default='')
    parser.add_argument('--epoch_size', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fold_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.ia_output, args.batch_size, args.epoch_size, args.fold_size, args.learning_rate, args.ckpt)
    elif args.mode == 'ia':
        ia(args.ia_start, args.ia_end, args.ia_output)
    elif args.mode == 'combine_ia':
        print(args.ia_output_list)
        combine_ia(args.ia_output_list)