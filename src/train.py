import pandas as pd
import numpy as np
import tensorflow as tf
import resnet_model
import data_augment_utils as utils
import os


def get_sic_training_data(path='../data/train.json'):
    train_data = pd.read_json(path)
    return np.array(train_data['band_1'].tolist()), \
           np.array(train_data['band_2'].tolist()), \
           np.array(train_data['is_iceberg'].tolist())

def model():
    with tf.name_scope('reshape'):
        x_in = tf.placeholder(tf.float32)
        y_in = tf.placeholder(tf.float32)
        x = tf.image.resize_image_with_crop_or_pad(tf.reshape(x_in, [-1, 75, 75, 3]), 256, 256)
        y = tf.reshape(y_in, [-1, 2])
    with tf.name_scope('resnet'):
        y_generator = resnet_model.imagenet_resnet_v2(18, 2, 'channels_last')
        y_ = y_generator(x, True)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    return x_in, y_in, y_, cost, accuracy, optimizer


def get_format_train_data():
    print('getting training data...')
    c1, c2, y = get_sic_training_data()
    print('concatenate channels...')
    c12 = np.concatenate((c1, c1, c2), axis=1)
    print('formatting images...')
    fmt_img = utils.format_img(c12)
    print('formatting labels...')
    fmt_lb = utils.format_label(y)
    print('image augmentation...')
    total_x, total_y = utils.ia(fmt_img, fmt_lb)
    np.save('../data/train_x.npy', total_x)
    np.save('../data/train_y.npy', total_y)
    test_x = total_x[0:200]
    test_y = total_y[0:200]
    train_x = total_x[200:]
    train_y = total_y[200:]
    return train_x, train_y, test_x, test_y


def train():
    print('training')
    train_x, train_y, test_x, test_y = get_format_train_data()
    x_in, y_in, y_, cost, accuracy, optimizer = model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.add_to_collection('accuracy', accuracy)
        tf.add_to_collection('y_', y_)
        tf.add_to_collection('x_in', x_in)
        tf.add_to_collection('y_in', y_in)
        # epoch
        batch_size = 64
        for i in range(1001):
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

def keep_train(ckpt):
    train_x, train_y, test_x, test_y = get_format_train_data()
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
    train()