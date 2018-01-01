import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os
import argparse
def format_label(labels_in):
    labels_out = tf.one_hot(labels_in, depth=2)
    with tf.Session() as sess:
        res = np.array(sess.run(labels_out))
    return res

def norm(array_in):
    array_out = []
    for arr in array_in:
        max_v = arr.max()
        min_v = arr.min()
        arr = 255 * (arr - min_v) / (max_v - min_v)
        # very very tricky, if the data type is float, plt.imshow can only print images with format (C,H,W)
        # and (H,W,C) will be print like a mess, after converting to int8 it can correctly print (H,W,C) format
        array_out.append(arr.astype(np.int8))
    return array_out

def ia(images_in, labels_in):
    images_out = []
    labels_out = []
    cnt = 0
    # put the graph in cpu, other wise it's just too slow to transmit data to gpu
    # back and forth through pci
    with tf.device('/cpu:0'):
        image_in = tf.placeholder(tf.int8, images_in[0].shape)
        up_down = tf.image.flip_up_down(image_in)
        left_right = tf.image.flip_left_right(image_in)
        rot90 = tf.image.rot90(image_in)
    for image, label in zip(images_in, labels_in):
        print('ia img {}...'.format(cnt))
        cnt += 1
        with tf.Session() as sess:
            images_out.append(image)
            labels_out.append(label)
            images_out.append(sess.run(up_down, feed_dict={image_in: image}))
            labels_out.append(label)
            images_out.append(sess.run(left_right, feed_dict={image_in: image}))
            labels_out.append(label)
            images_out.append(sess.run(rot90, feed_dict={image_in: image}))
            labels_out.append(label)
    return np.array(images_out), np.array(labels_out)

def get_test_data(path='../data/test.json', channels=3):
    test_data = pd.read_json(path)
    band1 = np.array(test_data['band_1'].tolist())
    band2 = np.array(test_data['band_2'].tolist())
    id = np.array(test_data['id'].tolist())
    images = preprocess_image(band1, band2, channels)
    return images, id


def preprocess_image(band_1, band_2, channels):
    band_1_norm = norm(band_1)
    band_2_norm = norm(band_2)
    if channels == 2:
        band_comb = np.stack((band_1_norm, band_2_norm), axis=-1)
    else:
        band_comb = np.stack((band_1_norm, band_1_norm, band_2_norm), axis=-1)
    band_comb = np.reshape(band_comb, (-1, 75, 75, channels))
    return band_comb


def get_train_data(path='../data/train.json', archive_id='', regen_data = False, no_ia = False, channels = 3):
    archive_x = '../data/train_x_{}.npy'.format(archive_id)
    archive_y = '../data/train_y_{}.npy'.format(archive_id)
    if (not regen_data) and os.path.isfile(archive_x) and os.path.isfile(archive_y):
        train_x = np.load(archive_x)
        train_y = np.load(archive_y)
    else:
        train_data = pd.read_json(path)
        c1 = np.array(train_data['band_1'].tolist())
        c2 = np.array(train_data['band_2'].tolist())
        train_y = np.array(train_data['is_iceberg'].tolist())
        train_x = preprocess_image(c1, c2, channels)
        # train_x = transpose_img(train_x)
        if not no_ia:
            raw_x = train_x.copy()
            format_x = np.reshape(raw_x, (-1, 75, 75, channels))
            print('image augmentation...')
            train_x, train_y = ia(format_x, train_y)
        print('formatting labels...')
        train_y = format_label(train_y)
        np.save('../data/train_x_{}.npy'.format(archive_id), train_x)
        np.save('../data/train_y_{}.npy'.format(archive_id), train_y)
    return train_x, train_y

if __name__ == '__main__':
    get_train_data(regen_data=True, channels=3)