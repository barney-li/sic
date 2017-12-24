import tensorflow as tf
import pandas as pd
import numpy as np

def format_label(labels_in):
    labels_out = tf.one_hot(labels_in, depth=2)
    with tf.Session() as sess:
        res = np.array(sess.run(labels_out))
    return res

def format_img(images_in):
    images_out = []
    cnt = 0
    for arr in images_in:
        print('formatting img {}'.format(cnt))
        cnt += 1
        max_v = arr.max()
        min_v = arr.min()
        arr = 255 * (arr - min_v) / (max_v - min_v)
        image = np.reshape(arr, (3, 75, 75))
        im_in = tf.placeholder(tf.int8, shape=[3, 75, 75])
        im_out = tf.transpose(im_in, [1, 2, 0])
        with tf.Session() as sess:
            images_out.append(sess.run(im_out, feed_dict={im_in: image}))
    return np.array(images_out)


def ia(images_in, labels_in):
    images_out = []
    labels_out = []
    for image, label in zip(images_in, labels_in):
        image_in = tf.placeholder(tf.int8, image.shape)
        up_down = tf.image.flip_up_down(image_in)
        left_right = tf.image.flip_left_right(image_in)
        rot90 = tf.image.rot90(image_in)
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