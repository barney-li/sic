import tensorflow as tf
import data_utils as utils
import numpy as np
import pandas as pd

def predict(model_path, ckpt, input_path = '../data/test.json', output_path = '../data/submission.csv', batch_size = 64):
    input, id = utils.get_test_data(input_path)
    input_size = input.shape[0]
    predict_value = None
    with tf.Session() as sess:
        print('loading ckpt {}'.format(ckpt))
        saver = tf.train.import_meta_graph('../models/{}.ckpt.meta'.format(ckpt))
        saver.restore(sess, '../models/{}.ckpt'.format(ckpt))
        print('ckpt loaded')
        y_ = tf.get_collection('y_')[0]
        x_in = tf.get_collection('x_in')[0]
        for batch in range(int(input_size/batch_size)):
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size
            input_batch = input[batch_start:batch_end]
            res = sess.run(y_, feed_dict={x_in : input_batch})
            if predict_value is None:
                predict_value = res
            else:
                predict_value = np.concatenate((predict_value, res))
        submit = pd.DataFrame()
        submit['id'] = id
        submit['is_iceberg'] = predict_value
        submit.to_csv(output_path)
