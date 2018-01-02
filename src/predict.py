import tensorflow as tf
import data_utils as utils
import numpy as np
import pandas as pd
import argparse


def predict(ckpt, model_path='../models', input_path='../data/test.json', output_path='../data/submission.csv', batch_size=64, channels=3):
    test_data, data_id = utils.get_test_data(input_path, channels)
    input_size = test_data.shape[0]
    predict_value = None
    with tf.Session() as sess:
        print('loading ckpt {}'.format(ckpt))
        saver = tf.train.import_meta_graph('{}/{}.ckpt.meta'.format(model_path, ckpt))
        saver.restore(sess, '{}/{}.ckpt'.format(model_path, ckpt))
        print('ckpt loaded')
        graph = sess.graph
        is_training_t = graph.get_tensor_by_name('input/is_training:0')
        x_in_t = graph.get_tensor_by_name('input/x_in:0')
        probability_t = graph.get_tensor_by_name('output/probability:0')

        for batch in range(int(input_size / batch_size) + 1):
            print('inference batch {} '.format(batch))
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size
            if batch_start + batch_size >= input_size:
                input_batch = test_data[batch_start:]
            else:
                input_batch = test_data[batch_start:batch_end]
            probability = sess.run(probability_t, feed_dict={x_in_t: input_batch, is_training_t: False})
            if predict_value is None:
                predict_value = probability
            else:
                predict_value = np.concatenate((predict_value, probability))
        submit = pd.DataFrame()
        submit['id'] = data_id
        submit['is_iceberg'] = round(predict_value[:, 1], 6)
        submit.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference probability')
    parser.add_argument('--ckpt', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--input_path', type=str, default='../data/test.json')
    parser.add_argument('--output_path', type=str, default='../data/submission.csv')
    args = parser.parse_args()
    predict(args.ckpt, args.model_path, args.input_path, args.output_path, args.batch_size, args.channels)
