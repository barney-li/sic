import tensorflow as tf
import resnet_model
def model():
    with tf.variable_scope('input'):
        is_training = tf.placeholder(tf.bool, name='is_training')
        trn_acc_labels_in = tf.placeholder(tf.float32, name='trn_acc_labels_in')
        trn_acc_logits_in = tf.placeholder(tf.float32, name='trn_acc_logits_in')
        tst_acc_labels_in = tf.placeholder(tf.float32, name='tst_acc_labels_in')
        tst_acc_logits_in = tf.placeholder(tf.float32, name='tst_acc_logits_in')
        x_in = tf.placeholder(tf.float32, name='x_in')
        y_in = tf.placeholder(tf.float32, name='y_in')
        learning_rate_in = tf.placeholder(tf.float32, name='learning_rate_in')
    with tf.name_scope('reshape'):
        x = tf.image.resize_images(tf.reshape(x_in, [-1, 75, 75, 2]), [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        y = tf.reshape(y_in, [-1, 2])
    with tf.name_scope('resnet'):
        y_generator = resnet_model.imagenet_resnet_v2(18, 2, 'channels_last')
        y_ = y_generator(x, is_training)
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
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate_in).minimize(cost)
    summ = tf.summary.merge_all()
    ret = {'x_in': x_in, 'y_in': y_in, 'y_': y_, 'cost': cost, 'accuracy': accuracy, 'optimizer': optimizer,
           'learning_rate_in': learning_rate_in, 'trn_acc_labels_in': trn_acc_labels_in,
           'trn_acc_logits_in': trn_acc_logits_in, 'trn_acc': trn_acc,
           'tst_acc_labels_in': tst_acc_labels_in, 'tst_acc_logits_in': tst_acc_logits_in,
           'tst_acc': tst_acc, 'trn_cost': trn_cost, 'tst_cost': tst_cost, 'summ': summ,
           'is_training': is_training}
    return ret