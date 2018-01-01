import tensorflow as tf
import resnet_model
def model(channels=3):
    with tf.variable_scope('input'):
        is_training = tf.placeholder(tf.bool, name='is_training')
        x_in = tf.placeholder(tf.float32, name='x_in')
        y_in = tf.placeholder(tf.float32, name='y_in')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        process = tf.placeholder(tf.string, name='process')
        images = tf.image.resize_images(tf.reshape(x_in, [-1, 75, 75, channels]), [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        labels = tf.reshape(y_in, [-1, 2])
    with tf.variable_scope('resnet'):
        y_generator = resnet_model.imagenet_resnet_v2(18, 2, 'channels_last')
        tmp = y_generator(images, is_training)
    with tf.variable_scope('output'):
        logits = tf.identity(tmp, 'logits')
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), name='cost')
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
        probability = tf.nn.softmax(logits, name='probability')
        process = tf.Print(process, [process], message='current process: ', name='print')
        tf.summary.scalar('{} cost'.format(process), cost)
        tf.summary.scalar('{} accuracy'.format(process), accuracy)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, name='optimizer')
    return optimizer