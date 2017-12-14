with tf.name_scope('forward'):
    W1 = tf.get_variable("W1", [5,5,1,5], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5,5,5,10], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z1 = tf.nn.conv2d(x, W1, [1,1,1,1], 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 5x5, sride 5, padding 'SAME'
    P1 = tf.nn.max_pool(A1, [1,5,5,1], [1,5,5,1], 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, [1,1,1,1], 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, [1,3,3,1], [1,3,3,1], 'SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    
    F3 = tf.layers.dense(P2, 50, activation=tf.nn.relu)
    F4 = tf.layers.dropout(F3, rate=0.5)
    y_ = tf.contrib.layers.fully_connected(F4, 2, activation_fn=None)