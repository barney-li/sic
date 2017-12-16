with tf.name_scope('forward'):
    W1 = tf.get_variable("W1", [3,3,1,5], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z1 = tf.nn.conv2d(x, W1, [1,1,1,1], 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 5x5, sride 5, padding 'SAME'
    P1 = tf.nn.max_pool(A1, [1,5,5,1], [1,5,5,1], 'SAME')
    
    # CONV2D: filters W2, stride 1, padding 'SAME'
    W2 = tf.get_variable("W2", [3,3,5,10], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z2 = tf.nn.conv2d(P1, W2, [1,1,1,1], 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, [1,3,3,1], [1,3,3,1], 'SAME')
    
    # CONV2D: filters W3, stride 1, padding 'SAME'
    W3 = tf.get_variable("W3", [3,3,10,20], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z3 = tf.nn.conv2d(P2, W3, [1,1,1,1], 'SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P3 = tf.nn.max_pool(A3, [1,2,2,1], [1,2,2,1], 'SAME')
    
    # CONV2D: filters W3, stride 1, padding 'SAME'
    W4 = tf.get_variable("W4", [3,3,20,40], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z4 = tf.nn.conv2d(P3, W4, [1,1,1,1], 'SAME')
    # RELU
    A4 = tf.nn.relu(Z4)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P4 = tf.nn.max_pool(A4, [1,5,5,1], [1,5,5,1], 'SAME')
    
    # FLATTEN
    P5 = tf.contrib.layers.flatten(P4)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    
    F6 = tf.layers.dense(P5, 50, activation=tf.nn.relu)
    F7 = tf.layers.dropout(F6, rate=0.5)
    y_ = tf.contrib.layers.fully_connected(F7, 2, activation_fn=None)