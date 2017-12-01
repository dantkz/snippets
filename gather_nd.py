import tensorflow as tf
import numpy as np

val_num = 6
val_dim = 2
neib_num = 2

with tf.device("/cpu:0"):
    val = tf.reshape(tf.range(val_num*val_dim), [val_num, val_dim])
    neib0 = tf.reshape(tf.constant([1, 3]), [-1, 1])
    neib1 = tf.reshape(tf.constant([0, 2]), [-1, 1])
    neib2 = tf.reshape(tf.constant([2, 4]), [-1, 1])

    val0 = tf.gather_nd(val, neib0)
    val1 = tf.gather_nd(val, neib1)
    val2 = tf.gather_nd(val, neib2)

    print(neib0)
    print(val)
    print(val0)

    neibs = tf.stack([neib0, neib1, neib2])
    #print(neibs)

    vals = tf.gather_nd(val, neibs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #print(sess.run(val))
    #print(sess.run(val0))
    #print(sess.run(val1))
    #print(sess.run(val2))

    print(sess.run(vals).shape)

