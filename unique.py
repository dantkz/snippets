import tensorflow as tf
import numpy as np

with tf.device("/gpu:0"):
    x = tf.convert_to_tensor(np.array([0,1,2,0,1,2], dtype=np.int64))
    y, idx = tf.unique(x, out_idx=tf.int64)
    ones = tf.ones(10, dtype=tf.int64)
    ones = ones[0:tf.reduce_sum(idx)+1]
    vals = tf.range(10, dtype=tf.int64)
    result = tf.scatter_nd(idx, ones, [10])
    sess = tf.Session()
    sess.run(tf.global_variable_initializer())
    print(sess.run(idx))
    print(sess.run(result))

