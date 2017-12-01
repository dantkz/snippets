import tensorflow as tf
import numpy as np

with tf.device("/gpu:0"):
    #x = tf.convert_to_tensor(np.array([0,1,2,0,1,2], dtype=np.int64))
    x = tf.convert_to_tensor(np.array([0,1,2,0,1,2], dtype=np.float32))
    y, idx = tf.unique(x, out_idx=tf.int64)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    print(sess.run(idx))

