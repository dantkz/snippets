import tensorflow as tf
import numpy as np

ref = tf.Variable(np.zeros([5], dtype=np.int32))
indices = tf.constant([0, 1, 0, 1, 0])
updates = tf.constant([1, 2, 3, 4, 5])
add = tf.scatter_nd_add(ref, indices, updates)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(add))
