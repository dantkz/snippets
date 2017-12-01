import tensorflow as tf
import numpy as np

val_num = 5
val_dim = 2

with tf.device("/gpu:0"):
    indices = tf.reshape(tf.range(val_num, dtype=tf.int32), [-1, 1])
    updates = tf.constant(np.tile(np.expand_dims(np.arange(val_num, dtype=np.int32), 1), [1, val_dim]))


    gather_res = tf.gather(updates, tf.reshape(indices,[-1]))
    gather_nd_res = tf.gather_nd(updates, indices)
    max_res = tf.reduce_max(indices)

    scatter_nd_res = tf.scatter_nd(indices, updates, [val_num, val_dim])
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    print(sess.run(gather_res))
    print(sess.run(gather_nd_res))
    print(sess.run(max_res))
    print(sess.run(scatter_nd_res))

