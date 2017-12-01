import tensorflow as tf
import numpy as np

val_num = 5
val_dim = 2

with tf.device("/gpu:0"):
    params = tf.constant([[0, 1, 2], [3, 4, 5]], dtype=tf.float32)
    indices = tf.constant([[7]])
    print(params.shape[0])
    gather_res = tf.gather(params, indices, axis=1)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
    sess.run(tf.global_variables_initializer())
    
    print(sess.run(gather_res))


