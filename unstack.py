import tensorflow as tf
import numpy as np

tile_num = 3
val_num = 5
val_dim = 2

cur_type = tf.float32

with tf.device("/gpu:0"):
    vals = tf.reshape(tf.range(val_num*val_dim, dtype=cur_type), [1, val_num, val_dim])
    vals = tf.tile(vals, [tile_num, 1, 1])
    
    vals_list = tf.unstack(vals)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    outvals_list = sess.run(vals_list)
    
    print(outvals_list)

