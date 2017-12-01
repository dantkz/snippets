import tensorflow as tf
import numpy as np

val_num = 5
val_dim = 2

mytypes = [(tf.int32, np.int32), (tf.int64, np.int64)]

with tf.device("/gpu:0"):

    indices = []
    updates = []
    gather_res = []
    gather_nd_res = []
    for tftype, nptype in mytypes:
        print(tftype, nptype)
        indices.append(tf.reshape(tf.range(val_num, dtype=tftype), [-1, 1]))
        updates.append(tf.constant(np.tile(np.expand_dims(np.arange(val_num, dtype=nptype), 1), [1, val_dim])))

        gather_res.append(tf.gather(updates[-1], tf.reshape(indices[-1], [-1])))
        gather_nd_res.append(tf.gather_nd(updates[-1], indices[-1]))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(len(mytypes)):
        print(sess.run(gather_res[i]))
        print(sess.run(gather_nd_res[i]))

