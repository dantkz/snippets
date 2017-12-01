import tensorflow as tf
import numpy as np

x = tf.convert_to_tensor(np.array([0,1,2,0,1,2], dtype=np.int64))
y, idx = tf.unique(x, out_idx=tf.int64)
sess = tf.Session()
print(sess.run(idx))

