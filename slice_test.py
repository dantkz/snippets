import tensorflow as tf
import numpy as np

data_num = 2
data_dim = 4
a = tf.zeros([data_num, data_dim], dtype=tf.float32)+1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(a))
print(sess.run(a_sub))
