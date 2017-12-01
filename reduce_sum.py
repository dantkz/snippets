import tensorflow as tf
import numpy as np

a = tf.reshape(tf.range(100, dtype=tf.int64), [10, 10])
print(a)

a_sum = tf.reduce_sum(a, axis=1, keep_dims=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(a_sum))

