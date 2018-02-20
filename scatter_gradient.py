import tensorflow as tf
import numpy as np

key_num = 2**4
key_dim = 4

tf_trg_i = tf.placeholder(dtype=tf.float32, shape=[3])
idx = tf.ones([key_num], dtype=tf.int64)

indices = key_dim*tf.tile(tf.expand_dims(idx, 1), [1, key_dim])
offset = tf.reshape(tf.range(key_dim, dtype=tf.int64), [1, -1])
indices += offset

keys = tf.tile(tf.expand_dims(tf.random_shuffle(tf.range(key_num, dtype=tf.int64)), 1), [1, key_dim])

indices = tf.reshape(indices, [-1, 1])
keys = tf.reshape(keys, [-1])

ukeys = tf.scatter_nd(indices, keys, [key_num*key_dim])
ukeys = tf.reshape(ukeys, [key_num, key_dim])


ukeys_sum = tf.reduce_sum(ukeys) 

grad = tf.gradients(ukeys_sum, tf_trg_i)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(grad)
print(ukeys_sum)
_, result = sess.run([grad, ukeys_sum], feed_dict={tf_trg_i: np.array([0., 1., 2.])})

