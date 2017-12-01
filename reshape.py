import tensorflow as tf
import numpy as np

batch_size = 2
h = 3
w = 3
chn = 1
vals_r = 0*tf.ones([batch_size, h, w, 1, chn], dtype=tf.float32)
vals_i = 1*tf.ones([batch_size, h, w, 1, chn], dtype=tf.float32)
vals = tf.concat([vals_r, vals_i], axis=3)
print(vals)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run(tf.reshape(vals, [batch_size, h, w, 2*chn]))

def print_array(a):
    for i in range(a.shape[0]):
        for ch in range(a.shape[3]):
            print(a[i,:,:,ch])
        
print_array(result[0:1,:,:,:])

