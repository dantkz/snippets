import tensorflow as tf
import numpy as np

with tf.device("/gpu:0"):
    data_num = 2**16
    #data_num = 16
    x = tf.placeholder(dtype=tf.float32, shape=[data_num, 1])
    
    cur_x = np.reshape(np.random.randn(data_num).astype(np.float32), [data_num, 1])
    
    x_floor = tf.floor(x)
    x_ceil = x_floor + 1
    
    x_uniques, x_indices = tf.unique(tf.reshape(tf.concat([x_floor, x_ceil], axis=0), [-1]), out_idx=tf.int64)
    
    x_minus = x_uniques-1
    x_plus = x_uniques+1
    
    _, u_indices = tf.unique(tf.reshape(tf.concat([x_uniques, x_minus, x_plus], axis=0), [-1]), out_idx=tf.int64)
    
    u3_indices = tf.reshape(u_indices, [3,-1])
    
    x_u_indices      = tf.reshape(u3_indices[0:1,:], [-1, 1])
    x_uminus_indices = tf.reshape(u3_indices[1:2,:], [-1, 1])
    x_uplus_indices  = tf.reshape(u3_indices[2:3,:], [-1, 1])
    
    
    x_indices = tf.reshape(x_indices, [-1, 1])
    bins_shape = [tf.reduce_max(u_indices)+1, 1]
    
    bins = tf.scatter_nd(x_indices, tf.concat([x-x_floor, x_ceil-x], axis=0), bins_shape)
    
    x_uminus_indices = tf.reshape(x_uminus_indices, [-1, 1])
    x_uplus_indices = tf.reshape(x_uplus_indices, [-1, 1])
    for i in range(3):
        updates = -0.5*tf.gather_nd(bins, x_u_indices) 
        updates += 0.5*tf.gather_nd(bins, x_uminus_indices) 
        updates += 0.5*tf.gather_nd(bins, x_uplus_indices) 
        bins += tf.scatter_nd(x_u_indices, updates, bins_shape)
    
    x_bins = tf.gather_nd(bins, x_indices)
    
    cost = tf.reduce_mean(tf.reduce_sum(x_bins, axis=1))
    
    grad = tf.gradients(cost, x)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        bins_np, cost_np, grad_np = sess.run([bins, cost, grad], feed_dict={x: cur_x})
    
    print(bins_np)
    print(cost_np)

