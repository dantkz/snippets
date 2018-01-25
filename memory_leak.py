import tensorflow as tf
import numpy as np

with tf.device("/gpu:0"):
    #data_num = 2**16
    data_num = 16
    x = tf.placeholder(dtype=tf.float32, shape=[data_num, 1])
    
    cur_x = np.reshape(np.random.randn(data_num).astype(np.float32), [data_num, 1])
    
    x_floor_f = tf.floor(x)
    x_ceil_f = x_floor_f + 1

    x_floor = tf.cast(x_floor_f, dtype=tf.int64)
    x_ceil = x_floor + 1
    
    x_uniques, x_indices = tf.unique(tf.reshape(tf.concat([x_floor, x_ceil], axis=0), [-1]), out_idx=tf.int64)
    
    x_uminus = x_uniques-1
    x_uplus = x_uniques+1
    
    _, u_indices = tf.unique(tf.reshape(tf.concat([x_uniques, x_uminus, x_uplus], axis=0), [-1]), out_idx=tf.int64)
    
    u3_indices = tf.reshape(u_indices, [3,-1])
    u3_indices = tf.unstack(u3_indices, axis=0)
    
    x_u_indices      = tf.reshape(u3_indices[0], [-1, 1])
    x_uminus_indices = tf.reshape(u3_indices[1], [-1, 1])
    x_uplus_indices  = tf.reshape(u3_indices[2], [-1, 1])
    
    x_indices = tf.reshape(x_indices, [-1, 1])
    bins_shape = tf.convert_to_tensor([tf.reduce_max(u_indices)+1, 1])
    
    bins = tf.scatter_nd(x_indices, tf.concat([x-x_floor_f, x_ceil_f-x], axis=0), bins_shape)
    
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

    
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))
sess.run(tf.global_variables_initializer())
sess.graph.finalize()

for i in range(3000):
    bins_np, cost_np, grad_np = sess.run([bins, cost, grad], feed_dict={x: cur_x})

print(bins_np)
print(cost_np)

