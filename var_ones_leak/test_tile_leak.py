import tensorflow as tf
import numpy as np
import time

if __name__=='__main__':
    tensor_size = 2**8-1
    iter_num = 20

    with tf.device("/gpu:0"):
        noise = tf.tile(10*tf.random_normal([tensor_size, tensor_size, tensor_size], dtype=tf.float32), [2,1,2])
        #noise = tf.random_normal([2*tensor_size, tensor_size, 2*tensor_size], dtype=np.float32)
        #noise = tf.random_uniform([2*tensor_size, tensor_size, 2*tensor_size], dtype=np.float32)
        cost = tf.reduce_sum(noise)
        grads = tf.gradients(cost, noise)

    get_mem_max = tf.contrib.memory_stats.MaxBytesInUse()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        #sess.graph.finalize()

        bef_memuse = sess.run(get_mem_max)
        bef_time = time.time()
        for i in range(iter_num):
            get_mem_use = tf.contrib.memory_stats.BytesInUse()
            cost_val, _, memuse, memuse2 = sess.run([cost, grads, get_mem_max, get_mem_use], feed_dict={})
            print("Run %d, GBs in use %.3f, %.3f."%(i, memuse/10**9,memuse2/10**9))
        aft_memuse = sess.run(get_mem_max)
        aft_time = time.time()
        print('Time for %d iters: %.3fs.' % (iter_num, aft_time-bef_time))
        print("Diff GBs: %.3f"%((aft_memuse-bef_memuse)/10**9))



