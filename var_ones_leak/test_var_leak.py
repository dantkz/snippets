import tensorflow as tf
import numpy as np
import time

if __name__=='__main__':
    size_max = 2**8-1
    iter_num = 20
    size = tf.placeholder(shape=[], dtype=tf.int32)

    with tf.device("/gpu:0"):
        ones_list = []
        ones_list.append(tf.constant(np.ones([size_max, size_max, size_max], dtype=np.float32), dtype=tf.float32, name='tf.constant'))
        ones_list.append(tf.ones([size_max, size_max, size_max], dtype=tf.float32, name='tf.ones'))
        ones_list.append(tf.convert_to_tensor(np.ones([size_max, size_max, size_max], dtype=np.float32), dtype=tf.float32, name='tf.convert_to_tensor'))
        ones_list.append(tf.get_variable('tf.get_variable', dtype=tf.float32, initializer=np.ones([size_max, size_max, size_max], dtype=np.float32), trainable=False))


        versions = []
        for ones in ones_list:
            ones_crop = ones[0:size, 0:size, 0:size]
            noise = tf.random_uniform([size, size, size], dtype=tf.float32)
            cost = tf.reduce_mean(ones_crop + noise)
            grads = tf.gradients(cost, ones)
            versions.append((cost, grads, ones))


    get_mem_max = tf.contrib.memory_stats.MaxBytesInUse()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        #sess.graph.finalize()

        # hot start
        for cost, grads, ones in versions:
            cur_size = np.random.randint(size_max//2+1, size_max, dtype=np.int32)
            cost_val, _, memuse = sess.run([cost, grads, get_mem_max], feed_dict={size: cur_size})

        for cost, grads, ones in versions:
            print("########################")
            print("Ones: ", ones)
            bef_memuse = sess.run(get_mem_max)
            bef_time = time.time()
            #print("Memory GBs in use %.3f before iters"%(bef_memuse/10**9))
            for i in range(iter_num):
                get_mem_use = tf.contrib.memory_stats.BytesInUse()
                cur_size = np.random.randint(size_max//2+1, size_max, dtype=np.int32)
                cost_val, _, memuse, memuse2 = sess.run([cost, grads, get_mem_max, get_mem_use], feed_dict={size: cur_size})
                #print("Run %d, %f, GBs in use %.3f, %.3f."%(i, cost_val, memuse/10**9,memuse2/10**9))
            aft_memuse = sess.run(get_mem_max)
            aft_time = time.time()
            print('Time for %d iters: %.3fs.' % (iter_num, aft_time-bef_time))
            #print("Memory GBs in use %.3f after iters"%(aft_memuse/10**9))
            print("Diff GBs: %.3f"%((aft_memuse-bef_memuse)/10**9))



