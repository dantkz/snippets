import tensorflow as tf
import numpy as np
import psutil
from os import getpid

val_num = 8*256*256
val_dim = 5
max_val = 200
num_steps = 10000


def main():

    with tf.device("/gpu:0"):
        x = tf.placeholder(tf.int32, [val_num, val_dim])
    
        def tf_unique_row_idxs(inp, max_dim=None, name=''):
            with tf.variable_scope('tf_unique_row_idxs_'+name) as scope:
                if not max_dim:
                    max_dim = inp.get_shape().as_list()[1]
                new_vals = inp[:,0]
                new_vals = tf.cast(new_vals, dtype=tf.int32)
                _, idx = tf.unique(new_vals, out_idx=tf.int32)
                for j in range(1, max_dim):
                    new_vals = inp[:,j]
                    new_vals = tf.cast(new_vals, dtype=tf.int32)
                    val_min = tf.reduce_min(new_vals)
                    val_max = tf.reduce_max(new_vals)
                    idx_shift = val_max - val_min + 1
                    vals = idx*idx_shift + new_vals - val_min
                    uvals, idx = tf.unique(vals, out_idx=tf.int32)
                max_pos = tf.shape(uvals, out_type=tf.int32)[0] + 0
                return idx, max_pos
    
        idxs, max_pos = tf_unique_row_idxs(x)
    
    
    process = psutil.Process(getpid())

    cur_config=tf.ConfigProto(allow_soft_placement=False,log_device_placement=False)
    sess = tf.Session(config=cur_config)
    sess.run(tf.global_variables_initializer())
    

    mem_usage = np.zeros([num_steps], dtype=np.float32)

    np.random.seed(0)
    for i in range(num_steps):
        cur_x = np.random.randint(0, max_val, [val_num, val_dim], dtype=np.int32)
        cur_feed_dict = {x: cur_x}
        cur_max_pos, cur_idxs = sess.run([max_pos, idxs], feed_dict=cur_feed_dict)
        mem_usage[i] = process.memory_info().rss/2**30
        if i%100==0:
            print(i/num_steps)
    

    np.savetxt('unique_memlog.txt', mem_usage)

if __name__ == "__main__":
    main()
