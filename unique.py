import tensorflow as tf
import numpy as np
import psutil
from os import getpid

val_num = 256*256
val_dim = 5
max_val = 20

def main():

    with tf.device("/gpu:0"):
        x = tf.placeholder(tf.int32, [val_num, val_dim])
    
        def tf_unique_row_idxs(inp, max_dim=None, name=''):
            with tf.variable_scope('tf_unique_row_idxs_'+name) as scope:
                if not max_dim:
                    max_dim = inp.get_shape().as_list()[1]
                _, idx = tf.unique(inp[:,0], out_idx=tf.int32)
                for j in range(1, max_dim):
                    new_vals = inp[:,j]
                    val_min = tf.reduce_min(new_vals)
                    val_max = tf.reduce_max(new_vals)
                    idx_shift = val_max - val_min + 1
                    vals = idx*idx_shift + new_vals - val_min
                    uvals, idx = tf.unique(vals, out_idx=tf.int32)
                max_pos = tf.shape(uvals, out_type=tf.int32)[0] + 0
                return idx, max_pos
    
        idxs, max_pos = tf_unique_row_idxs(x)
    
    
    process = psutil.Process(getpid())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    mem_before = process.memory_info().rss/2**30
    for i in range(1000):
        cur_x = np.random.randint(0, max_val, [val_num, val_dim], dtype=np.int32)
        cur_feed_dict = {x: cur_x}
        cur_max_pos, cur_idxs = sess.run([max_pos, idxs], feed_dict=cur_feed_dict)

        #print(cur_x)
        #print(cur_max_pos)
        #print(cur_idxs)

        if i%100==0:
            mem_after = process.memory_info().rss/2**30
            print("%d. memory change %.4f -> %.4f" % (i, mem_before, mem_after))
            mem_before = mem_after
    

if __name__ == "__main__":
    main()
