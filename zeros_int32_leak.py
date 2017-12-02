import argparse
import psutil

from os import getpid
import tensorflow as tf
import numpy as np

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--output_size', type=int, default=128)
    parser.add_argument('--device', type=str, default="gpu:0")
    return parser.parse_args(args=args)



def get_random_shape(output_size):
    shape = tf.clip_by_value(tf.cast(tf.random_normal([2]) * 38.0 + 64.0, tf.int32), 38, 120)
    shape = tf.concat([[1], shape, [output_size]], axis=0)
    return shape

def create_model_zeros(output_size):
    shape = get_random_shape(output_size)
    return tf.zeros(shape, dtype=tf.float32)

def create_model_zeros_cast(output_size):
    shape = get_random_shape(output_size)
    return tf.cast(tf.zeros(shape, dtype=tf.float32), dtype=tf.int32)

def create_model_zeros_np(output_size):
    shape = get_random_shape(output_size)
    return tf.constant(np.zeros([5, 5, output_size]), dtype=tf.int32)

def create_model_scatter(output_size):
    shape = get_random_shape(output_size)
    outer_shape = tf.reduce_prod(shape)
    indices = tf.convert_to_tensor([[0]], dtype=tf.int32)
    updates = tf.convert_to_tensor([0], dtype=tf.float32)
    result = tf.scatter_nd(indices, updates, [outer_shape])
    return tf.reshape(result, shape)

def create_model_range(output_size):
    shape = get_random_shape(output_size)
    outer_shape = tf.reduce_prod(shape)
    result = tf.cast(tf.range(tf.cast(outer_shape, dtype=tf.int64), dtype=tf.int64), dtype=tf.int32)
    result = tf.reshape(result, shape)
    return result

def create_model_tile(output_size):
    shape = get_random_shape(output_size)
    outer_shape = tf.cast(tf.reduce_prod(shape[:-1]), dtype=tf.float32)
    result = tf.range(outer_shape, dtype=tf.float32)
    result = tf.tile(result, [output_size])
    result = tf.reshape(result, shape)
    return result

def create_model_where(output_size):
    shape = get_random_shape(output_size)
    mask = tf.random_normal(shape) < 0
    ones = tf.cast(tf.ones(shape, dtype=tf.float32), dtype=tf.int32)
    zeros = tf.cast(tf.zeros(shape, dtype=tf.float32), dtype=tf.int32)
    result = tf.where(mask, ones, zeros)
    return result

def create_model_unique(output_size):
    shape = get_random_shape(output_size)
    inp = tf.cast(100*tf.random_normal(shape), dtype=tf.int32)
    inp = tf.reshape(inp, [-1, output_size])
    idx_shift = 2**8
    max_dim = inp.get_shape().as_list()[1]
    _, idx = tf.unique(inp[:,0], out_idx=tf.int64)
    for j in range(1, max_dim-1):
        vals = idx*idx_shift + tf.cast(inp[:,j], dtype=tf.int64)
        _, idx = tf.unique(vals, out_idx=tf.int64)
    j = max_dim-1
    vals = idx*idx_shift + tf.cast(inp[:,j], dtype=tf.int64)
    uvals, result = tf.unique(vals, out_idx=tf.int32)
    result = tf.reshape(result, tf.concat([shape[:-1], [-1]], axis=0))
    return result

def main():
    args = parse_args()
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    np.random.seed(1234)
    process = psutil.Process(getpid())

    with tf.Session(config=session_conf) as session, tf.device(args.device):
        #op = create_model_zeros(args.output_size)
        #op = create_model_zeros_cast(args.output_size)
        #op = create_model_zeros_np(args.output_size)
        #op = create_model_scatter(args.output_size)
        #op = create_model_where(args.output_size)
        #op = create_model_unique(args.output_size)
        #op = create_model_range(args.output_size)
        op = create_model_tile(args.output_size)

        session.run(tf.global_variables_initializer())
        session.graph.finalize()
        before = process.memory_percent()

        for epoch in range(args.max_epochs):
            val = session.run(op)
            #assert (np.sum(np.abs(val)))==0, 'make tf.zeros with tf.scatter_nd'
            
            if epoch % 100 == 0:
                after = process.memory_percent()
                print("MEMORY CHANGE %.4f -> %.4f" % (before, after))
                before = after

if __name__ == "__main__":
    main()
