import sys, os, math, random
import tensorflow as tf
import numpy as np

if __name__=='__main__':
  def run_iters(relu):
    from tensorflow.core.protobuf import rewriter_config_pb2
    rewrite_options = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
      memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
    graph_options=tf.GraphOptions(optimizer_options=optimizer_options,
                                  rewrite_options=rewrite_options)
    config = tf.ConfigProto(graph_options=graph_options)

    sess = tf.Session(config=config)
    
    size = 12000
    num_runs = 20

    images = tf.random_uniform([size, size])

    var = tf.Variable(tf.ones_like(images))
    sess.run(var.initializer)

    cost = tf.reduce_sum(relu(images+var))
    grads = tf.gradients(cost, var)

    memuse, memuse2 = sess.run([tf.contrib.memory_stats.MaxBytesInUse(), tf.contrib.memory_stats.BytesInUse()])
    print("Init: GBs in use %.2f, %.2f"%(memuse/10**9,memuse2/10**9))
    for i in range(10):
      _, memuse, memuse2 = sess.run([grads, tf.contrib.memory_stats.MaxBytesInUse(), tf.contrib.memory_stats.BytesInUse()])
      print("Run %d, GBs in use %.2f, %.2f"%(i, memuse/10**9,memuse2/10**9))
    [memuse] = sess.run([tf.contrib.memory_stats.MaxBytesInUse()])
    print("Memory GBs in use %.2f"%(memuse/10**9,))
    
    sess.close()

  alpha = 0.1

  def relu_nowhere(x):
    retval = alpha*x*tf.cast(tf.less(x, 0.), dtype=tf.float32) + x*tf.cast(tf.less(-x, 0.), dtype=tf.float32)
    return retval
  run_iters(relu_nowhere)

  tf.reset_default_graph()

  def relu_where(x):
    retval = tf.where(tf.less(x, 0.0), alpha*x, x, name='leaky_relu')
    return retval
  run_iters(relu_where)

