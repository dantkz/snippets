import tensorflow as tf

with tf.device("/gpu:0"):
    a = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    c = tf.Variable([0, 0, 0, 0, 0, 0, 0, 0])
    step_sos = tf.Variable([False, False, True, True, False, False, True, True])
    write_ops = []
    for b in range(8):
        write_ops.append(tf.cond(step_sos[b], lambda: tf.scatter_update(a, b, 0), lambda: a))

    with tf.control_dependencies(write_ops):
       d = tf.assign(c, a)


session_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)

with tf.Session(config=session_config) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(d)
    print(sess.run(c))
