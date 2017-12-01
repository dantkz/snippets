import tensorflow as tf
import numpy as np
import scipy.misc
import os

tfrecords_filename = '/Users/dturmukh/Code/tmp/cityscapes_val.tfrecords'

print(os.path.splitext(tfrecords_filename)[1])
exit()

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512 

def read_and_decode(filename_queue, batch_size=2, capacity=30, num_threads=2, min_after_dequeue=10):
    reader = tf.TFRecordReader()
    tmp, example = reader.read(filename_queue)
    features = tf.parse_single_example(
      example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'left_jpeg': tf.FixedLenFeature([], tf.string),
        'right_jpeg': tf.FixedLenFeature([], tf.string)
    })
   
    left_img = tf.image.decode_jpeg(features['left_jpeg'])
    right_img = tf.image.decode_jpeg(features['right_jpeg'])
    img_height = tf.cast(features['height'], tf.int32)
    img_width = tf.cast(features['width'], tf.int32)

    img_shape = tf.stack([img_height, img_width, 3])
    left_img = tf.reshape(left_img, img_shape)
    right_img = tf.reshape(right_img, img_shape)

    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)

    resized_left_img = tf.image.resize_image_with_crop_or_pad(image=left_img,
                                            target_height=IMAGE_HEIGHT,
                                            target_width=IMAGE_WIDTH)
    resized_right_img = tf.image.resize_image_with_crop_or_pad(image=right_img,
                                            target_height=IMAGE_HEIGHT,
                                            target_width=IMAGE_WIDTH)

    left_imgs, right_imgs = tf.train.shuffle_batch([resized_left_img, resized_right_img],
                                                        batch_size=batch_size, capacity=capacity,
                                                        num_threads=num_threads, min_after_dequeue=min_after_dequeue)
    return tmp, left_imgs, right_imgs



filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=10)
tmp, left_imgs, right_imgs = read_and_decode(filename_queue)
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
                
        # Let's read off 3 batches just for example
        for i in range(1):
            tmp_np, left_img_np, right_img_np = sess.run([tmp, left_imgs, right_imgs])
            scipy.misc.imsave('left.png', left_img_np[0,:,:,:])
            scipy.misc.imsave('right.png', right_img_np[0,:,:,:])
            print(tmp_np)

        coord.request_stop()
        coord.join(threads)


