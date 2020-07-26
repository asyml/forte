import tensorflow as tf

def variable_summaries(var):
    '''
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    From https://www.tensorflow.org/get_started/summaries_and_tensorboard
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    
def resize_tensor_variable(sess, tensor_variable, shape):
    sess.run(tf.assign(tensor_variable, tf.zeros(shape), validate_shape=False))
