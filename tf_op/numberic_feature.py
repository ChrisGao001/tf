import math
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

m = tf.load_op_library('./numberic_feature.so')
def sign(n):
    return float((n > 0) - (n < 0))


def get_capped_min_max_norm(x, fmin, fmax):
    return (tf.maximum(fmin, tf.minimum(tf.cast(x, tf.float32), fmax)) -
            fmin) / (fmax - fmin)


def min_max_norm(x=1.09861, fmin=0, fmax=6.97167):
    return (max(fmin, min(x, fmax)) - fmin) / (fmax - fmin)


def log_sign_numeric_plus_one(n):
    return sign(n) * math.log(abs(n) + 1.0)


def log_x_plus_one(input_tensor, numeric=False):
    if not numeric:
        y = tf.cast(input_tensor, tf.float32)
        return tf.sign(y) * tf.log(tf.abs(y) + 1.0)
    else:
        return log_sign_numeric_plus_one(input_tensor)


def log_min_max(x, fmin, fmax):
    if isinstance(x, tf.SparseTensor):
        x_value = x.values
    else:
        x_value = x
    x_log = log_x_plus_one(x_value)
    fmin_log = log_x_plus_one(fmin)
    fmax_log = log_x_plus_one(fmax)
    x_value = get_capped_min_max_norm(x_log, fmin_log, fmax_log)
    if isinstance(x, tf.SparseTensor):
        return tf.SparseTensor(
            indices=x.indices, values=x_value, dense_shape=x.dense_shape)
    else:
        return x_value

def log_min_max_opt(x, fmin, fmax):
    if isinstance(x, tf.SparseTensor):
        x_value = x.values
    else:
        x_value = x
    fmin_log = log_x_plus_one(fmin, numeric=True)
    fmax_log = log_x_plus_one(fmax, numeric=True)
    log_min_max_cpp = m.log_min_max
    x_value = log_min_max_cpp(x_value, fmin_log, fmax_log)
    if isinstance(x, tf.SparseTensor):
        return tf.SparseTensor(
            indices=x.indices, values=x_value, dense_shape=x.dense_shape)
    else:
        return x_value

def test_log_min_max():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
    #y_tensor = log_min_max(x, 0, 1065)
    y_tensor_opt = log_min_max_opt(x, 0, 1065)
    #with tf.Session() as sess:
    #    y = sess.run(y_tensor, feed_dict={x : [[-2], [0], [4], [5]]})
    print("begin to session run")
    with tf.Session() as sess:
        y_opt = sess.run(y_tensor_opt, feed_dict={x : [[-2], [0], [4], [5]]})
        print(y_opt)
if __name__ == '__main__':
    test_log_min_max()
