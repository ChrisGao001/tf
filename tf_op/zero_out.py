
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

zero_out_module = tf.load_op_library('./zero_out.so')
x = zero_out_module.zero_out([[1, 2], [3, 4]])
with tf.Session() as sess:
	y = sess.run(x)
	print(x.shape)
	print(y)
