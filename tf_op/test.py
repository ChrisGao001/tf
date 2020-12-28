import tensorflow as tf
print(tf.__version__)

row_widths = tf.placeholder(dtype=tf.int32, shape=[None], name="width")
a = tf.placeholder(dtype=tf.int32, shape=[None], name="a")
b = tf.add(a,1)
batch_size = tf.size(row_widths)
max_width = tf.reduce_max(row_widths)
indices = tf.to_int64(tf.where(tf.sequence_mask(row_widths, max_width)))
dense_shapes = tf.to_int64([batch_size, max_width])

with tf.Session() as sess:
	x,y = sess.run([indices,dense_shapes], feed_dict={"width:0" : [2, 3, 4, 5]})
	print(x)
	print(x.shape)
	print(y)
