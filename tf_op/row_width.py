import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

row_widths = tf.placeholder(dtype=tf.int32, shape=[None], name="width")
batch_size = tf.size(row_widths)
max_width = tf.reduce_max(row_widths)
indices = tf.to_int64(tf.where(tf.sequence_mask(row_widths, max_width)))
dense_shapes = tf.to_int64([batch_size, max_width])

with tf.Session() as sess:
	x,y = sess.run([indices,dense_shapes], feed_dict={"width:0" : [2, 3, 4, 5]})
	print(x)
	print(x.shape)
	print(y)

print("begin custom op")
sparse_module = tf.load_op_library('./sparse_transformer.so')
z = sparse_module.sparse_transformer(row_widths)
print("get the shape before run")
print(z.indices.get_shape())
print(z.shape.get_shape())
print("begin to run")
with tf.Session() as sess:
	x,y = sess.run(z, feed_dict={"width:0" : [2, 3, 4, 5]})
	print(x.shape)
	print(x)
	print(y)
