import tensorflow as tf 

one_hot = tf.one_hot([8],6)
with tf.Session() as sess:
	print(one_hot)
	print(sess.run(one_hot))