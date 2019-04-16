import tensorflow as tf 
#hh地方我吃饭
one_hot = tf.one_hot([8],6)
with tf.Session() as sess:
	print(one_hot)
	print(sess.run(one_hot))