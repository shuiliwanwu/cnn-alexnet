import numpy as np 
import random
import tensorflow as tf 

from tensorflow.contrib.data import Dataset 
from tensorflow.python.framework import dtypes 
from tensorflow.python.framework.ops import convert_to_tensor 

IMAGENET_MEAN = tf.constant([123.68,116.779,103.939], dtype=tf.float32)

class ImageDataGenerator(object):

	def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):
		"""Create a new ImageDataGenerator
		
		Args:
			data_dir: Path to the dataset.
			batch_size: Number of images batch.
			num_classes: Number of classes in the dataset.
			shuffle: Whether or not to shuffle the data in the dataset and the initial file list.file

		Raises:
			ValueError: If an invalid mode is passed
		"""

		self.txt_file = txt_file
		self.num_classes = num_classes

		self._read_txt_file()
		# Number of samples in the dataset
		self.data_size = len(self.labels)

		# Initial shuffling of the file and label lists
		if shuffle:
			self._shuffle_lists()

		self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
		self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
		# Create dataset

		data = Dataset.from_tensor_slices((self.img_paths, self.labels))
		if mode == "training":
			data = data.map(self._parse_function_train, num_threads=8, 
							output_buffer_size=100*batch_size)

		elif mode == "inference":
			data = data.map(self._parse_function_inference, num_threads=8,
							output_buffer_size=100*batch_size)
		else:
			raise ValueError("Invalid model '%s'."%(mode))

		if shuffle:
			data = data.shuffle(buffer_size=buffer_size)

		data = data.batch(batch_size)
		self.data = data

	def _read_txt_file(self):
		"""Read the content and the list of paths and labels.
		"""
		self.img_paths = []  
		self.labels = []

		with open(self.txt_file,'r') as f:
			lines = f.readlines()
			for line in lines:
				items = line.split(' ')
				self.img_paths.append(items[0])
				self.labels.append(int(items[1]))

	def _shuffle_lists(self):
		path = self.img_paths
		labels = self.labels
		permutation = np.random.permutation(self.data_size)
		self.img_paths = []
		self.labels = []
		for i in permutation:
			self.img_paths.append(path[i])
			self.labels.append(labels[i])

	def _parse_function_train(self, filename, label):
		# Convert label number into one-hot-encoding
		one_hot = tf.one_hot(label, self.num_classes)
		# Load and preprocess the image
		img_string = tf.read_file(filename)
		img_decoded = tf.image.decode_png(img_string, channels=3)
		img_resize = tf.image.resize_images(img_decoded, [227,227], method=0)
		img_centered = tf.subtract(img_resize,IMAGENET_MEAN)

		# RGB-> BGR
		# img_bgr = img_centered[:, :, ::-1]
		img_bgr = img_centered[:, :, ::-1]

		return img_bgr, one_hot

	def _parse_function_inference(self, filename, label):
		# Convert label number into one-hot-encoding
		one_hot = tf.one_hot(label, self.num_classes)
		# Load and preprocess the image
		img_string = tf.read_file(filename)
		img_decoded = tf.image.decode_png(img_string, channels=3)
		img_resize = tf.image.resize_images(img_decoded, [227,227], method=0 )
		img_centered = tf.subtract(img_resize,IMAGENET_MEAN)

		# RGB -> BGR
		# img_bgr = img_centered[:, :, ::-1]
		img_bgr = img_centered[:, :, ::-1]

		return img_bgr, one_hot











