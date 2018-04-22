import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
import data_utils
from datetime import datetime
from tensorflow.contrib.data import Iterator
from data_utils import ImageDataGenerator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = './images/train.txt'
val_file = './images/test.txt'

# Learning params
learning_rate = 0.0001
num_epochs = 200
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 6
train_layers = []

# How often we want to write the tf.summary data to disk
display_step = 2

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./alexnet/tensorboard"
checkpoint_path = "./alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the gpu
tr_data = ImageDataGenerator(train_file,
                             mode='training',
                             batch_size=batch_size,
                             num_classes=num_classes,
                             shuffle=True)
val_data = ImageDataGenerator(val_file,
                              mode='inference',
                              batch_size=batch_size,
                              num_classes=num_classes,
                              shuffle=False)

# create an reinitializable iterator given the dataset structure
iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)
next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)
global_step   = tf.Variable(0, trainable=False, name="glAAobal_step")

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes)

# Link variable to model output
score = model.fc8

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Train op
learning_rate = tf.Variable( float(learning_rate), trainable=False, name="learning_rate")

decay_steps = 1000  # empirical
decay_rate = 0.96     # empirical

learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)

opt = tf.train.AdamOptimizer( learning_rate )
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  # Update all the trainable parameters
  gradients = opt.compute_gradients(loss)
  gradients = [[] if i==None else i for i in gradients]
  updates = opt.apply_gradients(gradients, global_step=global_step)  
# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    # model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(updates, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))