# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib

FLAGS = tf.app.flags.FLAGS
FLAGS.width = 256
FLAGS.height = 256
FLAGS.depth = 40 # 3
batch_index = 0
filenames = []

# user selection
FLAGS.data_dir = '/home/ikbeom/Desktop/DL/MNIST_simpleCNN/data'
FLAGS.num_class = 4

def get_filenames(data_set):
i    global filenames
    labels = []

    with open(FLAGS.data_dir + '/labels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list

    for i, label in enumerate(labels):
        list = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + label)
        for filename in list:
            filenames.append([label + '/' + filename, i])

    random.shuffle(filenames)


def get_data_MRI(sess, data_set, batch_size):
    global batch_index, filenames

    if len(filenames) == 0: get_filenames(data_set) 
    max = len(filenames)

    begin = batch_index
    end = batch_index + batch_size

    if end >= max:
        end = max
        batch_index = 0

    x_data = np.array([], np.float32)
    y_data = np.zeros((batch_size, FLAGS.num_class)) # zero-filled list for 'one hot encoding'
    index = 0

    for i in range(begin, end):
        
        imagePath = FLAGS.data_dir + '/' + data_set + '/' + filenames[i][0]
        FA_org = nib.load(imagePath)
        FA_data = FA_org.get_data()  # 256x256x40; numpy.ndarray
        
        # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
        resized_image = tf.image.resize_images(images=FA_data, size=(FLAGS.width,FLAGS.height), method=1)

        image = sess.run(resized_image)  # (256,256,40)
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
        y_data[index][filenames[i][1]] = 1  # assign 1 to corresponding column (one hot encoding)
        index += 1

    batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)

    return x_data_, y_data
