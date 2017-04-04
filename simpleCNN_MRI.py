# A simple CNN to predict certain characteristics of the human subject from MRI images.
# 3d convolution is used in each layer.
# Reference: https://www.tensorflow.org/get_started/mnist/pros, http://blog.naver.com/kjpark79/220783765651
# Adjust needed for your dataset e.g., max pooling, convolution parameters, training_step, batch size, etc

width = 256
height = 256
depth = 40
nLabel = 4

# Start TensorFlow InteractiveSession
import input_3Dimage
import tensorflow as tf
sess = tf.InteractiveSession()

# Placeholders (MNIST image:28x28pixels=784, label=10)
x = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]

## Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')


## First Convolutional Layer
# Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]
print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
print(h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool 
print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32


## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
b_conv2 = bias_variable([64]) # [64]

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool 
print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64


## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([16*16*3*64, 1024])  # [7*7*64, 1024]
b_fc1 = bias_variable([1024]) # [1024]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*3*64])  # -> output image: [-1, 7*7*64] = 3136
print(h_pool2_flat.get_shape)  # (?, 2621440)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)  # -> output: 1024

## Readout Layer
W_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  # -> output: 10

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for i in range(100):
    batch = get_data_MRI(sess,'train',20)
    # Logging every 100th iteration in the training process.
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Evaulate our accuracy on the test data
testset = get_data_MRI(sess,'test',30)
print("test accuracy %g"%accuracy.eval(feed_dict={x: testset[0], y_: teseset[1], keep_prob: 1.0}))
