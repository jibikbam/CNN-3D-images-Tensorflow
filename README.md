# CNN_MRI

* Goal: MRI classification task using CNN (Convolutional Neural Network)

* Code Dependency: Tensorflow 1.0, Anaconda 4.3.8, Python 2.7

* Difficulty in learning a model for 3D medical images
  1. Data size is too big. e.g., 218x182x218 or 256x256x40
  2. There is only limited number of data. In other words, training size is too small.

* Possible solutions
  1. Be equipped with good machine if affordable. e.g., GTX 1080 TI x 4, 16GB RAM x 4, Intel Core i7-6900K
  2. Downsample images in the preprocessing
  3. Data augmentation e.g., rotate, shift, combination
  4. Transfer learning
