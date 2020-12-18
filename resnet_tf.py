import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np


class ResidualBlockTF(tf.keras.layers.Layer):
  def __init__(self, num_channels, output_channels, strides=1, is_used_conv11=False, **kwargs):
    super(ResidualBlockTF, self).__init__(**kwargs)
    self.is_used_conv11 = is_used_conv11
    self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same', 
                                        kernel_size=3, strides=1)
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(num_channels, padding='same', 
                                        kernel_size=3, strides=1)
    if self.is_used_conv11:
      self.conv3 = tf.keras.layers.Conv2D(num_channels, padding='same', 
                                          kernel_size=1, strides=1)
    # Last convolutional layer to reduce output block shape.
    self.conv4 = tf.keras.layers.Conv2D(output_channels, padding='same',
                                        kernel_size=3, strides=strides)
    self.relu = tf.keras.layers.ReLU()

  def call(self, X):
    if self.is_used_conv11:
      Y = self.conv3(X)
    else:
      Y = X
    X = self.conv1(X)
    X = self.relu(X)
    X = self.batch_norm(X)
    X = self.relu(X)
    X = self.conv2(X)
    X = self.batch_norm(X)
    X = self.relu(X+Y)
    X = self.conv4(X)
    return X

# X = tf.random.uniform((4, 224, 224, 3)) # shape=(batch_size, width, height, channels)
# X = ResidualBlockTF(num_channels=3, output_channels=64, strides=2, is_used_conv11=True)(X)
# print(X.shape)

class ResNet18TF(tf.keras.Model):
  def __init__(self, residual_blocks, output_shape):
    super(ResNet18TF, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same')
    self.relu = tf.keras.layers.ReLU()
    self.residual_blocks = residual_blocks
    self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
    self.dense = tf.keras.layers.Dense(units=output_shape)

  def call(self, X):
    X = self.conv1(X)
    X = self.batch_norm(X)
    X = self.relu(X)
    X = self.max_pool(X)
    for residual_block in residual_blocks:
      X = residual_block(X)
    X = self.global_avg_pool(X)
    X = self.dense(X)
    return X


# Initialize Data Generator
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 X, 
                 y, 
                 batch_size, 
                 input_dim,
                 n_channels,
                 shuffle=True):
        '''
        X: input images
        y: labels
        batch_size: batch size
        input_dim: (width, height)
        n_channels: channels of input image
        n_classes: number of classes 
        shuffle: whether shuffle after each epoch
        '''
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '''
        return:
          total iteration on each epoch
        '''
        return int(np.floor(self.X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        '''
        params:
          index: index of batch
        return:
          X_batch, y_batch
        '''
        # get indexes of all images in batch at the position index. 
        img_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Initialize dât
        X, y = self.__data_generation(img_indexes)
        return X, y

    def on_epoch_end(self):
        '''
        Shuffle data after each time epochs start or end.
        '''
        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        '''
        params:
          indexes: list of all images in batch
        return:
          data X, y of batch after preprocessing
        '''
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        X_batch = self.X[indexes]
        # Preprocessing data X and y
        for i, img in enumerate(X_batch):
            img = cv2.resize(img, self.input_dim)/255.0
            X[i,] = img
        mean = np.mean(X)
        std = np.std(X)
        X = (X-mean)/std
        y = self.y[indexes]
        return X, y

if __name__ == "__main__":
    # Step 1: Initilize model
    residual_blocks = [
        # Two start conv mapping
        ResidualBlockTF(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
        ResidualBlockTF(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
        # Next three [conv mapping + identity mapping]
        ResidualBlockTF(num_channels=64, output_channels=128, strides=2, is_used_conv11=True),
        ResidualBlockTF(num_channels=128, output_channels=128, strides=2, is_used_conv11=False),
        ResidualBlockTF(num_channels=128, output_channels=256, strides=2, is_used_conv11=True),
        ResidualBlockTF(num_channels=256, output_channels=256, strides=2, is_used_conv11=False),
        ResidualBlockTF(num_channels=256, output_channels=512, strides=2, is_used_conv11=True),
        ResidualBlockTF(num_channels=512, output_channels=512, strides=2, is_used_conv11=False)
    ]

    tfmodel = ResNet18TF(residual_blocks, output_shape=10)
    tfmodel.build(input_shape=(None, 224, 224, 3))
    
    
    # Step 2: Initialize Data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.stack((X_train,)*3, axis=-1)/255.0
    X_test = np.stack((X_test,)*3, axis=-1)/255.0
    y_train = y_train.astype(np.int8)
    y_test = y_test.astype(np.int8)
    print(X_test.shape, X_train.shape)
    image_generator = DataGenerator(
        X = X_train,
        y = y_train,
        batch_size = 256,
        input_dim = (96, 96),
        n_channels = 3,
        shuffle = True
    )

    # Step 3: Train model
    opt = Adam(lr=0.05)
    tfmodel.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tfmodel.fit(image_generator, epochs=10)

