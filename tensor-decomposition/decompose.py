from warnings import filterwarnings

filterwarnings("ignore")

# Data loading and pre-processing
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

import numpy as np
import os
import csv

# Model related imports
from utils.hyperparams import parse_args
from utils.model import get_model

# Run code on GPU
# config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
# sess = tf.Session(config=config)
# K.set_session(sess)

# Read Data and do initial pre-processing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# convert to float32 (to not make all features zero)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# divide features by 255 so that pixels have value in the range of [0,1]
x_train /= 255
x_test /= 255

x_train = 2 * x_train - 1
x_test = 2 * x_test - 1

# convert class vectors to binary class matrices (one-hot encoding)
n_output = 10
y_train = np_utils.to_categorical(y_train, n_output)
y_test = np_utils.to_categorical(y_test, n_output)

print("X_Train: {}\nX_Test:  {}\nY_Train: {}\nY_Test:  {}" \
      .format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))
print("Train Samples: {}, Test Samples: {}".format(x_train.shape[0], x_test.shape[0]))

arg = parse_args()

# Instantiate Model and load pre-trained weights
model = get_model(arg)
model.load_weights(arg.weights_path)
optimizer = Adam(lr=arg.lr, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# ensure weights are loaded correctly by evaluating the model here and printing the output
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=arg.batch_size)
print("Test accuracy: {}".format(test_acc))

