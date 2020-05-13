from warnings import filterwarnings

filterwarnings("ignore")

# Data loading and pre-processing
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import os
import csv

# Model related imports
from utils.hyperparams import parse_args
from utils.model import get_model
from utils.utils import create_dir_if_not_exists
from utils.tucker import tucker_reconstruction_loss, tucker_decomposition, compute_rank_list

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

r_fixed = 64
experiment_r1_fixed = "CONV_2_reconstruction_loss_r1_{}".format(r_fixed)
experiment_r2_fixed = "CONV_2_reconstruction_loss_r2_{}".format(r_fixed)
create_dir_if_not_exists('./results')
create_dir_if_not_exists(os.path.join('./results', experiment_r1_fixed))
create_dir_if_not_exists(os.path.join('./results', experiment_r2_fixed))
lines_r1, lines_r2 = [], []

for r in list(range(1, 64, 1)):
    lines_r1.append([r_fixed, r, tucker_reconstruction_loss(model.layers[3], [r_fixed, r])])
    lines_r2.append([r, r_fixed, tucker_reconstruction_loss(model.layers[3], [r, r_fixed])])

with open(os.path.join('./results', experiment_r1_fixed, 'results.csv'), 'w', newline='') as file:
    header = ['r1', 'r2', 'reconstruction_loss']
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(header)
    writer.writerows(lines_r1)

with open(os.path.join('./results', experiment_r2_fixed, 'results.csv'), 'w', newline='') as file:
    header = ['r1', 'r2', 'reconstruction_loss']
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(header)
    writer.writerows(lines_r2)
