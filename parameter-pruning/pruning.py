from warnings import filterwarnings

filterwarnings("ignore")

# Data loading and pre-processing
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np

# Model related imports
from keras.optimizers import Adam
from keras.layers import Input

from utils.hyperparams import parse_args
from utils.pruned_layers import *
from utils.utils import create_dir_if_not_exists
from utils.model import get_model, convert_to_masked_model

import os
import csv

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

batch_size = arg.batch_size
training = arg.training
compressing = arg.compressing

# make sure weights are loaded correctly by evaluating the model here and printing the output
#test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
#print("Non-pruned accuracy: {}".format(test_acc))

# convert the layers to masked layers
pruned_model = convert_to_masked_model(model)
optimizer = Adam(lr=arg.lr, decay=1e-6)
pruned_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

pruned_test_loss, pruned_test_acc = pruned_model.evaluate(x_test, y_test, batch_size=batch_size)
print("Pruned accuracy: {}".format(pruned_test_acc))

experiment_name = "Training_{}_Compressing_{}_Layer_{}".format(arg.training, arg.compressing, arg.Layer)
create_dir_if_not_exists('./results')
create_dir_if_not_exists(os.path.join('./results', experiment_name))

for pruning_percentage in list(range(5, 100, 5)):
    # Read the weights
    weights = pruned_model.layer[0].get_weights()[0]
    # getting the indices of weight values above threshold of pruning_percentage
    index = np.abs(weights) > np.percentile(np.abs(weights), pruning_percentage)
    pruned_model.layer[0].set_mask(index.astype(float))

    # ensure weights loaded correctly by evaluating the prunable model
    pruned_test_loss, pruned_test_acc = pruned_model.evaluate(x_test, y_test, batch_size=batch_size)

    # save_loss_plots(history, os.path.join('./results', experiment_name))

    with open(os.path.join('./results', experiment_name, 'results.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['pruning_percentage', 'pruned_test_loss', 'pruned_test_acc'])
