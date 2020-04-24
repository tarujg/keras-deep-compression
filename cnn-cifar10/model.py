# Data loading and pre-processing
from keras.datasets import cifar10
from keras.utils import np_utils

import tensorflow as tf
from keras import backend as K

# Model related imports
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator

from .utils.hyperparams import parse_args
from .utils.utils import *

import os

from warnings import filterwarnings

filterwarnings("ignore")

# Run code on GPU
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
sess = tf.Session(config=config)
K.backend.set_session(sess)

# Read Data and do initial pre-processing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# convert to float32 (to not make all features zero)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# divide features by 255 so that pixels have value in the range of [0,1]
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices (one-hot encoding)
n_output = 10
y_train = np_utils.to_categorical(y_train, n_output)
y_test = np_utils.to_categorical(y_test, n_output)

print("X_Train: {}\nX_Test:  {}\nY_Train: {}\nY_Test:  {}" \
      .format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

arg = parse_args()
experiment_name = "opt_{}_lr_{}_maxpool_{}_augmentation_{}".format(arg.optimizer, arg.lr, arg.add_maxpool, arg.data_aug)

# Model Creation
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
if arg.add_maxpool:
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
if arg.add_maxpool:
    model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Test for checking initalization of weights
# for layer in model.get_weights():
#    print(np.average(layer))

# Choosing optimizer
if arg.optimizer == 'RMSprop':
    optimizer = RMSprop(learning_rate=arg.lr, rho=0.9)
elif arg.optimizer == 'Adam':
    optimizer = Adam(learning_rate=arg.lr, beta_1=0.9, beta_2=0.999)
else:
    optimizer = SGD(lr=arg.lr, momentum=0.9, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

if not arg.data_aug:
    # Training without data augmentation
    history = model.fit(x_train, y_train, batch_size=arg.batch_size, epochs=arg.epochs,
                        validation_data=(x_test, y_test))
else:
    # Data augmentation
    train_datagen = ImageDataGenerator(width_shift_range=0.15, height_shift_range=0.15, horizontal_flip=True)

    train_generator = train_datagen.flow(x_train, y_train, batch_size=arg.batch_size)

    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(x_test, y_test, batch_size=arg.batch_size)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(x_train) // arg.batch_size,
                                  epochs=arg.epochs,
                                  validation_data=val_generator,
                                  validation_steps=len(x_test) // arg.batch_size)

create_dir_if_not_exists('./checkpoints')
create_dir_if_not_exists(os.path.join('./checkpoints', experiment_name))
save_loss_plots(history, os.path.join('./checkpoints', experiment_name))
model.save_weights(os.path.join('./checkpoints', experiment_name, 'model.h5'))
