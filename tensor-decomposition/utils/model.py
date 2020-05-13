# Model related imports
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def get_model(args):
    BN_alpha = args.BN_alpha
    BN_eps = args.BN_eps

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', input_shape=[32, 32, 3]))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model
