# Model related imports
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from .pruned_layers import pruned_Conv2D, pruned_Dense
import numpy as np


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


def convert_to_masked_model(model):
    """x
    :param model: input model structure as Sequential structure
    :return: another model with masked_conv and masked_dense layers
    """
    masked_model = Sequential()

    for (index, layer) in list(enumerate(model.layers)):
        if isinstance(layer, Conv2D):
            conf = layer.__class__.get_config(layer)
            if index == 0:
                masked_model.add(
                    pruned_Conv2D(filters=conf['filters'], kernel_size=conf['kernel_size'], strides=conf['strides'],
                                  padding=conf['padding'], input_shape=list(conf['batch_input_shape'][1:])))
            else:
                masked_model.add(
                    pruned_Conv2D(filters=conf['filters'], kernel_size=conf['kernel_size'], strides=conf['strides'],
                                  padding=conf['padding']))

        elif isinstance(layer, Dense):
            conf = layer.__class__.get_config(layer)
            masked_model.add(pruned_Dense(conf['units']))

        else:
            masked_model.add(layer)

    for masked_layer, model_layer in zip(masked_model.layers, model.layers):
        if isinstance(model_layer, Conv2D) or isinstance(model_layer, Dense):
            new_weights = model_layer.get_weights()
            new_weights.append(masked_layer.get_weights()[-1])
            masked_layer.set_weights(new_weights)
        else:
            masked_layer.from_config(model_layer.get_config())
            masked_layer.set_weights(model_layer.get_weights())

    return masked_model
