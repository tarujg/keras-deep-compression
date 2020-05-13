# Model related imports
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from .tucker import compute_rank_list, tucker_decomposition


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


def decomposed_model(model, k):
    decomposed_model = Sequential()

    for (index, layer) in list(enumerate(model.layers)):
        if isinstance(layer, Conv2D) and (index is not 0):
            #if index == 0:
            #    decomposed_model.add(layer)
            #    decomposed_model.layers[-1].from_config(model.layers[index].get_config())
            #    decomposed_model.layers[-1].set_weights(model.layers[index].get_weights())

            #else:
            input_layer, core_layer, output_layer = tucker_decomposition(layer,
                                                                         compute_rank_list(layer, k))

            decomposed_model.add(input_layer)
            decomposed_model.add(core_layer)
            decomposed_model.add(output_layer)

        else:
            decomposed_model.add(layer)
            decomposed_model.layers[-1].from_config(model.layers[index].get_config())
            decomposed_model.layers[-1].set_weights(model.layers[index].get_weights())

    return decomposed_model
