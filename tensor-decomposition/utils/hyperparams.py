import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=100, type=int,
                        help='Mini batch size for optimization algorithms')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs, typically passes through the entire dataset')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate for the optimization algorithm')
    parser.add_argument('--data_aug', dest='data_aug', action='store_true',
                        help='Add MaxPool layers to Program')
    parser.add_argument('--no-data_aug', dest='data_aug', action='store_false',
                        help='Data augmentation for the model')
    parser.set_defaults(data_aug=False)
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD', 'RMSprop', 'Adam'],
                        help='Options of optimizer to be used')
    parser.add_argument('--BN_alpha', default=0.9, type=float,
                        help='Momentum for the Batch Normalization')
    parser.add_argument('--BN_eps', default=1e-4, type=float,
                        help='Epsilon added to variance to avoid dividing by zero')
    parser.add_argument('--weights_path', default='pretrained_cifar10.h5', type=str,
                        help='Pre-trained model weights')

    args = parser.parse_args()

    return args
