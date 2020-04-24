import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=100, type=int,
                        help='Mini batch size for optimization algorithms')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs, typically passes through the entire dataset')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate for the optimization algorithm')
    parser.add_argument('--add_maxpool', action='store_true', default=True,
                        help='Add MaxPool layers to Program')
    parser.add_argument('--data_aug', action='store_true', default=True,
                        help='Data augmentation for the model')
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'RMSprop', 'Adam'],
                        help='Options of optimizer to be used')

    args = parser.parse_args()

    return args
