import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=100, type=int,
                        help='Mini batch size for optimization algorithms')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs, typically passes through the entire dataset')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate for the optimization algorithm')

    parser.add_argument('--add_maxpool', dest='add_maxpool', action='store_true',
                        help='Add MaxPool layers to Program')
    parser.add_argument('--no-add_maxpool', dest='add_maxpool', action='store_false',
                        help='Add MaxPool layers to Program')
    parser.set_defaults(add_maxpool=False)
    parser.add_argument('--data_aug', dest='data_aug', action='store_true',
                        help='Add MaxPool layers to Program')
    parser.add_argument('--no-data_aug', dest='data_aug', action='store_false',
                        help='Data augmentation for the model')
    parser.set_defaults(data_aug=False)

    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'RMSprop', 'Adam'],
                        help='Options of optimizer to be used')

    args = parser.parse_args()

    return args
