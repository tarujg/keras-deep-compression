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
    parser.add_argument('--pruning_percentage', default=0.9, type=float,
                        help='Percentage of weights to prune for a model')

    parser.add_argument('--Layer', type=str,
                        help='Layer to be pruned')

    parser.add_argument('--training', dest='training', action='store_true',
                        help='Boolean flag to train model')
    parser.add_argument('--no-training', dest='training', action='store_false',
                        help='Boolean flag to train model')
    parser.set_defaults(training=True)

    parser.add_argument('--compressing', dest='compressing', action='store_true',
                        help='Boolean flag to compress model')
    parser.add_argument('--no-compressing', dest='compressing', action='store_false',
                        help='Boolean flag to compress model')
    parser.set_defaults(compressing=False)

    args = parser.parse_args()

    return args
