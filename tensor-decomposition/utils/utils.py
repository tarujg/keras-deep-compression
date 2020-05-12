import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")


def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_loss_plots(history, directory):
    """Saves a plot of the training and test curves.
    """
    for metric in list(['accuracy', 'loss']):
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(directory, metric + '_plot.pdf'))
        plt.close()
