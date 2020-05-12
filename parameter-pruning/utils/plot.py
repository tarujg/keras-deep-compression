import matplotlib.pyplot as plt
import csv
import os

layers = [0, 3, 7, 10, 14, 17, 21, 24,27]

for layer_index in layers:
    experiment_name = "Layer_{}".format(layer_index)

    x = []
    y = []
    directory = os.path.join('../results', experiment_name)

    with open(os.path.join(directory, 'results.csv'), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        next(plots)
        for row in plots:
            x.append(int(row[0]))
            y.append(100*float(row[2]))

        plt.figure()
        plt.plot(x, y)
        plt.title('Test accuracy after pruning')
        plt.ylabel("Accuracy")
        plt.xlabel('Pruning Percentage')
        # plt.legend(['train', 'test'], loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(directory, 'plot.pdf'))
        plt.close()
        plt.tight_layout()
