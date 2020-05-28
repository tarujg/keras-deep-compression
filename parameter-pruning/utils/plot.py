import os
import pandas as pd
import matplotlib.pyplot as plt

layers = {"CONV": [0, 3, 7, 10, 14, 17],
          "DENSE": [21, 24, 27]}

keys = list(layers.keys())

for item in keys:
    fig, ax = plt.subplots()
    for index, layer_index in enumerate(layers[item]):
        experiment_name = "Layer_{}".format(layer_index)
        directory = os.path.join('../results', experiment_name)

        df = pd.read_csv(os.path.join(directory, 'results.csv'), sep='\t')
        df['Accuracy'] = 100 * df['Accuracy']
        df.plot(x='Pruning_percentage', y='Accuracy', ax=ax, label="{}_{}".format(item, index + 1))

        # Pruning Percentage where accuracy is higher than no pruning
        higher_acc = df['Accuracy'] > df['Accuracy'][0]

        print("{:5}{} - Pruning Percentage: {}% Test Accuracy: {:.2f}%".format(item, index + 1, df['Accuracy'][higher_acc].idxmax(), df['Accuracy'][higher_acc].max()))

    ax.set_xticks(range(0, 105, 10))
    ax.set_yticks(range(0, 95, 10))
    ax.set_xlabel("Percentage of Neurons Pruned")
    ax.set_ylabel("Accuracy on CIFAR-10 test set")
    ax.set_title("Pruning Percentage for {} Layers".format(item))
    ax.grid(linewidth=0.25)

    plt.savefig('../results/{}.pdf'.format(item))
