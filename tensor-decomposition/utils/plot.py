import os
import pandas as pd
import matplotlib.pyplot as plt

r_fixed = ['r1', 'r2']

fig, ax = plt.subplots()
for index, r in enumerate(r_fixed):
    experiment_name = "CONV_2_reconstruction_loss_{}_64".format(r)

    directory = os.path.join('../results', experiment_name)
    df = pd.read_csv(os.path.join(directory, 'results.csv'), sep='\t')
    df.plot(x=r_fixed[(index + 1) % 2], y='reconstruction_loss', ax=ax, label="{}=64 (fixed)".format(r))

ax.set_xticks(range(0, 65, 4))
ax.set_xlabel("Varying Rank")
ax.set_ylabel("Reconstruction Error")
ax.set_title("Reconstruction Error for Varying Ranks")
ax.grid(linewidth=0.25)
plt.savefig('../results/recon.pdf')
