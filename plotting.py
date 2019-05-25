import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(filepath_or_buffer="output/20190516:051701_VGG/data_20190516:051701.csv",
                   header=0,
                   index_col=0
                   ).drop(["Epoch", "Step"], axis=1)

x = data.shape[0]

plt.figure()
plt.plot(data['Discriminator Loss'], 'g')
plt.ylabel('Discriminator Loss')
plt.xlabel('Batches')
plt.title('Discriminator Loss')

# fig, axs = plt.subplots(3, 1, constrained_layout=True)
fig, axs = plt.subplots(2, 1, constrained_layout=True)
fig.suptitle('Generator Losses', fontsize=16)
axs[0].plot(data['Generator Classification Loss'] * 0.002, 'b')
axs[0].set_xlabel('Batches')
axs[0].set_ylabel('Generator Adversarial Loss')
axs[0].set_title('Generator Adversarial Loss')

axs[1].plot(data['Generator MSE Loss'], 'r')
axs[1].set_xlabel('Batches')
axs[1].set_ylabel('Generator Content Loss')
axs[1].set_title('Generator Content Loss')

# axs[2].plot(data['Generator Loss'], 'g')
# axs[2].set_xlabel('Batches')
# axs[2].set_ylabel('Generator Total Loss')
# axs[2].set_title('Generator Total Loss')

plt.show()

