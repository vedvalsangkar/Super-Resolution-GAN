from torch import nn, cuda, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from baseline_models import Generator, Discriminator

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

import os


class ValidDataSet(Dataset):
    """
    Custom dataset class designed to handle 2 datasets, DIV2k and Flickr2k.
    """

    def __init__(self, root_folder, lr_transform=transforms.ToTensor()):
        self.root_image_folder = root_folder  # Data/DIV_Flickr/train/

        self.lr_transform = lr_transform

        self.main_df = self.root_image_folder + pd.DataFrame(data={"LD": sorted(os.listdir(self.root_image_folder))})

        self.length = self.main_df.shape[0]

    def __getitem__(self, item):

        row = self.main_df.iloc[item, 0]

        try:
            low_res = Image.open(row)
            # print(row)
            filename = os.path.splitext(os.path.basename(row))
            return self.lr_transform(low_res), filename[0]
        except FileNotFoundError:
            print("File not found:", row)

    def __len__(self):
        return self.length


testing_set = ValidDataSet(root_folder="val/") # This is the folder to load images in.

test_loader = DataLoader(dataset=testing_set,
                         batch_size=1,
                         shuffle=False)

device = torch.device("cuda:0" if cuda.is_available() else "cpu")
op_dir = "output_/"

gen = Generator(upscale_factor=2, bias=False).to(device)

gen.load_state_dict(torch.load("models/gen_model.pt"))

for i, (low_res, name) in enumerate(test_loader):
    # print(low_res)

    output = gen(low_res.to(device))

    image = output.detach().cpu()[0]

    save_image(tensor=image,
               filename=op_dir + "SR_{0}_SR.jpg".format(name[0]),
               nrow=1)