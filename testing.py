import torch
from torch import nn, cuda, optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchsummary import summary
from torchvision.utils import save_image

from baseline_models import Generator, Discriminator
from datasets import DIVFlickrDataSet

import os
import time


# use_vgg = True
use_vgg = False

testing_set = DIVFlickrDataSet(root_folder="Data/DIV_Flickr/val/",
                               test_mode=True)

test_loader = DataLoader(dataset=testing_set,
                         batch_size=1,
                         shuffle=True)

device = torch.device("cuda:0" if cuda.is_available() else "cpu")

gen_4x = Generator(upscale_factor=2, bias=False).to(device)
dis = Discriminator(image_size=(224, 224), bias=False).to(device)

timestamp = time.strftime("%Y%M%d:%H%M%S")
op_dir = "output/{0}_{1}/".format(timestamp, "VGG" if use_vgg else "RAW")

os.mkdir(op_dir)

data = torch.load("")

gen_4x.load_state_dict(data["generator"])

for i, (low_res, high_res, name) in enumerate(test_loader):
    output = gen_4x(low_res.to(device))

    image = output.detach().cpu()[0]
    ref = high_res[0]

    save_image(tensor=torch.stack([image, ref]),
               filename=op_dir + "SR_{0}.jpg".format(name[0]),
               nrow=2,
               normalize=False
               )

    save_image(tensor=image,
               filename=op_dir + "SR_{0}_SR.jpg".format(name[0]),
               nrow=1
               )

    save_image(tensor=ref,
               filename=op_dir + "SR_{0}_HR.jpg".format(name[0]),
               nrow=1
               )