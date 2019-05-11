from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

import os


class TemplateDataSet(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class DIVFlickrDataSet(Dataset):

    def __init__(self, root_folder, lr_transform=transforms.ToTensor(), hr_transform=transforms.ToTensor()):
        self.root_image_folder = root_folder   # Data/DIV_Flickr/train/
        self.DIV_HR_folder = root_folder + "DIV_HR/"
        self.DIV_LR_folder = root_folder + "DIV_LR/"
        self.Flickr_HR_folder = root_folder + "Flickr_HR/"
        self.Flickr_LR_folder = root_folder + "Flickr_LR/"

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.DIV_df = pd.DataFrame(data={"HD": sorted(os.listdir(self.DIV_HR_folder)),
                                         "LD": sorted(os.listdir(self.DIV_LR_folder))}
                                   )
        self.DIV_df["HD"] = self.DIV_HR_folder + self.DIV_df["HD"]
        self.DIV_df["LD"] = self.DIV_LR_folder + self.DIV_df["LD"]

        self.Flickr_df = pd.DataFrame(data={"HD": sorted(os.listdir(self.Flickr_HR_folder)),
                                            "LD": sorted(os.listdir(self.Flickr_LR_folder))}
                                      )

        self.main_df = pd.concat([self.DIV_df, self.Flickr_df], axis=0)

        self.length = self.main_df.shape[0]

    def __getitem__(self, item):

        row = self.main_df.iloc[item, :]

        low_res = Image.open(row["LD"])
        high_res = Image.open(row["HD"])

        return self.lr_transform(low_res), self.hr_transform(high_res)

    def __len__(self):
        return self.length


class Set5DataSet(Dataset):

    def __init__(self, im_set=2, lr_transform=transforms.ToTensor(), hr_transform=transforms.ToTensor()):

        self.image_folder = "Data/Set5/image_SRF_{0}/".format(im_set)

        self.low_res_imgs = []
        self.high_res_imgs = []

        list_of_imgs = os.listdir(self.image_folder)

        for img_name in sorted(list_of_imgs):
            if "LR" in img_name:
                self.low_res_imgs.append(img_name)
            else:
                self.high_res_imgs.append(img_name)

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.length = len(self.low_res_imgs)

    def __getitem__(self, index):

        low_res = Image.open(self.image_folder + self.low_res_imgs[index])
        high_res = Image.open(self.image_folder + self.high_res_imgs[index])

        return self.lr_transform(low_res), self.hr_transform(high_res)

    def __len__(self):
        return self.length
