# https://github.com/aitorzip/PyTorch-SRGAN

import torch
from torch import nn, cuda, optim
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# from torchsummary import summary

import time

from baseline_models import Generator, Discriminator
from datasets import Set5DataSet


def main():

    # -------------------------------- Hyper-parameters --------------------------------

    verbose = True

    img_size = (256, 256)

    bias = False

    learning_rate = 0.001
    lamb = 0
    epochs = 0

    batch_size = 2
    batch_print = 20

    nll_loss_factor = 0.001

    sgd_momentum = 0.3

    op_dir = "models/"
    # ----------------------------------------------------------------------------------

    # --------------------------------- Fetching model ---------------------------------

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    training_set = Set5DataSet(im_set=4)

    train_loader = DataLoader(dataset=training_set,
                              batch_size=batch_size,
                              shuffle=True)

    if verbose:
        print("Dataset and data loaders acquired.")

    gen_2x = Generator(upscale_factor=2, bias=bias).to(device)
    dis = Discriminator(image_size=img_size, bias=bias).to(device)

    gen_optim = optim.Adam(params=gen_2x.parameters(),
                           lr=learning_rate,
                           weight_decay=lamb,
                           amsgrad=False
                           )

    dis_optim = optim.Adam(params=gen_2x.parameters(),
                           lr=learning_rate,
                           weight_decay=lamb,
                           amsgrad=False
                           )

    gen_mse_criterion = nn.MSELoss()
    # gen_nll_criterion = nn.NLLLoss()
    # dis_criterion = nn.NLLLoss()
    dis_criterion = nn.BCELoss()

    # ----------------------------------------------------------------------------------

    # ------------------------------- Start of training --------------------------------
    print("\nStart of training.")
    print("Total epochs: {0}".format(epochs))

    start_time = time.time()

    # Data flow/Algorithm:
    #   1. Set generator to eval() and discriminator to train().
    #   2. Give LR input to GEN and get SR.
    #       2.1 Pass on SR to DIS with label 0. <Calc backprop?>
    #       2.2 Pass HR to DIS with label 1. Calc backprop
    #   3. Set DIS to eval() and GEN to train().
    #   4. Pass LR input to GEN. Get SR.
    #       4.1 Calculate MSELoss() between SR and HR.
    #       4.2 Pass on SR to DIS with label 1.
    #       4.3 Add losses and backprop.

    label_0 = torch.tensor(0).expand(batch_size).to(device)
    label_1 = torch.tensor(1).expand(batch_size).to(device)

    print("LABEL_0:", label_0)

    for epoch in range(epochs):

        for i, (low_res, high_res) in enumerate(train_loader):

            low_res = low_res.to(device)
            high_res = high_res.to(device)

            #
            # 1. Set generator to eval() and discriminator to train().
            #
            gen_2x.eval()
            dis.train()

            gen_optim.zero_grad()
            dis_optim.zero_grad()

            #
            # 2. Give LR input to GEN and get SR.
            #
            super_res = gen_2x(low_res)

            #
            # 2.1 Pass on SR to DIS with label 0. <Calc backprop?>
            #
            sr_output = dis(super_res)

            dis_loss = dis_criterion(sr_output, label_0)

            dis_loss.backward()
            dis_optim.step()

            #
            # 2.2 Pass HR to DIS with label 1. Calc backprop.
            #
            hr_output = dis(high_res)

            dis_optim.zero_grad()

            dis_loss = dis_criterion(hr_output, label_1)

            dis_loss.backward()
            dis_optim.step()

            #
            # 3. Set DIS to eval() and GEN to train().
            #
            gen_2x.train()
            dis.eval()

            gen_optim.zero_grad()
            dis_optim.zero_grad()

            #
            # 4. Pass LR input to GEN. Get SR.
            #
            super_res = gen_2x(low_res)

            #
            # 4.1 Calculate MSELoss() between SR and HR.
            #
            gen_mse_loss = gen_mse_criterion(super_res, high_res)

            #
            # 4.2 Pass on SR to DIS with label 1.
            #
            sr_output = dis(super_res)

            gen_nll_loss = dis_criterion(sr_output, label_1)

            #
            # 4.3 Add losses and backprop.
            #
            total_loss = gen_mse_loss + (gen_nll_loss * nll_loss_factor)

            total_loss.backward()
            gen_optim.step()

    train_time = time.time()
    print("\nTraining completed in {0} sec\n".format(train_time - start_time))
    # ----------------------------------------------------------------------------------
    gen_2x.eval()
    dis.eval()

    for i, (low_res, high_res) in enumerate(train_loader):

        output = gen_2x(low_res.to(device))

        image = output.detach().cpu().numpy()[0]
        ref = high_res[0].numpy()

        image = np.rollaxis(image, 0, 3)
        ref = np.rollaxis(ref, 0, 3)

        print("IMAGE:", image.shape)
        print("REF:", ref.shape)

        plt.figure()

        plt.subplot(2, 1, 1)

        plt.imshow(image)

        # plt.xlabel('time (s)')
        # plt.ylabel('voltage (mV)')
        plt.title('Super Resolution Image')

        plt.subplot(2, 1, 2)

        plt.imshow(ref)

        # plt.xlabel('time (s)')
        # plt.ylabel('voltage (mV)')
        plt.title('Original High Resolution Image')

        plt.show()


if __name__ == '__main__':
    main()
