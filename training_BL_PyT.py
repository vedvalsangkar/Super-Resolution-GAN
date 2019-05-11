# https://github.com/aitorzip/PyTorch-SRGAN

import torch
from torch import nn, cuda, optim
from torch.utils.data import DataLoader
import torchvision.models as models
from PIL import Image
from torchsummary import summary

from matplotlib import pyplot as plt
import time

import numpy as np
import argparse as ap

from baseline_models import Generator, Discriminator
from datasets import DIVFlickrDataSet


def main(input_args):
    # -------------------------------- Hyper-parameters --------------------------------

    verbose = True
    # verbose = input_args.verbose

    img_size = (224, 224)

    bias = False

    learning_rate = input_args.lr
    lamb = input_args.weight_decay
    epochs = input_args.num_epochs

    batch_size = input_args.batch_size
    batch_print = 20

    nll_loss_factor = 0.001

    sgd_momentum = 0.3

    op_dir = "models/"
    # ----------------------------------------------------------------------------------

    # --------------------------------- Fetching model ---------------------------------

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    if verbose:
        pass

    # training_set = Set5DataSet(im_set=4)
    training_set = DIVFlickrDataSet(root_folder="Data/DIV_Flickr/train/")

    train_loader = DataLoader(dataset=training_set,
                              batch_size=batch_size,
                              shuffle=True)

    testing_set = DIVFlickrDataSet(root_folder="Data/DIV_Flickr/val/",
                                   test_mode=True)

    test_loader = DataLoader(dataset=testing_set,
                             batch_size=1,
                             shuffle=True)

    if verbose:
        print("Dataset and data loaders acquired.")

    gen_4x = Generator(upscale_factor=2, bias=bias).to(device)
    dis = Discriminator(image_size=img_size, bias=bias).to(device)

    gen_optim = optim.Adam(params=gen_4x.parameters(),
                           lr=learning_rate,
                           weight_decay=lamb,
                           amsgrad=False
                           )

    dis_optim = optim.Adam(params=dis.parameters(),
                           lr=learning_rate,
                           weight_decay=lamb,
                           amsgrad=False
                           )

    gen_mse_criterion = nn.MSELoss()
    # vgg_mse_criterion = nn.MSELoss()
    # gen_nll_criterion = nn.NLLLoss()
    # dis_criterion = nn.NLLLoss()
    dis_criterion = nn.BCELoss()

    percept_model = models.vgg19(pretrained=True).to(device).features

    percept_model.eval()

    # ----------------------------------------------------------------------------------

    # ------------------------------- Start of training --------------------------------
    if verbose:
        print("\nStart of training.")
        print("Total epochs: {0}".format(epochs))
        print("Start time:", time.asctime())

        summary(gen_4x, (3, 56, 56), batch_size)
        summary(dis, (3, 224, 224), batch_size)
        summary(percept_model, (3, 224, 224), batch_size)

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

    # label_0 = torch.tensor(0.0).expand(batch_size, 1).to(device)
    # label_1 = torch.tensor(1.0).expand(batch_size, 1).to(device)

    # print("LABEL_0:", label_0)

    disp_loss_dis_bce = 0
    disp_loss_gen_class = 0
    disp_loss_gen_feat = 0
    disp_total_loss = 0

    tot_len = len(train_loader)

    for epoch in range(epochs):

        if verbose:
            print("\nTime:", time.asctime())

        for i, (low_res, high_res, label_0, label_1) in enumerate(train_loader):
            try:
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                label_0 = label_0.to(device).view(-1, 1)
                label_1 = label_1.to(device).view(-1, 1)

                #
                # 1. Set generator to eval() and discriminator to train().
                #
                gen_4x.eval()
                dis.train()

                gen_optim.zero_grad()
                dis_optim.zero_grad()

                #
                # 2. Give LR input to GEN and get SR.
                #
                super_res = gen_4x(low_res)

                #
                # 2.1 Pass on SR to DIS with label 0. <Calc backprop?>
                #
                sr_output = dis(super_res)

                dis_loss = dis_criterion(sr_output, label_0)
                disp_loss_dis_bce += dis_loss.item()

                dis_loss.backward()
                dis_optim.step()

                #
                # 2.2 Pass HR to DIS with label 1. Calc backprop.
                #
                hr_output = dis(high_res)

                dis_optim.zero_grad()

                dis_loss = dis_criterion(hr_output, label_1)
                disp_loss_dis_bce += dis_loss.item()

                dis_loss.backward()
                dis_optim.step()

                # ---------------------------------------------------------------------------------------------------------
                #
                # 3. Set DIS to eval() and GEN to train().
                #
                gen_4x.train()
                dis.eval()

                gen_optim.zero_grad()
                dis_optim.zero_grad()

                #
                # 4. Pass LR input to GEN. Get SR.
                #
                super_res = gen_4x(low_res)

                #
                # 4.1 Calculate MSELoss() between SR and HR.
                # This step can be pixel-wise loss or perceptual loss.
                #

                # gen_feat_loss = gen_mse_criterion(super_res, high_res)
                with torch.no_grad():
                    sr_features = percept_model(super_res)
                    hr_features = percept_model(high_res)

                gen_feat_loss = gen_mse_criterion(sr_features, hr_features)
                disp_loss_gen_feat += gen_feat_loss.item()

                #
                # 4.2 Pass on SR to DIS with label 1.
                #
                sr_output = dis(super_res)

                gen_class_loss = dis_criterion(sr_output, label_1)
                disp_loss_gen_class += gen_class_loss.item()

                #
                # 4.3 Add losses and backprop.
                #
                total_loss = gen_feat_loss + (gen_class_loss * nll_loss_factor)
                disp_total_loss += total_loss.item()

                total_loss.backward()
                gen_optim.step()

                if (i + 1) % batch_print == 0:
                    disp_loss_dis_bce /= batch_print
                    disp_loss_gen_class /= batch_print
                    disp_loss_gen_feat /= batch_print
                    disp_total_loss /= batch_print
                    print(
                        "\rEpoch: {4}, Batch: {5}/{6} || DIS Loss = {0:.4f}, Gen CLS Loss = {1:.4f}, Gen MSE Loss= {2:.4f}, Total Gen Loss: {3:.4f}        ".format(
                            disp_loss_dis_bce, disp_loss_gen_class, disp_loss_gen_feat, disp_total_loss, epoch, i + 1,
                            tot_len), end="")

                    disp_loss_dis_bce = 0
                    disp_loss_gen_class = 0
                    disp_loss_gen_feat = 0
                    disp_total_loss = 0
            except FileNotFoundError as e:
                print(i, "EXCEPTION:", e.__cause__)

    train_time = time.time()
    if verbose:
        print("\nTraining completed in {0} sec\n".format(train_time - start_time))
    # ----------------------------------------------------------------------------------
    gen_4x.eval()
    dis.eval()

    for i, (low_res, high_res, name) in enumerate(test_loader):
        output = gen_4x(low_res.to(device))

        image = output.detach().cpu().numpy()[0]
        ref = high_res[0].numpy()

        image = np.rollaxis(image, 0, 3)
        ref = np.rollaxis(ref, 0, 3)

        # print("IMAGE:", image.shape)
        # print("REF:", ref.shape)

        plt.figure()

        plt.subplot(1, 2, 1)

        plt.imshow(image)

        # plt.xlabel('time (s)')
        # plt.ylabel('voltage (mV)')
        plt.title('Super Resolution Image')

        plt.subplot(1, 2, 2)

        plt.imshow(ref)

        # plt.xlabel('time (s)')
        # plt.ylabel('voltage (mV)')
        plt.title('Original High Resolution Image')

        # plt.show()
        plt.savefig("Output Images/SR_{0}.jpg".format(name))


if __name__ == '__main__':

    parser = ap.ArgumentParser()

    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=1,
                        help="Batch Size (default 2)."
                        )

    parser.add_argument("-e",
                        "--num-epochs",
                        type=int,
                        default=10,
                        help="Number of epochs (default 3)."
                        )

    parser.add_argument("--lr",
                        "--learning-rate",
                        type=float,
                        default=0.001,
                        help="Initial learning rate (default 0.01)"
                        )

    parser.add_argument("-d",
                        "--lr-decay",
                        type=float,
                        default=0.995,
                        help="Learning rate decay (default 0.95)"
                        )

    parser.add_argument("-w",
                        "--weight-decay",
                        type=float,
                        default=0.995,
                        help="Learning rate decay (default 0.95)"
                        )

    parser.add_argument("-v",
                        "--verbose",
                        help="Verbose Output",
                        action="store_true"
                        )

    args = parser.parse_args()

    main(args)