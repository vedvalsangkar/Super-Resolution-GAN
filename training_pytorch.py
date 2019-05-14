# https://github.com/aitorzip/PyTorch-SRGAN

import torch
from torch import nn, cuda, optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchsummary import summary
from torchvision.utils import save_image

import time
import argparse as ap
import pandas as pd
from os import mkdir

from baseline_models import Generator, Discriminator
from enhancement_models import Generator2
from datasets import DIVFlickrDataSet


def main(input_args):
    """
    Main function.

    :param input_args: Commandline inputs
    :return: N/A
    """
    # -------------------------------- Hyper-parameters --------------------------------

    # verbose = True
    verbose = input_args.verbose

    use_enhancement = args.enhancement
    use_vgg = args.vgg
    use_hybrid = args.hybrid
    use_kl = args.kl

    vgg_loss_factor = 0.01

    img_size = (224, 224)

    bias = False

    learning_rate = input_args.lr
    lamb = input_args.weight_decay
    epochs = input_args.num_epochs

    batch_size = input_args.batch_size
    batch_print = 20

    adv_loss_factor = input_args.adv_loss_factor

    sgd_momentum = 0.1
    sgd_nesterov = False

    timestamp = time.strftime("%Y%m%d:%H%M%S")
    op_dir = "output/{0}_{1}/".format(timestamp, "VGG" if use_vgg else "RAW")
    op_models_dir = "models/"

    offset = 0.1 if args.soft_labels else 0.0
    # ----------------------------------------------------------------------------------

    # --------------------------------- Fetching model ---------------------------------

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

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

    if use_enhancement:
        gen_4x = Generator2(upscale_factor=2, bias=bias).to(device)
    else:
        gen_4x = Generator(upscale_factor=2, bias=bias).to(device)
    dis = Discriminator(image_size=img_size, bias=bias).to(device)

    gen_optim = optim.Adam(params=gen_4x.parameters(),
                           lr=learning_rate,
                           weight_decay=lamb,
                           amsgrad=False
                           )

    dis_optim = optim.SGD(params=dis.parameters(),
                          lr=learning_rate,
                          momentum=sgd_momentum,
                          weight_decay=lamb,
                          nesterov=sgd_nesterov
                          )

    if use_kl:
        gen_content_criterion = nn.KLDivLoss()
    else:
        gen_content_criterion = nn.MSELoss()

    dis_criterion = nn.BCELoss()

    if use_vgg:
        percept_model = nn.Sequential(models.vgg19(pretrained=True).to(device).features[:-1])
        percept_model.train(False)

    # ----------------------------------------------------------------------------------

    # ------------------------------- Start of training --------------------------------
    if verbose:
        summary(gen_4x, (3, 56, 56), batch_size)
        summary(dis, (3, 224, 224), batch_size)
        if use_vgg:
            summary(percept_model, (3, 224, 224), batch_size)

        print("\nStart of training.")
        print("Total epochs: {0}".format(epochs))
        print("Start time:", time.asctime())

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

    disp_loss_dis_bce = 0
    disp_loss_gen_class = 0
    disp_loss_gen_feat = 0
    disp_total_loss = 0

    tot_len = len(train_loader)

    printer = []

    for epoch in range(epochs):

        print("")

        for i, (low_res, high_res, label_0, label_1) in enumerate(train_loader):
            # try:
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

            dis_loss = dis_criterion(sr_output, label_0 + offset)
            disp_loss_dis_bce += dis_loss.item()

            dis_loss.backward()
            dis_optim.step()

            #
            # 2.2 Pass HR to DIS with label 1. Calc backprop.
            #
            hr_output = dis(high_res)

            dis_optim.zero_grad()

            dis_loss = dis_criterion(hr_output, label_1 - offset)
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

            if use_vgg:
                sr_features = percept_model(super_res)
                hr_features = percept_model(high_res)

                gen_feat_loss = gen_content_criterion(sr_features, hr_features)

                if use_hybrid:

                    gen_feat_loss = vgg_loss_factor * gen_feat_loss + gen_content_criterion(super_res, high_res)

            else:
                gen_feat_loss = gen_content_criterion(super_res, high_res)

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
            total_loss = gen_feat_loss + (gen_class_loss * adv_loss_factor)
            disp_total_loss += total_loss.item()

            total_loss.backward()
            gen_optim.step()

            if (i + 1) % batch_print == 0:
                disp_loss_dis_bce /= batch_print
                disp_loss_gen_class /= batch_print
                disp_loss_gen_feat /= batch_print
                disp_total_loss /= batch_print
                print(
                    "\rEpoch: {4}, Batch: {5}/{6} || DIS Loss = {0:.4f}, Gen CLS Loss = {1:.4f}, Gen MSE Loss= {2:.4f}, Total Gen Loss: {3:.4f}    ".format(
                        disp_loss_dis_bce, disp_loss_gen_class, disp_loss_gen_feat, disp_total_loss, epoch + 1, i + 1,
                        tot_len), end="")

                printer.append([epoch, i, disp_loss_dis_bce, disp_loss_gen_class, disp_loss_gen_feat, disp_total_loss])

                disp_loss_dis_bce = 0
                disp_loss_gen_class = 0
                disp_loss_gen_feat = 0
                disp_total_loss = 0
            # except FileNotFoundError as e:
            #     print(i, "EXCEPTION:", e.__cause__)

    train_time = time.time()
    if verbose:
        print("\nTraining completed in {0} sec\n".format(train_time - start_time))
    # ----------------------------------------------------------------------------------
    gen_4x.eval()
    dis.eval()

    torch.save(obj={"generator": gen_4x.state_dict(),
                    "discriminator": dis.state_dict()
                    },
               f=op_models_dir + "file_{0}_{1}.pt".format(timestamp, "VGG" if use_vgg else "RAW")
               )

    mkdir(op_dir)

    pd.DataFrame(data=printer,
                 columns=["Epoch", "Step", "Discriminator Loss", "Generator Classification Loss",
                          "Generator MSE Loss", "Generator Loss"]
                 ).to_csv(path_or_buf=op_dir + "data_{0}.csv".format(timestamp))

    for i, (low_res, high_res, name) in enumerate(test_loader):
        output = gen_4x(low_res.to(device))

        image = output.detach().cpu()[0]
        ref = high_res[0]

        save_image(tensor=torch.stack([image, ref]),
                   filename=op_dir+"SR_{0}.jpg".format(name[0]),
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


if __name__ == '__main__':

    parser = ap.ArgumentParser()

    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=16,
                        help="Batch Size (default 4)."
                        )

    parser.add_argument("-e",
                        "--num-epochs",
                        type=int,
                        default=10,
                        help="Number of epochs (default 10)."
                        )

    parser.add_argument("--lr",
                        "--learning-rate",
                        type=float,
                        default=0.001,
                        help="Initial learning rate (default 0.001)"
                        )

    parser.add_argument("-d",
                        "--lr-decay",
                        type=float,
                        default=0.995,
                        help="Learning rate decay (default 0.995)"
                        )

    parser.add_argument("-w",
                        "--weight-decay",
                        type=float,
                        default=0,
                        help="L2 Norm (default 0)"
                        )

    parser.add_argument("-v",
                        "--verbose",
                        help="Verbose Output",
                        action="store_true"
                        )

    parser.add_argument("--vgg",
                        help="Use VGG to calculate features before MSE loss",
                        action="store_true"
                        )

    parser.add_argument("--adv-loss-factor",
                        type=float,
                        default=0.001,
                        help="Factor to determine effect of adversarial loss (default 0.001)"
                        )

    parser.add_argument("--vgg-loss-factor",
                        type=float,
                        default=0.01,
                        help="Factor to determine effect of content loss from VGG (default 0.01)"
                        )

    parser.add_argument("--soft-labels",
                        help="Use 0.9 and 0.1 instead of 1 and 0 resp. to introduce noise in discriminator",
                        action="store_true"
                        )

    parser.add_argument("--hybrid",
                        help="Use both image and feature MSE losses",
                        action="store_true"
                        )

    parser.add_argument("--enhancement",
                        help="Use enhanced model (with squeeze and excitation model) instead",
                        action="store_true"
                        )

    parser.add_argument("--kl",
                        help="Use KL divergence loss instead",
                        action="store_true"
                        )

    args = parser.parse_args()

    main(args)
