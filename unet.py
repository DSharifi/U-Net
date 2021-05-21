import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from helpers import plot_imgs, crop_feature_map
from unet_parts import twoConvs
from image_processor import preprocess_image

class Unet(nn.Module):

    def __init__(self, in_channels=3):
        super(Unet, self).__init__()

        # From the paper "2x2 max pooling operation with 2 stride"
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # From the paper, each of the bellow:
        # 1. 3x3 unpadded convolution
        # 2. ReLU
        # 3. 3x3 unpadded convolution
        # 4. ReLU
        self.down_conv_1 = twoConvs(in_channels, 64)
        self.down_conv_2 = twoConvs(64, 128)
        self.down_conv_3 = twoConvs(128, 256)
        self.down_conv_4 = twoConvs(256, 512)
        self.down_conv_5 = twoConvs(512, 1024)

        # NOTE: Maybe use stride=2  but it is not specified in architecture.
        # NOTE: https://discuss.pytorch.org/t/torch-nn-convtranspose2d-vs-torch-nn-upsample/30574
        #
        # UP CONV
        self.up_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2)
        self.up_conv_1 = twoConvs(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2)
        self.up_conv_2 = twoConvs(512, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2)
        self.up_conv_3 = twoConvs(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, kernel_size=2)
        self.up_conv_4 = twoConvs(128, 64)



    def forward(self, img):
        ########################
        ### Contracting Path ###
        ########################

        # TODO: Remove below
        img1 = img.reshape(767, 1022, 3).detach().numpy()

        # Down Step 1
        x_1 = self.down_conv_1(img)                 # TODO: Copy & Crop 1

        # plot_imgs(img1, img2)
        x_2 = self.maxPool(x_1)

        # Down Step 2
        x_3 = self.down_conv_2(x_2)                 # TODO: Copy & Crop 2
        x_4 = self.maxPool(x_3)

        # Down Step 3
        x_5 = self.down_conv_3(x_4)                 # TODO: Copy & Crop 3
        x_6 = self.maxPool(x_5)

        # Down Step 5
        x_7 = self.down_conv_4(x_6)                 # TODO: Copy & Crop 4
        x_8 = self.maxPool(x_7)

        # "Horizontal" Step
        x_9 = self.down_conv_5(x_8)

        ########################
        #### Expanding Path ####
        ########################
        x_10 = self.up_1(x_9)
        print()
        print("x_10: " + str(x_10.size()))
        print("x_7 : " + str(x_7.size()))
        x_7_crop = crop_feature_map(x_7, x_10.size()[2], x_10.size()[3])
        y_1 = torch.cat([x_7_crop, x_10], 1)
        x_11 = self.up_conv_1(y_1)
        print("x_11: " + str(x_10.size()))
        print()
    
        x_12 = self.up_2(x_11)
        print()
        print("x_12: " + str(x_12.size()))
        x_5_crop = crop_feature_map(x_5, x_12.size()[2], x_12.size()[3])
        y_2 = torch.cat([x_5_crop, x_12], 1)
        x_13 = self.up_conv_2(y_2)
        print("x_13: " + str(x_13.size()))
        print()

        x_14 = self.up_3(x_13)
        print()
        print("x_14: " + str(x_14.size()))
        x_3_crop = crop_feature_map(x_3, x_14.size()[2], x_14.size()[3])
        y_3 = torch.cat([x_3_crop, x_14], 1)
        x_15 = self.up_conv_3(y_3)
        print("x_15: " + str(x_15.size()))
        print()

        x_16 = self.up_4(x_15)
        print()
        print("x_16: " + str(x_16.size()))
        x_1_crop = crop_feature_map(x_1, x_16.size()[2], x_16.size()[3])
        y_4 = torch.cat([x_1_crop, x_16], 1)
        x_17 = self.up_conv_4(y_4)
        print("x_17: " + str(x_17.size()))
        print()

        plot_imgs(img1, img1)


        return x_17


if __name__ == "__main__":
    # Tensor shape = (1, #filters, pxWidth, pxHeight)

    img = torch.rand((1, 1, 1022,676))

    img_1 = Image.open("ISIC_0000003.jpg")
    img_1 = preprocess_image(img_1, scale=1)

    model = Unet()
    res = model(img_1)

    print("Output Img: " + str(res.shape))

    exit()

    # Prpare result img
    res = torch.reshape(res, (64*56, 16*34))
    res = res.detach().numpy()

    plot_imgs(Image.open("ISIC_0000003.jpg"), res)

