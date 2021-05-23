import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from helpers import plot_imgs, crop_feature_map
from unet_parts import twoConvs
from image_processor import preprocess_image

class Unet(nn.Module):

    def __init__(self, addPadding=False, in_channels=3,):
        super(Unet, self).__init__()

        # From the paper "2x2 max pooling operation with 2 stride"
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.n_channels = in_channels

        # From the paper, each of the bellow:
        # 1. 3x3 unpadded convolution
        # 2. ReLU
        # 3. 3x3 unpadded convolution
        # 4. ReLU
        self.down_conv_1 = twoConvs(in_channels, 64,addPadding)
        self.down_conv_2 = twoConvs(64, 128,addPadding)
        self.down_conv_3 = twoConvs(128, 256,addPadding)
        self.down_conv_4 = twoConvs(256, 512,addPadding)
        self.down_conv_5 = twoConvs(512, 1024,addPadding)

        # NOTE: Maybe use stride=2  but it is not specified in architecture.
        # NOTE: https://discuss.pytorch.org/t/torch-nn-convtranspose2d-vs-torch-nn-upsample/30574
        #
        # UP CONVs
        self.up_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = twoConvs(1024, 512,addPadding)
        self.up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = twoConvs(512, 256,addPadding)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = twoConvs(256, 128,addPadding)
        self.up_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = twoConvs(128, 64,addPadding)

        self.output = nn.Conv2d(64, 1, kernel_size=1)




    def forward(self, img):
        ########################
        ### Contracting Path ###
        ########################

        # TODO: Remove below

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

        #print("Before first up: ",x_9.size())
        x_10 = self.up_1(x_9)
        #print("After transpose2D: ",x_10.size())
        x_7_crop = crop_feature_map(x_7, x_10.size()[2], x_10.size()[3])
        y_1 = torch.cat([x_7_crop, x_10], 1)
        #print("After copy and crop: ",y_1.size())
        x_11 = self.up_conv_1(y_1)
        #print("After twoConv(kernel=3x3): ",x_11.size())

        x_12 = self.up_2(x_11)
        x_5_crop = crop_feature_map(x_5, x_12.size()[2], x_12.size()[3])
        y_2 = torch.cat([x_5_crop, x_12], 1)
        x_13 = self.up_conv_2(y_2)

        x_14 = self.up_3(x_13)
        x_3_crop = crop_feature_map(x_3, x_14.size()[2], x_14.size()[3])
        y_3 = torch.cat([x_3_crop, x_14], 1)
        x_15 = self.up_conv_3(y_3)

        x_16 = self.up_4(x_15)
        x_1_crop = crop_feature_map(x_1, x_16.size()[2], x_16.size()[3])
        y_4 = torch.cat([x_1_crop, x_16], 1)
        x_17 = self.up_conv_4(y_4)

        y_final = self.output(x_17)




        return y_final


if __name__ == "__main__":
    # Tensor shape = (1, #filters, pxWidth, pxHeight)

    img = torch.rand((1, 1, 1022,676))

    img_1 = Image.open("ISIC_0000003.jpg")
    img_1 = preprocess_image(img_1, scale=1)

    model = Unet()
    res = model(img_1)

    print(img_1.size())
    print(res.size())
