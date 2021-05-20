import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from helpers import plot_imgs
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



    def forward(self, img):
        print("In: " + str(img.size()))

        ########################
        ### Contracting Path ###
        ########################

        # Down Step 1
        x_1 = self.down_conv_1(img)                 # TODO: Copy & Crop 1
        print("First down: " + str(x_1.size()))
        print("img       : " + str(img.size()))
        print(img[0][0][0][0])
        img1 = img[0].detach().reshape((767, 1022, 3)).numpy()
        img1 = Image.fromarray(img1, 'RGB')
        img2 = x_1[0][24].detach().reshape((763, 1018)).numpy()
    
        plot_imgs(img1, img2)
        x_2 = self.maxPool(x_1)
        print("First maxPool: " + str(x_2.size()))

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
        print("Last Contraction: " + str(x_9.size()))

        ########################
        #### Expanding Path ####
        ########################

        # TODO

        return x_9


if __name__ == "__main__":
    # Tensor shape = (1, #filters, pxWidth, pxHeight)

    img = torch.rand((1, 1, 1022,676))

    img_1 = Image.open("ISIC_0000003.jpg")
    img_1 = preprocess_image(img_1, scale=1)

    model = Unet()
    res = model(img_1)
    print("Output Img: " + str(res.shape))

    # Prpare result img
    res = torch.reshape(res, (64*56, 16*34))
    res = res.detach().numpy()

    plot_imgs(Image.open("ISIC_0000003.jpg"), res)

