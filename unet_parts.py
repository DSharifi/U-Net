import torch
import torch.nn as nn

# Double ReLU convolution as applied after every 'vertical' step
def twoConvs(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=True), #TODO remove padding
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=True),
        nn.ReLU(inplace=True)
    )
