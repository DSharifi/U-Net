import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def scale_img(original_img, scale):
    original_width, original_height = original_img.size

    scaled_width = int(scale * original_width)
    scaled_height = int(scale * original_height)

    scaled_img = original_img.resize((scaled_width, scaled_height))

    return scaled_img

def crop_image(original_img, width, height):

    # TODO
    cropped_img = original_img

    return cropped_img


def preprocess_image(img, scale=1.0, channels=3):

    if  scale <= 0 or scale < 1:
        raise Exception("Scale must be greater than 0 and at most 1.0")
    if scale < 1:
        img = scale_img(img, scale)
    
    width, height = img.size
    img = np.array(img).reshape(1, channels, width, height)

    img = img / 255                 # Map RGB to [0,1]
    img = torch.from_numpy(img).type(torch.FloatTensor)

    return img

