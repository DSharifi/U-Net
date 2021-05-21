from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import matplotlib.pyplot as plt

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

def crop_feature_map(tensor, target_height, target_width):

    curr_height = tensor.size()[2]
    curr_width = tensor.size()[3]

    if target_height > curr_height or target_width > curr_width:
        raise Exception("Target size must not be greater than feature map size of tensor.")

    x_margin = (curr_width - target_width) // 2
    y_margin = (curr_height - target_height) // 2

    from_y, to_y = y_margin, curr_height - y_margin
    from_x, to_x = x_margin, curr_width - x_margin

    if target_height < to_y - from_y:
        to_y -= 1
    if target_width < to_x - from_x:
        to_x -= 1

    cropped_tensor = tensor[:, :, from_y:to_y, from_x:to_x]

    return cropped_tensor


    

def plot_imgs(img_left, img_right):
    _, ax = plt.subplots(1,2, constrained_layout=True, figsize= (10,10))

    ax[0].set_title('Input Image')
    ax[0].imshow(img_left)

    ax[1].set_title('Output Image')
    ax[1].imshow(img_right)

    plt.show()
