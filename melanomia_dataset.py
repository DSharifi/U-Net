from os import listdir
from os.path import join
from typing import Dict, Tuple
import numpy as np
from numpy.core.numeric import outer
import torch
from torch import Tensor

from torch.utils.data import Dataset
from PIL.Image import open, Image


class MelanomiaDataset(Dataset):
    def __init__(self, image_directory, mask_directory, scale = 1, scale_mask = True):
        self.scale = scale

        self.scale_mask = scale_mask
        
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        
        self.images = listdir(image_directory)
        self.masks = listdir(mask_directory)
        
        self.images.sort()
        self.masks.sort()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Returns image and mask at given index.

        Args:
            idx (int): index

        Returns:
            Tuple[Tensor, Tensor]: (image, mask) pair.
        """
        image_path = join(self.image_directory, self.images[idx])
        mask_path = join(self.mask_directory, self.masks[idx])
        
        with open(image_path) as image, open(mask_path) as mask:
            output = self.scale_file_(image, mask)
        
        output['id'] = self.images[idx]
        
        return output
    
    def scale_file_(self, image: Image, mask: Image) -> Tuple[Tensor, Tensor]:
        """Takes in image, mask pointer pair and returns
        scaled versions as tensors.

        Args:
            image (Image): image file
            mask (Image): mask file

        Returns:
            Tuple[Tensor, Tensor]: tensor versions of image and mask.
        """
        width, height = image.size
        width, height = int(width*self.scale), int(height*self.scale)
        
        image = image.resize((width, height))

        if self.scale_mask:
            mask = mask.resize((width, height))

        image, mask = np.array(image), np.array(mask)
        
        mask = np.expand_dims(mask, axis=2)
        
        # HWC to CHW
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        
        image = image / 255
        mask = mask / 255
        
        # TODO: Convert mask to IntTensor? (all values are 0, 1)
        return {"image":torch.from_numpy(image).type(torch.FloatTensor),
                "mask":torch.from_numpy(mask).type(torch.FloatTensor)}