import os
import torch
from torch.functional import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import ntpath

import numpy as np

from dice_loss import dice_coeff

from torchvision.utils import save_image


def eval_net(net, loader, device, **kwargs):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    if 'desc' in kwargs:
        desc = kwargs['desc']
    else:
        desc = 'Validation round'

    loss = 0

    booleanTOT = 0

    with tqdm(total=n_val, desc=desc, unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, id = batch['image'], batch['mask'], batch['id'][0]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks)

            if 'output_directory' in kwargs: # save output
                save_prediction(pred, id, **kwargs)

            if 'criterion' in kwargs:
                loss += kwargs['criterion'](pred, true_masks).item()

            # print(f'')

            pbar.update()

    if 'criterion' in kwargs:
        return tot/n_val, loss/n_val

    return tot / n_val

def save_prediction(pred, id, **kwargs):
    outputDir = os.path.join(kwargs['output_directory'] + '/' + ntpath.basename(kwargs['model_path']))
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    pred = pred[:,:,:,:]
    output_path = os.path.join(outputDir, f'{id.strip(".jpg")}.png')
    save_image(pred, output_path)


# def DSC(prediction: Tensor, ground_truth: Tensor) -> float:
    
#     #assert prediction.shape() == ground_truth.shape()
    
#     # intersection = 1,1 ; 0,0
#     # intersection = |set| - |xor|
#     intersection = prediction.numel() - torch.sum(torch.logical_xor(prediction, ground_truth))

#     numerator   = 2 * intersection
#     denominator = prediction.numel() + ground_truth.numel()

#     return numerator / denominator

# def booleanDSC(prediction, ground_truth) -> float:
    
#     # false_positives = 1,1
#     true_positives = sum(torch.logical_and(prediction, ground_truth))
    
#     # false_positives = 1,0
#     # false_negative  = 0,1
#     # XOR             = false_positives + false_negatives
#     xor = torch.sum(torch.logical_xor(prediction, ground_truth))

#     numerator   = 2 * true_positives
#     denominator = 2 * true_positives + xor
    
    
#     return numerator / denominator