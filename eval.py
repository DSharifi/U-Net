import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

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

    with tqdm(total=n_val, desc=desc, unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, id = batch['image'], batch['mask'], batch['id'][0]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()

            if 'output_directory' in kwargs: # save output
                save_prediction(pred, id, **kwargs)

            pbar.update()

    net.train()
    return tot / n_val

def save_prediction(pred, id, **kwargs):
    output_path = os.path.join(kwargs['output_directory'], f'{id.strip(".jpg")}.png')
    save_image(pred, output_path)
    