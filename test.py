import ntpath
import os
import torch
from torch.functional import Tensor
import torch.nn.functional as F
from torch.types import Number
from tqdm import tqdm

from dice_loss import dice_coeff

from ignite.metrics import Recall, Accuracy

from scipy.spatial.distance import jaccard

from torchvision.utils import save_image


def engine(net, loader, device, output_dir, model_name):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()

    samples = len(loader)  # the number of batch
    
    dice_score = 0
    jaccard_score = 0

    recall_positives = Recall()
    recall_negatives = Recall()
    
    accuracy = Accuracy()

    with tqdm(total=samples, desc='Testing rodund', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, id = batch['image'], batch['mask'], batch['id'][0]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)


            output_size = list(true_masks.size())
            output_size = [output_size[2],output_size[3]]
            mask_pred = F.interpolate(mask_pred, output_size)



            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            
            save_prediction(pred, id, output_dir, model_name)

            dice_score += dice_coeff(pred, true_masks)

            pos_pred = (pred > 0).int()
            pos_masks = (true_masks > 0).int()

            state = pos_pred, pos_masks

            recall_positives.update(state)
            accuracy.update(state)

            jaccard_score += jaccard(pos_pred.cpu().flatten(), pos_masks.cpu().flatten())

            neg_pred = (pred <= 0).int()
            neg_masks = (true_masks <= 0).int()
            state = neg_pred, neg_masks

            recall_negatives.update(state)

            pbar.update()
    
    dice_score /= samples
    jaccard_score /= samples

    jaccard_score = 1 - jaccard_score
    
    print(f'Sensitivity: {recall_positives.compute()}')
    print(f'Specificity: {recall_negatives.compute()}')
    print(f'Accuracy: {accuracy.compute()}')
    print(f'Dice Score: {dice_score}')
    print(f'Jaccard index: {jaccard_score}')

    return 0

def save_prediction(pred, id, output_dir, model_name):
    outputDir = os.path.join(output_dir + '/' + model_name)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    pred = pred[:,:,:,:]
    output_path = os.path.join(outputDir, f'{id.strip(".jpg")}.png')
    save_image(pred, output_path)