import argparse
import os
from test import engine

import torch
from unet import Unet

from torch.utils.data.dataloader import DataLoader
from eval import eval_net

from melanomia_dataset import MelanomiaDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mp',
                        '--model_path',
                        type=str,
                        required = True,
                        dest='model_path')
    
    parser.add_argument('-i',
                        '--images',
                        type=str,
                        required = True,
                        dest='image_directory')
    
    parser.add_argument('-m',
                        '--masks',
                        type=str,
                        dest='mask_directory',
                        required = True)
    
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=True,
                        dest='output_directory',
                        help='Output destination of segmentations')
    
    parser.add_argument('-p',
                        '--padding',
                        type=bool,
                        default=True,
                        help='Add padding in the convolutions')
    
    parser.add_argument('-s',
                    '--scale',
                    type=float,
                    default='.5',
                    dest='scale',
                    help='Scaling of images')
    
    parser.add_argument('-u',
                    '--upscale',
                    type=bool,
                    default=True,
                    help='Upscaling of segmenatation to input resolution')
    
    return parser.parse_args()

def run_predictions(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MelanomiaDataset(args.image_directory,
                               args.mask_directory,
                               args.scale)
    loader = DataLoader(dataset, num_workers=1, pin_memory=True)
    
    net = Unet(addPadding=args.padding)
    net.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )
    net.to(device=device)
    
    print(engine(net, loader, device, args.output_directory, 'final_model'))

if __name__ == '__main__':
    args = parse_args()
    run_predictions(args)
    
