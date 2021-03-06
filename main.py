import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim


from tqdm import tqdm
from eval import eval_net
from unet import Unet

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from melanomia_dataset import MelanomiaDataset
from datetime import datetime


dir_img = 'data/ISBI2016_ISIC_Part1_Training_Data/'
dir_mask = 'data/ISBI2016_ISIC_Part1_Training_GroundTruth/'




def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              time='',
              padding=0):


    dir_checkpoint = f'checkpoints/{time}LR_{lr}_BS_{batch_size}_SCALE_{img_scale}_PADDING{padding}'

    dataset = MelanomiaDataset(dir_img, dir_mask, img_scale)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}_PADDING{padding}')

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                image_batch = batch['image'].to(device=device, dtype=torch.float32)
                mask_batch = batch['mask'].to(device=device, dtype=torch.float32)

                predictions = net(image_batch)

                loss = criterion(predictions, mask_batch)
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (epoch)': epoch_loss})

                # training step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(image_batch.shape[0])

                global_step += 1

                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_images('images', image_batch, global_step)
                    writer.add_images('masks/true', mask_batch, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(predictions) > 0.5, global_step)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        # Write the total loss and eval score.
        val_dice_score, val_loss = eval_net(net, val_loader, device, criterion = criterion)

        writer.add_scalar('Loss/Train Epoch', epoch_loss/n_train, epoch + 1)
        writer.add_scalar('Loss/Validation Epoch', val_loss, epoch + 1)
        writer.add_scalar('Dice/Validation', val_dice_score, epoch + 1)

        scheduler.step(epoch_loss)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-p', '--padding', dest='padding', type=int, default=1,
                        help='Add padding in the convolutions')

    return parser.parse_args()


if __name__ == '__main__':

    # Firering up torch and check for GPU support
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H_%M_%S")
    logfile = "./logs/"+current_time+".log"

    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Loading Network
    net = Unet(addPadding=args.padding)

    # Loading saved model if defined
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    # Assign Model to device (CPU or GPU)
    net.to(device=device)

    # Start actual training loop
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  time = current_time)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
