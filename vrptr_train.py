#!/usr/bin/env python3
"""
VRPTR Training Script

Handles:
- Argument parsing
- Data loading
- Model creation
- Training loop
- Checkpointing
"""

import os
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from vrptr_dataset import VRPTRDataset
from vrptr_model import RPTR


def parse_train_arguments():
    """ Parse command-line arguments for training. """
    parser = argparse.ArgumentParser(description="Train the VRPTR model.")
    
    # Basic
    parser.add_argument('--experiment', default='vrptr_experiment', type=str,
                        help='Experiment name (used for logging/checkpoints)')
    
    # Data
    parser.add_argument('--root', default='/data/', type=str,
                        help='Root directory for data')
    parser.add_argument('--rsfc_dir', default='rsfc_d50_sample1', type=str,
                        help='Directory with rsFC .npy files')
    parser.add_argument('--contrast_dir', default='ts_obj', type=str,
                        help='Directory with contrast .npy files')
    parser.add_argument('--subj_list', default='subj_tumor.txt', type=str,
                        help='Text file listing subject IDs')
    
    # Training
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--gpu', default='0', type=str, help='Which GPU to use')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='Total epochs')
    parser.add_argument('--save_freq', default=100, type=int,
                        help='How often (in epochs) to save a checkpoint')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--load', action='store_true',
                        help='If set, loads from checkpoint in --resume')
    
    args = parser.parse_args()
    return args


def setup_logging(experiment_name):
    """ Set up a logger that prints to console and also saves to file. """
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    # Stream Handler (console)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    args = parse_train_arguments()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging setup
    logger = setup_logging(args.experiment)
    logger.info("===== VRPTR Training Start =====")
    logger.info(f"Arguments: {args}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load subject IDs
    subj_list_path = os.path.join(args.root, args.subj_list)
    subj_ids = np.loadtxt(subj_list_path, dtype=str)
    
    # Build Dataset and DataLoader
    train_dataset = VRPTRDataset(
        subj_ids=subj_ids,
        rsfc_dir=os.path.join(args.root, args.rsfc_dir),
        contrast_dir=os.path.join(args.root, args.contrast_dir),
        num_samples=1  # Adjust if you have multiple samples
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    
    # Create Model
    # Example: in_ch=100, out_ch=2, adjust as needed
    model = RPTR(mesh_dir='data/fs_LR_mesh_templates', in_ch=100, out_ch=2)
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.load and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    else:
        logger.info("Starting from scratch...")
    
    # Create checkpoint directory
    ckpt_dir = os.path.join('checkpoints', args.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Training loop
    logger.info("===== Beginning Training =====")
    total_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time_epoch = time.time()
        
        for batch_data in train_loader:
            x, target = batch_data
            x = x.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            
            # If the model tracks a KL term (VAE), incorporate it here
            loss = criterion(y_pred, target) + 0.0 * model.kl
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        logger.info(f"Epoch {epoch}/{args.epochs - 1} - Loss: {epoch_loss:.5f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")
        
        # Timing
        end_time_epoch = time.time()
        epoch_mins = (end_time_epoch - start_time_epoch) / 60.0
        remaining_hours = (args.epochs - epoch - 1) * epoch_mins / 60.0
        logger.info(f"Epoch time: {epoch_mins:.2f} minutes - "
                    f"Estimated remaining: {remaining_hours:.2f} hours")
    
    # Final save
    final_ckpt = os.path.join(ckpt_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    }, final_ckpt)
    logger.info(f"Final checkpoint saved: {final_ckpt}")
    
    total_hours = (time.time() - total_start_time) / 3600.0
    logger.info(f"Total training time: {total_hours:.2f} hours")
    logger.info("===== Training Finished =====")


if __name__ == "__main__":
    main()
