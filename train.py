#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_clas_transformer.py
Created on Oct 08 2019 16:08

@author: Tu Bui tb0035@surrey.ac.uk
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models
import dataloaders

def main():
    parser = argparse.ArgumentParser(
        description='Train modified transformer with sketch data using PyTorch')
    parser.add_argument("model_name", help="Model that we are going to train")
    parser.add_argument("--dataset", default=None, help="Input data folder")
    parser.add_argument("-o", "--output_dir", default="", help="Output directory")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU ID to run on")
    parser.add_argument("-r", "--resume", default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--data_loader", default='stroke3', help="Data loader name")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")

    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load model
    ModelClass = getattr(models, args.model_name)
    model = ModelClass().to(device)

    # Load dataset and dataloader
    DatasetClass = getattr(dataloaders, args.data_loader)
    dataset = DatasetClass(args.dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

if __name__ == '__main__':
    main()
