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

import torch
from torch.utils.data import DataLoader
from models.sketchformer import Transformer
from dataloaders.distributed_stroke3 import DistributedStroke3Dataset

def train(model, dataloader, epochs, learning_rate):
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for sketches, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sketches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def main():
    # Hyperparameters
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

    
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = 1000  # Example size, change as per your dataset
    target_vocab_size = 1000 # Example size, change as per your dataset
    dropout_rate = 0.1
    max_seq_len = 200

   
    # Dataset and DataLoader
    dataset = DistributedStroke3Dataset('path/to/your/npz/files')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = Transformer(num_layers, d_model, dff, num_heads, input_vocab_size, target_vocab_size, dropout_rate, max_seq_len)

    # Train the model
    train(model, dataloader, epochs, learning_rate)

if __name__ == '__main__':
    main()
