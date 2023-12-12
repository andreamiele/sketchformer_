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
from models.sketchformer import Transformer, Sketchformer
from dataloaders.distributed_stroke3 import DistributedStroke3Dataset

def pad_collate(batch):
    """
    A custom collate function to pad sketches to the same length.
    """
    max_len = max(x.shape[0] for x, _ in batch)
    padded_sketches = []
    labels = []
    for sketch, label in batch:
        padded_sketch = torch.zeros(max_len, sketch.shape[1])
        padded_sketch[:sketch.shape[0], :] = sketch
        padded_sketches.append(padded_sketch)
        labels.append(label)
    return torch.stack(padded_sketches, dim=0), torch.tensor(labels)


def train(model, dataloader, epochs, learning_rate):
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for sketches, labels in dataloader:
            optimizer.zero_grad()
            dummy_tgt = torch.zeros_like(sketches)
            outputs = model(sketches,dummy_tgt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def main():
    # Hyperparameters
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

   
    # Dataset and DataLoader
    dataset = DistributedStroke3Dataset(data_directory='/content/quickdraw_prepared', use_continuous_data=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    
    unique_tokens = set()

    for data in dataloader:
        sketches = data[0]  # Assuming sketches are the first element in the tuple
        max_index = sketches.max()
        if max_index >= vocab_size:
            print(f"Found token index {max_index} which exceeds vocab size")
            break
    # Example hyperparameters for the Transformer model
        # Define hyperparameters
    VOCAB_SIZE = 10000  # The size of your vocabulary
    NUM_LAYERS = 6      # Number of Transformer encoder and decoder layers
    D_MODEL = 512       # Dimensionality of the Transformer layers
    N_HEADS = 8         # Number of heads in the multi-head attention mechanisms
    DIM_FEEDFORWARD = 2048  # Dimensionality of the feed-forward network in the Transformer layers
    MAX_SEQ_LENGTH = 500    # Maximum sequence length of your sketches
    DROPOUT_RATE = 0.1  # Dropout rate

    # Instantiate the Sketchformer model
    model = Sketchformer(vocab_size=VOCAB_SIZE, 
                                    num_layers=NUM_LAYERS, 
                                    d_model=D_MODEL, 
                                    nhead=N_HEADS, 
                                    dim_feedforward=DIM_FEEDFORWARD, 
                                    max_seq_length=MAX_SEQ_LENGTH, 
                                    dropout=DROPOUT_RATE)
    # Train the model
    train(model, dataloader, epochs, learning_rate)

if __name__ == '__main__':
    main()
