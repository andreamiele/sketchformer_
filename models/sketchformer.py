"""
sketchformer.py
Created on Oct 06 2019 16:34
ref: https://www.tensorflow.org/tutorials/text/transformer
"""


import numpy as np

import builders
from utils.hparams import HyperParams as HParams
from core.models import BaseModel
from .evaluation_mixin import TransformerMetricsMixin
from builders.layers.transformer import (Encoder, Decoder, DenseExpander)

import torch
import torch.nn as nn
import torch.optim as optim

class Transformer(nn.Module):
    def __init__(self, dataset, num_layers=4, d_model=128, dff=512, num_heads=8, dropout_rate=0.1, max_seq_len=128, n_classes=10):
        super(Transformer, self).__init__()

        # Configurations
        self.vocab_size = dataset.tokenizer.VOCAB_SIZE if not dataset.hps['use_continuous_data'] else None
        self.seq_len = max_seq_len
        self.n_classes = n_classes

        # Encoder and Decoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dff, dropout=dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dff, dropout=dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer for reconstruction
        self.output_layer = nn.Linear(d_model, self.vocab_size or 5)

        # Classification head
        self.classify_layer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.n_classes)
        )

        # Loss and Optimizer
        self.loss_fn = nn.CrossEntropyLoss()  # or any other loss
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Define the forward pass
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output_layer(output)

    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        for src, tgt, labels in train_loader:
            src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def evaluate(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for src, tgt, labels in val_loader:
                src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)
                output = model(src, tgt)
                loss = criterion(output, labels)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    def accuracy(outputs, labels):
        _, predicted = torch.max(outputs, dim=1)
        return (predicted == labels).sum().item() / labels.size(0)
    
    def predict(model, input_data, device):
        model.eval()
        with torch.no_grad():
            input_data = input_data.to(device)
            outputs = model(input_data)
            # Apply softmax, argmax, etc., depending on your requirements
            return outputs

    def encode(self, inp, inp_mask, training):
        enc_output = self.encoder(inp, mask=inp_mask)
        # Bottle-neck layer, classification, etc.
        # Return a dictionary or relevant outputs
        return {'enc_output': enc_output, 'class': None}  # Example
    
    def decode(self, embedding, target, target_mask, training):
        dec_output = self.decoder(target, embedding, tgt_mask=target_mask)
        # Further processing as needed
        return {'recon': dec_output}  # Example
    
    def predict(model, inp_seq, device):
        model.eval()
        with torch.no_grad():
            # Process inp_seq to match input format, apply model, and return outputs
            return model(inp_seq)

        ef train_on_batch(model, batch, optimizer, criterion, device):
        model.train()
        inp, tar, labels = batch
        inp, tar, labels = inp.to(device), tar.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(inp, tar, ... )  # Add additional arguments as needed
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        return {'loss': loss.item(), 'other_metrics': None}  # Replace with actual metrics

    def prepare_for_epoch():
        # Reset or initialize things at the start/end of each epoch
        pass
    
    
    def classify_from_embedding(self, embedding, training):
        if self.hps['class_buffer_layers']:
            fc = embedding
            for lid in range(self.hps['class_buffer_layers']):
                fc = self.class_buffer[lid](fc)
                if training:
                    fc = self.class_dropout[lid](fc)
        else:
            fc = embedding
        pred_labels = self.classify_layer(fc)
        return pred_labels
    
    def prepare_for_start_of_epoch():
        # Reset or initialize any state or variable
        pass
    
    def make_dummy_input(self, expected_len, nattn, batch_size):
        nignore = self.seq_len - nattn

        if self.dataset.hps['use_continuous_data']:
            dummy = torch.cat([
                torch.zeros(batch_size, nattn, 5),
                torch.zeros(batch_size, nignore, 5).fill_(1)
            ], dim=1)
        else:
            dummy = torch.cat([
                torch.ones(batch_size, nattn),
                torch.zeros(batch_size, nignore)
            ], dim=1).long()  # Use .long() for integer tensors
        return dummy
