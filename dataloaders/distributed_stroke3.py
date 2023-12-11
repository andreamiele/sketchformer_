import numpy as np
import os
import glob
import utils
import time

from core.data import BaseDataLoader, DatasetSplit
from torch.utils.data import Dataset, DataLoader



class Stroke3Dataset(Dataset):
    def __init__(self, file_paths, hps, tokenizer=None):
        self.file_paths = file_paths
        self.hps = hps
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def load_data(self):
        # Implement the data loading logic here
        # This could involve loading numpy files and organizing them into a dictionary
        data = {'x': [], 'y': []}
        for file_path in self.file_paths:
            loaded_data = np.load(file_path, allow_pickle=True)
            data['x'].extend(loaded_data['x'])
            data['y'].extend(loaded_data['y'])
        return data

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        sketch = self.data['x'][idx]
        label = self.data['y'][idx]

        # Preprocess the sketch
        sketch = self.preprocess_sketch(sketch)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return sketch, label

    def preprocess_sketch(self, sketch):
        # Implement sketch preprocessing here
        # This might include normalization, padding, and conversion to a PyTorch tensor
        # ...

    def random_scale(self, data):
        # Augment data by stretching x and y axis randomly [1-e, 1+e]
        x_scale_factor = (random.random() - 0.5) * 2 * self.hps['random_scale_factor'] + 1.0
        y_scale_factor = (random.random() - 0.5) * 2 * self.hps['random_scale_factor'] + 1.0
        data[:, 0] *= x_scale_factor
        data[:, 1] *= y_scale_factor
        return data
    
    def preprocess_sketch(self, sketch):
        # Remove large gaps from the data
        sketch = np.minimum(sketch, self.hps['limit'])
        sketch = np.maximum(sketch, -self.hps['limit'])
        sketch = np.array(sketch, dtype=np.float32)

        # Normalize the sketch
        min_x, max_x, min_y, max_y = self.get_bounds(sketch)
        max_dim = max(max_x - min_x, max_y - min_y, 1.0)
        sketch[:, :2] /= max_dim

        # Augment sketch if needed
        if self.hps['augment_stroke_prob'] > 0 and random.random() < self.hps['augment_stroke_prob']:
            sketch = self.random_scale(sketch)

        # Convert to absolute coordinates if required
        if self.hps['use_absolute_strokes']:
            sketch = self.convert_to_absolute(sketch)

        # Tokenize if using discrete representation
        if not self.hps['use_continuous_data']:
            sketch = self.tokenizer.encode(sketch)

        # Pad the sketch to the desired sequence length
        sketch = self.pad_sketch(sketch)

        # Convert to PyTorch tensor
        return torch.tensor(sketch, dtype=torch.float32)

    def get_bounds(self, sketch):
        # Calculate the bounds of the sketch
        min_x = np.min(sketch[:, 0])
        max_x = np.max(sketch[:, 0])
        min_y = np.min(sketch[:, 1])
        max_y = np.max(sketch[:, 1])
        return min_x, max_x, min_y, max_y
    
    def convert_to_absolute(self, sketch):
        # Convert a sketch from relative to absolute coordinates
        # This is a placeholder; implement according to your specific needs
        # ...

    def pad_sketch(self, sketch):
        # Pad the sketch to a fixed length
        padded_sketch = np.zeros((self.hps['max_seq_len'], sketch.shape[1]))
        padded_sketch[:len(sketch)] = sketch
        return padded_sketch


class DistributedStroke3DataLoader:
    def __init__(self, hps, data_directory, tokenizer=None):
        self.hps = hps
        self.data_directory = data_directory
        self.tokenizer = tokenizer
        self.train_dataset, self.test_dataset, self.valid_dataset = self.get_data_splits()

    def get_data_splits(self):
        # Get file paths for each data split
        train_files = glob.glob(os.path.join(self.data_directory, 'train*.npz'))
        test_files = glob.glob(os.path.join(self.data_directory, 'test*.npz'))
        valid_files = glob.glob(os.path.join(self.data_directory, 'valid*.npz'))

        train_dataset = Stroke3Dataset(train_files, self.hps, self.tokenizer)
        test_dataset = Stroke3Dataset(test_files, self.hps, self.tokenizer)
        valid_dataset = Stroke3Dataset(valid_files, self.hps, self.tokenizer)

        return train_dataset, test_dataset, valid_dataset

    def get_loader(self, split_name, batch_size, shuffle=True):
        if split_name == 'train':
            dataset = self.train_dataset
        elif split_name == 'test':
            dataset = self.test_dataset
        else:
            dataset = self.valid_dataset

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

