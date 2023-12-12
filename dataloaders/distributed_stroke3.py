import numpy as np
import os
import glob
import utils
import time

from core.data import BaseDataLoader, DatasetSplit
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset

class DistributedStroke3Dataset(Dataset):
    def __init__(self, data_directory, max_seq_len=200, shuffle_stroke=False, token_type='dictionary', use_continuous_data=False, use_absolute_strokes=False, tokenizer_dict_file='prep_data/sketch_token/token_dict.pkl', tokenizer_resolution=100, augment_stroke_prob=0.1, random_scale_factor=0.1):
        self.max_seq_len = max_seq_len
        self.shuffle_stroke = shuffle_stroke
        self.token_type = token_type
        self.use_continuous_data = use_continuous_data
        self.use_absolute_strokes = use_absolute_strokes
        self.tokenizer_dict_file = tokenizer_dict_file
        self.tokenizer_resolution = tokenizer_resolution
        self.augment_stroke_prob = augment_stroke_prob
        self.random_scale_factor = random_scale_factor

        # Load data here
        self.data_directory = data_directory
        self.data, self.labels = self.load_data(data_directory)

    def load_data(self):
        all_data = []
        all_labels = []
        for npz_file in glob.glob(os.path.join(self.data_directory, 'train_*.npz')):
            with np.load(npz_file, allow_pickle=True) as data:
                all_data.extend(data['x'])
                all_labels.extend(data['y'])
        return all_data, all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sketch = self.data[idx]
        label = self.labels[idx]
        # Apply any preprocessing here if needed
        # Convert sketch to a PyTorch tensor
        sketch_tensor = torch.tensor(sketch, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sketch_tensor, label_tensor

    def preprocess(self, data, augment=False):
        preprocessed = []
        for sketch in data:
            sketch = np.clip(sketch, -self.limit, self.limit)
            sketch = np.array(sketch, dtype=np.float32)

            if augment:
                sketch = self._augment_sketch(sketch)

            min_x, max_x, min_y, max_y = utils.sketch.get_bounds(sketch)
            max_dim = max(max_x - min_x, max_y - min_y, 1)
            sketch[:, :2] /= max_dim

            if self.shuffle_stroke:
                lines = utils.tu_sketch_tools.strokes_to_lines(sketch, scale=1.0, start_from_origin=True)
                np.random.shuffle(lines)
                sketch = utils.tu_sketch_tools.lines_to_strokes(lines)

            if self.use_absolute_strokes:
                sketch = utils.sketch.convert_to_absolute(sketch)

            if not self.use_continuous_data:
                sketch = self.tokenizer.encode(sketch)

            if len(sketch) > self.max_seq_len:
                sketch = sketch[:self.max_seq_len]

            sketch = self._cap_pad_and_convert_sketch(sketch)
            preprocessed.append(sketch)

        return np.array(preprocessed)

    def random_scale(self, data):
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        data[:, 0] *= x_scale_factor
        data[:, 1] *= y_scale_factor
        return data

    def _cap_pad_and_convert_sketch(self, sketch):
        desired_length = self.max_seq_len
        if not self.use_continuous_data:
            converted_sketch = np.ones((desired_length, 1), dtype=int) * self.tokenizer.PAD
            converted_sketch[:len(sketch), 0] = sketch
        else:
            converted_sketch = np.zeros((desired_length, 5), dtype=float)
            converted_sketch[:len(sketch), :2] = sketch[:, :2]
            converted_sketch[:len(sketch), 2] = 1 - sketch[:, 2]
            converted_sketch[:len(sketch), 3] = sketch[:, 2]
            converted_sketch[len(sketch):, 4] = 1
            converted_sketch[-1, 4] = 1

        return torch.tensor(converted_sketch, dtype=torch.float)

    def _augment_sketch(self, sketch):
        if self.augment_stroke_prob > 0 and self.use_continuous_data:
            sketch = self.random_scale(sketch)
            sketch = utils.sketch.augment_strokes(sketch, self.augment_stroke_prob)
        return sketch

    def preprocess_extra_sets_from_interp_experiment(self, data):
        preprocessed_sketches = []
        for sketch in data:
            if self.use_absolute_strokes:
                sketch = utils.sketch.convert_to_absolute(sketch)
            if not self.use_continuous_data:
                sketch = self.tokenizer.encode(sketch)

            if len(sketch) > self.max_seq_len:
                sketch = sketch[:self.max_seq_len]

            sketch = self._cap_pad_and_convert_sketch(sketch)
            preprocessed_sketches.append(sketch)

        return np.array(preprocessed_sketches)

    def get_class_exclusive_random_batch(self, split_name, n, class_list):
        data = self.get_split_data(split_name)
        x, y = data['x'], data['y']

        np.random.seed(14)
        idx = np.random.permutation(len(x))
        np.random.seed()

        n_per_class = n // len(class_list)
        sel_skts = []
        for chosen_class in class_list:
            n_from_class = 0
            for i in idx:
                if y[i] == chosen_class:
                    sel_skts.append(x[i])
                    n_from_class += 1
                    if n_from_class >= n_per_class:
                        break
        return np.array(sel_skts)
