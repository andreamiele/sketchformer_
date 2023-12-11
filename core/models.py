import os
import pickle
import time
import pprint
import socket
from abc import ABCMeta, abstractmethod


import utils
import metrics
from core.metrics import QuickMetric
from core.notifyier import Notifyier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class BaseModel(nn.Module, ABC):
    """
    Base class for all PyTorch models.
    """

    @classmethod
    def base_default_hparams(cls):
        # Define base hyperparameters
        base_hparams = {
            'batch_size': 128,
            'num_epochs': 10,
            'save_every': 1,  # Save every n epochs
            'log_every': 100,  # Log every n steps
            'notify_every': 1000,  # Notify every n steps
            'goal': 'No description',  # Experiment goal
            # Additional hyperparameters...
        }
        return base_hparams

    def __init__(self, hps, dataset, outdir, experiment_id):
        super().__init__()
        self.hps = hps
        self.dataset = dataset
        self.host = socket.gethostname()
        self.experiment_id = experiment_id

        # Directory setup
        self.out_dir = os.path.join(outdir, self.experiment_id)
        self.plots_out_dir = os.path.join(self.out_dir, 'plots')
        self.wgt_out_dir = os.path.join(self.out_dir, 'weights')
        os.makedirs(self.plots_out_dir, exist_ok=True)
        os.makedirs(self.wgt_out_dir, exist_ok=True)

        # Logger setup
        self.writer = SummaryWriter(log_dir=self.out_dir)

        # Model setup
        self.build_model()
        self.optimizer = self.configure_optimizer()

        # Checkpointing setup
        self.checkpoint_path = os.path.join(self.wgt_out_dir, 'checkpoint.pt')
        self.load_checkpoint()

    @abstractmethod
    def build_model(self):
        """
        Build the model architecture.
        """
        pass

    @abstractmethod
    def configure_optimizer(self):
        """
        Configure the optimizer.
        """
        pass

    def load_checkpoint(self):
        """
        Load model checkpoint if it exists.
        """
        if os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from {self.checkpoint_path}")
            
    def save_checkpoint(self):
        """
        Save model checkpoint.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)

    def train_model(self):
        """
        Main training loop.
        """
        for epoch in range(self.hps['num_epochs']):
            self.train()
            for batch_idx, batch in enumerate(self.dataset.train_loader):
                self.optimizer.zero_grad()
                loss = self.train_on_batch(batch)
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.hps['log_every'] == 0:
                    self.log_metrics(loss, epoch, batch_idx)

            if epoch % self.hps['save_every'] == 0:
                self.save_checkpoint()
    @abstractmethod
    def train_on_batch(self, batch):
        """
        Train on a single batch of data.
        """
        pass

    def log_metrics(self, loss, epoch, batch_idx):
        """
        Log metrics to TensorBoard.
        """
        self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.dataset.train_loader) + batch_idx)
        # Add more metrics as needed

    @classmethod
    def default_hparams(cls):
        """
        Combine base and specific hparams.
        """
        return {**cls.base_default_hparams(), **cls.specific_default_hparams()}

    @classmethod
    @abstractmethod
    def specific_default_hparams(cls):
        """
        Specific hyperparameters for the derived model.
        """
        pass         
    
    def train_on_batch(self, batch):
        raise NotImplementedError

    def log_metrics(self, loss):
        # Placeholder for logging metrics
        print(f"Epoch: {self.epoch}, Step: {self.current_step}, Loss: {loss.item()}")

    def save_checkpoint(self):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'step': self.current_step
        }, os.path.join(self.wgt_out_dir, 'model_checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.wgt_out_dir, 'model_checkpoint.pt')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            print("Checkpoint loaded")

    # Placeholder for metrics and plotting
    def compute_all_metrics(self):
        pass

    def plot_all_metrics(self):
        pass

    def prepare_plot_send_slow_metrics(self, msg):
        self.compute_all_metrics()
        plot_file = self.plot_all_metrics()
        # Implement notification logic if necessary
        return msg

    def update_metrics(self, metrics):
        # This function should be defined to update the training metrics
        # Example implementation:
        for metric_name, value in metrics.items():
            self.writer.add_scalar(metric_name, value, self.current_step)

    def plot_metrics(self):
        # This function should plot the desired metrics
        # Placeholder implementation
        metrics = {}  # Replace with actual metrics
        plt.figure(figsize=(10, 5))
        for idx, (metric_name, values) in enumerate(metrics.items()):
            plt.subplot(1, len(metrics), idx + 1)
            plt.plot(np.arange(len(values)), values)
            plt.title(metric_name)
        plt.tight_layout()
        plot_path = os.path.join(self.plots_out_dir, f'metrics_epoch_{self.epoch}.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def send_notifications(self, message, plot_path=None):
        # This function should send notifications with updates
        # Placeholder implementation
        print(message)
        if plot_path:
            print(f"Plot saved at {plot_path}")
        # Integrate with actual notification system (e.g., email or Slack)

    def validate_on_dataset(self, validation_dataset):
        # This function performs validation on a given dataset and updates metrics
        # Placeholder implementation
        for batch in validation_dataset:
            # Perform validation on the batch and update metrics
            pass

    def save_precomputed_data(self, data, filename):
        file_path = os.path.join(self.tmp_out_dir, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_precomputed_data(self, filename):
        file_path = os.path.join(self.tmp_out_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None
