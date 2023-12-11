import torch
import torch.nn as nn

class LossManager(object):

    def __init__(self):
        self.loss_names = []
        self.loss_weights = {}
        self.loss_fns = {}

    def _add_loss(self, name, weight, func):
        self.loss_names.append(name)
        self.loss_weights[name] = weight
        self.loss_fns[name] = func

    def add_sparse_categorical_crossentropy(self, name='class', weight=1.0):
        self._add_loss(
            name, weight,
            nn.CrossEntropyLoss(weight=weight))

    def add_reconstruction_loss(self, name='recon', weight=1.0):
        def loss_function(real, pred):
            mask = (real != 0)
            loss_ = nn.functional.cross_entropy(pred, real, reduction='none')
            loss_ *= mask.float()
            return loss_.mean()
        
        self._add_loss(name, weight, loss_function)

    def add_continuous_reconstruction_loss(self, name='recon', weight=1.0):
        def loss_function(real, pred):
            mask = (real[..., -1] != 1)
            tgt_locations = real[..., :2]
            pred_locations = pred[..., :2]
            tgt_metadata = real[..., 2:]
            pred_metadata = pred[..., 2:]

            location_loss = nn.functional.mse_loss(pred_locations, tgt_locations)
            metadata_loss = nn.functional.cross_entropy(pred_metadata, tgt_metadata.argmax(dim=-1), reduction='mean')
            loss_ = location_loss + metadata_loss

            loss_ *= mask.float()
            return loss_.mean()

        self._add_loss(name, weight, loss_function)

    def add_mae_loss(self, name, weight=1.):
        self._add_loss(name, weight, nn.L1Loss())

    def add_mean_loss(self, name, weight=1.):
        self._add_loss(name, weight, lambda x: x.mean())

    def add_mse_loss(self, name, weight=1.):
        self._add_loss(name, weight, nn.MSELoss())

    def compute_all_loss(self, rp_dict):
        losses = {}
        for name in self.loss_names:
            losses[name] = self.loss_weights[name] * self.loss_fns[name](*rp_dict[name])
        return losses

    def compute_loss(self, name, *args):
        assert name in self.loss_names, "Error! Loss name {} not found".format(name)
        return self.loss_weights[name] * self.loss_fns[name](*args)
