import torch
from torch.optim.lr_scheduler import LambdaLR

class WarmupDecay(LambdaLR):
    """
    lrate = d**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps
        super(WarmupDecay, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

class StepDecay(LambdaLR):
    """
    Decay the learning rate by a factor when a certain step is reached.
    """
    def __init__(self, optimizer, init_lr, decay_rate=0.1, decay_steps=50000, min_lr_ratio=1e-2):
        self.init_lr = init_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_lr_ratio = min_lr_ratio
        super(StepDecay, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        return max(
            self.init_lr * (self.decay_rate ** (step // self.decay_steps)),
            self.init_lr * self.min_lr_ratio
        )
