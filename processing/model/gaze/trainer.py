import torch
import torch.nn as nn

from processing.model.trainer import Trainer
from processing.utils.gaze.early_stopping import GazeEarlyStopping
from processing.utils.utils import mask_mse_loss


class GazeTrainer(Trainer):
    def __init__(self, cf, model, train_dl, val_dl, optim, scheduler, eval_dir,
                 task, device, monitor, monitor_mode):
        early_stop = GazeEarlyStopping(cf, model, val_dl, eval_dir, device, task, monitor, monitor_mode)
        super().__init__(cf, model, train_dl, eval_dir, early_stop, task, device)

        self.optim = optim
        self.scheduler = scheduler
        self.max_grad_norm = cf.max_grad_norm
        self.target_pad = train_dl.target_pad

        self.criterion = nn.MSELoss(reduction="mean")

    def train_one_step(self, batch):
        self.model.zero_grad()

        b_input, b_target, b_mask = batch
        b_input = b_input.to(self.device)
        b_target = b_target.to(self.device)
        b_mask = b_mask.to(self.device)

        b_output = self.model(input_ids=b_input, attention_mask=b_mask)[0]

        active_outputs, active_targets = mask_mse_loss(b_output, b_target, self.target_pad, self.model.d_out)
        loss = self.criterion(active_outputs, active_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
        self.optim.step()
        self.scheduler.step()

        return loss.item()
