import os
from abc import ABC, abstractmethod

import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from processing.settings import LOGGER


class Trainer(ABC):
    def __init__(self, cf, model, train_dl, eval_dir, early_stop, task, device):
        self.model = model
        self.train_dl = train_dl
        self.eval_dir = eval_dir
        self.early_stop = early_stop
        self.n_epochs = cf.n_epochs
        self.task = task
        self.device = device

        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        self.writer = SummaryWriter(self.eval_dir)

    @abstractmethod
    def train_one_step(self, batch):
        pass

    def train(self):
        n_batches_one_epoch = len(self.train_dl)
        n_params = sum(p.numel() for p in self.model.parameters())
        mlflow.log_metric("n_params", n_params)
        LOGGER.info(f"Num epochs: {self.n_epochs}")
        LOGGER.info(f"Num parameters: {n_params}")
        LOGGER.info(f"Begin training task {self.task}")

        self.model.to(self.device)
        self.model.train()

        epoch_loss_ls = []
        it = 1

        for epoch in tqdm(range(1, self.n_epochs + 1)):
            for batch in tqdm(self.train_dl):
                it += 1

                loss = self.train_one_step(batch)
                self.writer.add_scalar("train/loss", loss, it)
                epoch_loss_ls.append(loss)

            epoch_loss_avg = sum(epoch_loss_ls) / len(epoch_loss_ls)
            epoch_loss_ls = []
            LOGGER.info(f"Done epoch {epoch} / {self.n_epochs}")
            LOGGER.info(f"Avg loss epoch {epoch}: {epoch_loss_avg:.4f}")

            self.early_stop()

            for key, metric in self.early_stop.tester.metrics.items():
                self.writer.add_scalar(f"val/{key}", metric, it // n_batches_one_epoch)

            if self.early_stop.stop:
                break
