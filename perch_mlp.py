import os
from typing import Any

import torch
import torch.nn as nn
import lightning as pl

class PerchMLP(pl.LightningModule):
    def __init__(self, n_in, n_out, lr, log_path):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_out),
            nn.Sigmoid()
        )
        self.lr = lr
        self.criterion = nn.BCELoss()
        # pas BCEWithLogits parce que c'est bien que le forward se termine par une Sigmoid
        # pour avoir sortie dans [0, 1]
        #self.save_hyperparameters()
        self.log_path = log_path
        self.train_step_outputs = []
        self.train_step_targets = []
        self.val_step_outputs = []
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.layers(X.view(X.size(0), -1))
        train_loss = self.criterion(y_hat, y)
        self.log('train_loss', train_loss)

        #y_pred = y_hat.cpu().detach().numpy()
        #y_true = y.cpu().detach().numpy()
        #self.train_step_outputs.extend(y_pred)
        #self.train_step_targets.extend(y_true)

        return train_loss

    #def on_train_epoch_end(self):
        #torch.save(self.train_step_outputs,
        #           os.path.join(self.log_path, f'ep{self.current_epoch}_train_outputs.pt'))
        #torch.save(self.train_step_targets,
        #           os.path.join(self.log_path, f'ep{self.current_epoch}_train_targets.pt'))

        #self.train_step_outputs.clear()
        #self.train_step_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.layers(X.view(X.size(0), -1))
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss)

        y_pred = y_hat.cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()
        self.val_step_outputs.extend(y_pred)
        self.val_step_targets.extend(y_true)

        return val_loss

    def on_validation_epoch_end(self):
        torch.save(self.val_step_outputs, 
                   os.path.join(self.log_path, f'ep{self.current_epoch}_val_outputs.pt'))
        torch.save(self.val_step_targets,
                   os.path.join(self.log_path, f'ep{self.current_epoch}_val_targets.pt'))

        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.layers(X.view(X.size(0), -1))
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

        y_pred = y_hat.cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()
        self.test_step_outputs.extend(y_pred)
        self.test_step_targets.extend(y_true)
        
        torch.save(self.test_step_outputs,
                   os.path.join(self.log_path, f'test_outputs.pt'))
        torch.save(self.test_step_targets,
                   os.path.join(self.log_path, f'test_targets.pt'))

        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))