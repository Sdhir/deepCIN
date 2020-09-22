import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, monitor='val_loss', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor
        self.val_loss_min = np.Inf
        self.val_acc_max = 0

    def __call__(self, monitor_var, model,hist_dir):

        if self.monitor == 'val_acc':
            score = monitor_var
        else:
            score = -monitor_var

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(monitor_var, model,hist_dir)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(monitor_var, model,hist_dir)
            self.counter = 0

    def save_checkpoint(self, monitor_var, model,hist_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.monitor == 'val_acc':
                print(f'Validation acc increased ({self.val_acc_max:.6f} --> {monitor_var:.6f}).  Saving model ...')
                self.val_acc_max = monitor_var
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {monitor_var:.6f}).  Saving model ...')
                self.val_loss_min = monitor_var
        #torch.save(model.state_dict(), os.path.join(hist_dir,'checkpoint_loss_{:.3f}.pt'.format(val_loss)))
        
