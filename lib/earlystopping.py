import numpy as np


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0, restore_best_weights=False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_weights = None
        self.restore_best_weights = restore_best_weights


    def __call__(self, val_loss, model=None):
        print("val loss :", val_loss, end=' ')

        if val_loss < self.best_loss - self.min_delta:

            if self.restore_best_weights and model is not None:
                self.best_weights = model.get_weights()

            self.best_loss = val_loss
            self.counter = 0
            print(f"Validation loss decreased. Resetting counter.")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.set_weights(self.best_weights)
                
        return self.early_stop

