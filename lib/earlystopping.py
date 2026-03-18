import numpy as np

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf # Start at infinity so any first loss is an improvement
        self.early_stop = False

    def __call__(self, val_loss):
        print("val loss :", val_loss, end=' ')
        # 1. Check if the new loss is significantly better than the best recorded loss
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            print(f"Validation loss decreased. Resetting counter.")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop