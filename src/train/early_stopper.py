import numpy as np

class EarlyStopper:
    """
    This class assumes the goal is to minimize (designed to use with loss function)
    """
    def __init__(self, patience):
        self._patience = patience
        self._counter = 0
        self._best_metric = np.inf 

    def should_stop(self, metric):
        if metric < self._best_metric:
            self._counter = 0
            self._best_metric = metric
            return False
        self._counter += 1
        if self._counter > self._patience:
            return True
        
    def reset_state(self):
        self._counter = 0
        self._best_metric = np.inf 
