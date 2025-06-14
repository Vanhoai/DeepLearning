from typing import Any
from enum import Enum


class MonitorEarlyStopping(Enum):
    VAL_LOSS = "VAL_LOSS"
    VAL_ACCURACY = "VAL_ACCURACY"


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-3,
        is_store: bool = True,
        monitor: MonitorEarlyStopping = MonitorEarlyStopping.VAL_LOSS,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.is_store = is_store
        self.monitor = monitor

        self.best_value = None
        self.wait = 0
        self.best_weights = None
        self.best_bias = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, current_value: float, weights, bias):
        if self.monitor == MonitorEarlyStopping.VAL_LOSS:
            # Lower is better for loss
            if (
                self.best_value is None
                or current_value < self.best_value - self.min_delta
            ):
                self.best_value = current_value
                self.wait = 0
                if self.is_store:
                    self.best_weights = weights
                    self.best_bias = bias
            else:
                self.wait += 1
        else:
            # Higher is better for accuracy
            if (
                self.best_value is None
                or current_value > self.best_value + self.min_delta
            ):
                self.best_value = current_value
                self.wait = 0
                if self.is_store:
                    self.best_weights = weights
                    self.best_bias = bias
            else:
                self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True

        return False
