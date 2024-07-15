import pandas as pd
import numpy as np


class ErrorMetrics:
    def __init__(self, dataframe, estimated_cols, true_cols):
        self.df = dataframe
        self.estimated_cols = estimated_cols
        self.true_cols = true_cols
        self.errors = self.calculate_errors()

    def calculate_errors(self):
        errors = {}
        for est, true in zip(self.estimated_cols, self.true_cols):
            error_squared = (self.df[est] - self.df[true]) ** 2
            error_abs = (self.df[est] - self.df[true]).abs()
            errors[est] = {
                'mse': error_squared.mean(),
                'mae': error_abs.mean(),
                'rmse': np.sqrt(error_squared.mean())
            }
        return errors

    def print_errors(self):
        for col, metrics in self.errors.items():
            print(f"Metrics for {col}:")
            print(f"  MSE: {metrics['mse']}")
            print(f"  MAE: {metrics['mae']}")
            print(f"  RMSE: {metrics['rmse']}")