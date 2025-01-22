import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, make_scorer, f1_score, confusion_matrix


class MetricsGenerator:
    def __init__(self, df_dev, df_demographic, dev_y_pred, dem_y_pred):
        self.df_dev = df_dev
        self.df_demographic = df_demographic
        self.dev_y_pred = dev_y_pred
        self.dem_y_pred = dem_y_pred
    
    def fpr(self) -> float:
        """Calculates fpr for a df that contains the actual and prediction values."""

        """Args: df (pd.Dataframe): Pandas df that contains 'label' and y_pred column
        
        Returns: float: FPR for the dataframe."""

        y_true = self.df_dev['label']
        y_pred = self.dev_y_pred

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['NOT', 'OFF']).ravel()

        self.fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return self.fpr
    
    def fpr_arg(self, df: pd.DataFrame) -> float:
        """Calculates fpr for a df that contains the actual and prediction values."""
        
        """Args: df (pd.Dataframe): Pandas df that contains 'label' and y_pred column
        
        Returns: float: FPR for the dataframe."""
        
        y_true = df['label']
        y_pred = df['y_pred']
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['NOT', 'OFF']).ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return fpr
    
    def fpr_demographic(self) -> dict:
        """Generate FPR for each unique demographic in a dataframe
        
        Returns: dict: Dictionary with the FPR for each demographic."""

        self.df_demographic['y_pred'] = self.dem_y_pred
        
        unique_demographics = self.df_demographic['demographic'].unique()
        
        self.fpr_by_demographic = {}
        
        for demographic in unique_demographics:
            self.fpr_by_demographic[demographic] = self.fpr_arg(self.df_demographic[self.df_demographic['demographic'] == demographic])
            
        return self.fpr_by_demographic
    
    def run_metrics (self) -> dict:
        """Runs metrics on the dataframe and returns a dictionary with the metrics."""
        
        y_true = self.df_dev['label']
        y_pred = self.dev_y_pred

        accuracy = accuracy_score(y_true, y_pred)
        labels = ['OFF', 'NOT']
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels = labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
        }
