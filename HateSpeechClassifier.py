import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, make_scorer, f1_score, confusion_matrix

class HateSpeechClassifier:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.model = None
        self.threshold = 0.5
    
    def false_positive_rate(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred,labels = ['NOT', 'OFF']).ravel()
        return fp / (fp + tn)
   
    def generate_model(self, train_mode)-> LogisticRegression:
        """
        Generates a logistic regression model using X and y to train with model selection based on flag.
        
        Returns:
            LogisticRegression: Logistic regression model. 
        """
        
        # Define the parameter grid:
        if train_mode == 'low':
            param_grid = {
                'C': [100],
                'penalty': ['l2'],
                'solver': ['newton-cg'],
                'max_iter': [500],
                'tol': [0.001]
            }
        if train_mode == 'med':
            param_grid = {
                'solver': ['lbfgs'],  # Reduce solver options.
                'penalty': ['l2'],  # Keep one penalty type.
                'C': [0.001, 0.01, 1, 10],  # Select a subset of values for regularization.
                'max_iter': [500, 1000],  # Use fewer iteration options.
                'tol': [1e-4, 1e-3],  # Keep fewer tolerance values.
                'fit_intercept': [True],  # Keep this as is (only one option).
                'class_weight': [None]  # Reduce class weighting options.
            }
        if train_mode == 'high':
            param_grid = {
                'solver': ['lbfgs', 'newton-cg'],
                'penalty': ['l2', 'none'],
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'max_iter': [500, 1000, 2500, 5000],
                'tol': [1e-4, 1e-3, 1e-2],
                'fit_intercept': [True],
                'class_weight': [None, 'balanced']
            }


        # Create the logistic regression model
        log_reg = LogisticRegression(random_state=42)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'fpr': make_scorer(self.false_positive_rate, greater_is_better=False)
        }
        
        # Create the GridSearchCV object
        model = GridSearchCV(
            log_reg, 
            param_grid, 
            scoring=scoring,
            verbose=1,
            n_jobs=-1,  
            cv = 5,
            refit='accuracy'
        )

        model.fit(self.X, self.y)
            
        print(model.best_params_)
        self.model = model.best_estimator_
        return self.model
    
    def classify(self, X: pd.Series, threshold: float) -> pd.Series:
        """Classifies datapoints based on a threshold

        Args:
            X (pd.Series): Pandas series of data points
            threshold (float): Float threshold to use as the classification boundary

        Returns:
            pd.Series: Data points classified as 'OFF' or 'NOT'
        """
        y = X > threshold

        return y.replace({True: 'OFF', False: 'NOT'})

    def optimize_threshold(self, goal_accuracy: float):
        #get probability of offensive
        y_prob = pd.Series(self.model.predict_proba(self.X)[:, 1])
        
        #specify thresholds to try
        thresholds = np.arange(0.0, 1.01, 0.005)
        
        # Lists to store FPR and accuracy values
        fpr_values = []
        accuracy_values = []

        labels = ['NOT', 'OFF']
        for value in thresholds:
            y_pred = self.classify(y_prob, value)
            fpr = self.false_positive_rate(self.y, y_pred)
            accuracy = accuracy_score(self.y, y_pred)

            # Store FPR and accuracy for the current threshold
            fpr_values.append(fpr)
            accuracy_values.append(accuracy)
        
        values = pd.DataFrame({'threshold': thresholds, "accuracy": accuracy_values, 'fpr': fpr_values})
        
        pd.set_option('display.max_rows', None)  # Disable row truncation

        # Plot Threshold vs. FPR and Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, fpr_values, label='False Positive Rate (FPR)', color='red')
        plt.plot(thresholds, accuracy_values, label='Accuracy', color='blue')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold vs. Accuracy and FPR')
        plt.legend()
        plt.grid(True)
        plt.savefig("classify.png")
        
        self.threshold = values[values['accuracy'] > goal_accuracy]['threshold'].max()
        if pd.isna(self.threshold):
            self.threshold = (values.sort_values(by='accuracy', ascending=False))['threshold'].iloc[0]
            print(f"Accuracy goal not hit. Threshold {self.threshold} instead.")
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def predict(self, X):
        y = pd.Series(self.model.predict_proba(X)[:, 1])
        return self.classify(y, self.threshold)