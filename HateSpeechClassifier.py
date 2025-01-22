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
   
    def generate_model(self)-> LogisticRegression:
        """
        Generates a logistic regression model using X and y to train with model selection based on flag.
        
        Returns:
            LogisticRegression: Logistic regression model. 
        """
        
        # Define the parameter grid:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000,2500],
            'tol': [1e-3]
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
            cv=5,  
            scoring=scoring,
            verbose=1,
            n_jobs=-1,  
            refit='accuracy'
        )

        model.fit(self.X, self.y)
            
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

    def select_threshold(self):
        #get probability of offensive
        y_prob = pd.Series(self.model.predict_proba(self.X)[:, 1])
        
        #specify thresholds to try
        thresholds = np.arange(0.0, 1.01, 0.005)
        
        # Lists to store FPR and accuracy values
        fpr_values = []
        f1_values = []

        labels = ['NOT', 'OFF']
        for value in thresholds:
            y_pred = self.classify(y_prob, value)
            fpr = self.false_positive_rate(self.y, y_pred)
            precision, recall, fscore, support = precision_recall_fscore_support(self.y, y_pred, average='weighted', zero_division=0, labels = labels)

            # Store FPR and accuracy for the current threshold
            fpr_values.append(fpr)
            f1_values.append(fscore)
        
        values = pd.DataFrame({'threshold': thresholds, "f1": f1_values, 'fpr': fpr_values})
        
        pd.set_option('display.max_rows', None)  # Disable row truncation
        #print(values[values['f1'] > .7])
        # Plot Threshold vs. FPR and Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, fpr_values, label='False Positive Rate (FPR)', color='red')
        plt.plot(thresholds, f1_values, label='F1', color='blue')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold vs. F1 and FPR')
        plt.legend()
        plt.grid(True)
        plt.savefig("classify.png")
        
        self.threshold = values[values['f1'] > .7]['threshold'].max()
    
    def predict(self, X):
        y = pd.Series(self.model.predict_proba(X)[:, 1])
        return self.classify(y, self.threshold)