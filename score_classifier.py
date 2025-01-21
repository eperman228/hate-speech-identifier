"""
Perspective API Score Classifier

This script provides utilities to classify text data based on Perspective API scores
and evaluate the results. It includes methods for:
1. Classifying data using a threshold for toxicity.
2. Running evaluation metrics like accuracy, precision, recall, and F1-score.
3. Calculating false positive rates (FPR) overall and for specific demographic groups.

Dependencies:
- pandas
- scikit-learn

To use, ensure the input files `civility_data/dev.tsv` and `civility_data/mini_demographic_dev.tsv` are
formatted correctly and include the necessary columns.

Author: Julie Lawler
"""
import sklearn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

DEV_FILE = "civility_data/dev.tsv"
DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"
THRESHOLD = 0.8

def classify(df: pd.DataFrame, threshold: float) -> pd.Series:
    """classifies the perspective_score for a df given a specific thresold

    Args:
        df (pd.DataFrame): A pandas df containing the 'perspective_score' column.
        threshold (float): A float value represententing the threshold for classification.

    Returns:
        pd.Series: A pandas series of values indicating whether each score exceeds the threshold. ('OFF' = toxic 'NOT' = not toxic)
    """
    
    if 'perspective_score' not in df.columns:
        raise ValueError("The DataFrame must contain a 'perspective_score' column.")

    X = df['perspective_score']
    y = X > threshold

    return y.replace({True: 'OFF', False: 'NOT'})

def run_metrics(df: pd.DataFrame, col_names: list) -> dict:
    """Runs metrics to evaluate results for classification

    Args:
        df (pd.Datafram): Pandas df to run metrics on.
        col_names (list): List of column names for true and predicted values. 0 should be the true value and 1 should be the predicted value.

    Returns:
        dict: Dictionary that contains the metrics accuracy, precision, recall, and fscore.
    """
    y_true = df[col_names[0]]
    y_pred = df[col_names[1]]

    accuracy = accuracy_score(y_true, y_pred)
    labels = ['OFF', 'NOT']
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels = labels)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }

def fpr(df: pd.DataFrame, col_names: list) -> float:
    """Calculates fpr for a df that contains the actual and prediction values.

    Args:
        df (pd.DataFrame): Pandas df that contains 'label' and 'y_pred' column
        col_names (list): List of column names for true and predicted values. 0 should be the true value and 1 should be the predicted value.

    Returns:
        float: FPR for the dataframe.
    """
    #extract true and predicted values from the df
    y_true = df[col_names[0]]
    y_pred = df[col_names[1]]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return fpr
    
def fpr_demographic(df: pd.DataFrame, col_names: list) -> dict:
    """Generate FPR for each unique demographic in a dataframe

    Args:
        df (pd.DataFrame): Pandas df containing 'demographic' column and true/predicted values.
        col_names (list): List of column names for true and predicted values. 0 should be the true value and 1 should be the predicted value.

    Returns:
        dict: Dictionary with the FPR for each demographic.
    """
    
    #extract all demographics
    unique_demographics = df['demographic'].unique()
    
    #initalize dictionary to hold fpr for each demographic
    fpr_by_demographic = {}
    
    #calculate FPR for each demographic
    for demographic in unique_demographics:
        fpr_by_demographic[demographic] = fpr(df[df['demographic'] == demographic], col_names)
        
    return fpr_by_demographic

def score_classifier():
    #load data from file locations
    df_dev = pd.read_csv(DEV_FILE, delimiter= "\t")
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE, delimiter= "\t")
    
    #create target variable in demographic file (this assumes all are not toxic)
    df_demographic['label'] = "NOT"

    #classify based on perspective_score
    df_dev['pred'] = classify(df_dev, THRESHOLD)
    df_demographic['pred'] = classify(df_demographic, THRESHOLD)
    
    #run metrics to evaluate results
    metrics_dev = run_metrics(df_dev, col_names = ['label', 'pred'])
    metrics_demographic = run_metrics(df_demographic, col_names = ['label', 'pred'])
    
    #asses based on demographics 
    fpr_demo = fpr_demographic(df_demographic, col_names = ['label', 'pred'])  
    
    print(metrics_dev)
    print(metrics_demographic)
    print(fpr_demo)
    
if __name__ == "__main__":
    score_classifier()
