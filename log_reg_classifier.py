import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import FeatureGenerator
import HateSpeechClassifier
from HateSpeechClassifier import *
from FeatureGenerator import *
from MetricsGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, f1_score, confusion_matrix

TRAIN_FILE = "civility_data/train.tsv"
DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"
DEV_FILE = "civility_data/dev.tsv"

def run_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Runs metrics to evaluate results for classification

    Args:
        y_true (pd.Series): Pandas series of true values.
        y_pred (pd.Series): Pandas series of predicted values.

    Returns:
        dict: Dictionary that contains the metrics accuracy, precision, recall, and fscore.
    """
    accuracy = accuracy_score(y_true, y_pred)
    labels = ['OFF', 'NOT']
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels = labels)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }

def fpr(df: pd.DataFrame) -> float:
    """Calculates fpr for a df that contains the actual and prediction values.

    Args:
        df (pd.DataFrame): Pandas df that contains 'label' and 'y_pred' column

    Returns:
        float: FPR for the dataframe.
    """
    #extract true and predicted values from the df
    y_true = df['label']
    y_pred = df['y_pred']

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = ['NOT', 'OFF']).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return fpr
    
def fpr_demographic(df: pd.DataFrame, y_pred: pd.Series) -> dict:
    """Generate FPR for each unique demographic in a dataframe

    Args:
        df (pd.DataFrame): Pandas df containing 'demographic' column and 'label' column.
        y_pred (pd.Series): Pandas series with the predicted values for the df.

    Returns:
        dict: Dictionary with the FPR for each demographic.
    """
    #add predictions to df
    df['y_pred'] = y_pred
    
    #extract all demographics
    unique_demographics = df['demographic'].unique()
    
    #initalize dictionary to hold fpr for each demographic
    fpr_by_demographic = {}
    
    #calculate FPR for each demographic
    for demographic in unique_demographics:
        fpr_by_demographic[demographic] = fpr(df[df['demographic'] == demographic])
        
    return fpr_by_demographic

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from tweets

    Args:
        df (pd.Series): Pandas dataframe that contains a column called "text".

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
        
    feature_generator = FeatureGenerator(df)
    feature_generator.add_punctuation_count()
    feature_generator.add_str_length()
    feature_generator.add_capital_ratio()
    feature_generator.add_count_profanity()
    feature_generator.add_sentiment_analysis()
    feature_generator.scale_features()
    return feature_generator.get_features()

if __name__ == "__main__":
    #import data
    df_train = pd.read_csv(TRAIN_FILE, delimiter= "\t")
    df_dev = pd.read_csv(DEV_FILE, delimiter = "\t")
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE, delimiter= "\t")
    
    #add target column for demographic file 
    df_demographic['label'] = "NOT"
    print("Data imported successfully!")
    
    #generate features and scales (scaling features improves model performance)
    X_train = generate_features(df_train)
    print("Features generated successfully!")
    
    #train model
    classifier = HateSpeechClassifier(X_train, df_train['label'])
    model = classifier.generate_model()
    classifier.select_threshold()
    print("Model Trained!")
    
    #generate features
    X_dev = generate_features(df_dev)
    X_demographics = generate_features(df_demographic)
    print("Dev features generated")
    
    #adjust threshold 
    df_dev['pred'] = classifier.predict(X_dev)
    df_demographic['pred'] = classifier.predict(X_demographics)

    #run evaluations
    metrics = MetricsGenerator(df_dev, df_demographic, df_dev['pred'], df_demographic['pred'])
    fpr = metrics.fpr()
    metrics_dev = metrics.run_metrics()
    fpr_demo = metrics.fpr_demographic()
    # metrics_dev = run_metrics(df_dev['label'], df_dev['pred'])
    # fpr_demo = fpr_demographic(df_demographic, df_demographic['pred'])  

    print(metrics_dev)
    print(fpr_demo)