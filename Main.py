import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import FeatureGenerator
import HateSpeechClassifier
import Comparison
import ScoreClassifierClass
from HateSpeechClassifier import *
from FeatureGenerator import *
from MetricsGenerator import *
from ScoreClassifierClass import *
from Comparison import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, f1_score, confusion_matrix

TRAIN_FILE = "civility_data/train.tsv"
DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"
DEV_FILE = "civility_data/dev.tsv"

PERSPECTIVE_DEV_FILE = "civility_data/dev.tsv"
PERSPECTIVE_DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"

def import_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Imports data from string literals.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns train, dev, and demographic data. 
    """
    #import data
    df_train = pd.read_csv(TRAIN_FILE, delimiter= "\t")
    df_dev = pd.read_csv(DEV_FILE, delimiter = "\t")
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE, delimiter= "\t")
    
    #add target column for demographic file 
    df_demographic['label'] = "NOT"

    #return data 
    return df_train, df_dev, df_demographic

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from tweets

    Args:
        df (pd.Series): Pandas dataframe that contains a column called "text".

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
        
    feature_generator = FeatureGenerator(df)
    feature_generator.preprocess()
    feature_generator.add_punctuation_count()
    feature_generator.add_str_length()
    feature_generator.add_capital_ratio()
    feature_generator.add_count_profanity()
    feature_generator.add_sentiment_analysis()
    feature_generator.scale_features()
    return feature_generator.get_features()

def generate_classifier(X, y)-> tuple[LogisticRegression, HateSpeechClassifier]:
    """Generates model from engineered features and original data. 

    Args:
        X (pd.DataFrame): Engineered features df. 
        y (pd.Series): Series containing true labels for df 

    Returns:
        LogisticRegression: Model trained using train. 
    """
    #pull actual values from df
    classifier = HateSpeechClassifier(X, y)
    classifier.set_threshold(threshold=0.7)
    model = classifier.generate_model('med') #low, med, high
    classifier.optimize_threshold(goal_accuracy=0.7)
    return model, classifier

def generate_predictions(classifier, df_dev, df_demographic):
    #generate features
    X_dev = generate_features(df_dev)
    X_demographics = generate_features(df_demographic)
    
    #generate predictions    
    pred_dev = classifier.predict(X_dev)
    pred_dem = classifier.predict(X_demographics)
    
    return pred_dev, pred_dem

def generate_perspective(df_dev, df_demographic):
    #classify 
    threshold = 0.7
    perspective = ScoreClassifierClass(df_dev, df_demographic, threshold)

    df_dev['pred'] = perspective.classify_dev()
    df_demographic['pred'] = perspective.classify_demographic()
    
    metrics_dev_class = perspective.run_metrics_dev()
    metrics_demographic_class = perspective.run_metrics_dem()
    
    fpr_class = perspective.fpr()
    fpr_demo_class = perspective.fpr_demographic()
    
    return fpr_class, fpr_demo_class, metrics_dev_class

def run_metrics(df_dev, df_demographic, pred_dev, pred_dem):
    metrics = MetricsGenerator(df_dev, df_demographic, pred_dev, pred_dem)
    fpr = metrics.fpr()
    metrics_dev = metrics.run_metrics()
    fpr_demo = metrics.fpr_demographic()
    
    compare = Comparison(fpr, fpr_demo, metrics_dev, fpr_class, fpr_demo_class, metrics_dev_class)

    compare.compare()
    
    return fpr, fpr_demo, metrics_dev
    
if __name__ == "__main__":
    #load data
    df_train, df_dev, df_demographic = import_data()
    
    #perspectiveAPI model
    fpr_class, fpr_demo_class, metrics_dev_class = generate_perspective(df_dev, df_demographic)
    
    #create custom model
    X_train = generate_features(df_train)
    model, classifier = generate_classifier(X_train, df_train['label'])    
    df_dev['pred'], df_demographic['pred'] = generate_predictions(classifier, df_dev, df_demographic)

    #run evaluations
    fpr, fpr_demo, metrics_dev = run_metrics(df_dev, df_demographic, df_dev['pred'], df_demographic['pred'])