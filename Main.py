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
PERSPECTIVE_THRESHOLD = 0.8

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

def generate_classifier(X_train, df_train)-> tuple[LogisticRegression, HateSpeechClassifier]:
    """Generates model from engineered features and original data. 

    Args:
        X_train (_type_): Engineered features df. 
        df_train (_type_): Complete df including labels. 

    Returns:
        LogisticRegression: Model trained using train. 
    """
    #train model
    classifier = HateSpeechClassifier(X_train, df_train['label'])
    model = classifier.generate_model()
    classifier.select_threshold()
    return model, classifier

def generate_predictions(classifier, df_dev, df_demographics):
    #generate features
    X_dev = generate_features(df_dev)
    X_demographics = generate_features(df_demographic)
    print("Dev features generated")
    
    pred_dev = classifier.predict(X_dev)
    pred_dem = classifier.predict(X_demographics)
    
    return pred_dev, pred_dem

def generate_perspective():
    df_dev_class = pd.read_csv(PERSPECTIVE_DEV_FILE, delimiter = "\t")
    df_demographic_class = pd.read_csv(PERSPECTIVE_DEMOGRAPHIC_FILE, delimiter = "\t")

    df_demographic_class['label'] = "NOT"

    perspective = ScoreClassifierClass(df_dev_class, df_demographic_class, PERSPECTIVE_THRESHOLD)

    df_dev_class['pred'] = perspective.classify_dev()
    df_demographic_class['pred'] = perspective.classify_demographic()
    
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
    fpr_class, fpr_demo_class, metrics_dev_class = generate_perspective()
    
    #create custom model
    X_train = generate_features(df_train)
    model, classifier = generate_classifier(X_train, df_train)    
    df_dev['pred'], df_demographic['pred'] = generate_predictions(classifier, df_dev, df_demographic)

    #run evaluations
    fpr, fpr_demo, metrics_dev = run_metrics(df_dev, df_demographic, df_dev['pred'], df_demographic['pred'])