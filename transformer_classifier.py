import pandas as pd
import sklearn 
import torch
import pickle
import os.path
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, make_scorer, f1_score, confusion_matrix
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV

TRAIN_FILE = "civility_data/train.tsv"
TEST_FILE = "civility_data/test.tsv"
DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"
DEV_FILE = "civility_data/dev.tsv"

def generate_transformers(text: pd.Series, model: str) -> pd.DataFrame:
    """Generates features from text using Hugging Face transformers

    Args:
        text (pd.Series): Pandas series of text.
        model (str): Str that indicates the name of the model to use

    Returns:
        pd.DataFrame: Dataframe of features.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    feature_extractor = pipeline("feature-extraction", model=model, device = device)
    
    X_text = text.tolist()
    
    X_layers = feature_extractor(X_text)
    
    X_feature = [token[-1] for feature in X_layers for token in feature]
    
    return pd.DataFrame(X_feature)

def preprocess(text: pd.Series) -> pd.Series:
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.add("user")

    text = text.str.lower()

    text = text.apply(word_tokenize)
    
    def filter_stopwords(sentence):
        return [w for w in sentence if not w.lower() in stop_words]

    text = text.apply(filter_stopwords)
    
    return text

def generate_features(df: pd.DataFrame, flag: int, file: str) -> pd.DataFrame:
    """Generates features based on text column in df using multiple frameworks. Flag 0 uses 

    Args:
        df (pd.DataFrame): Pandas df containing a column text.
        flag (int): Value between 0 and 0 indicating which method to use. 

    Returns:
        pd.DataFrame: Dataframe of features. 
    """
    filename = f"{file}_{flag}.pkl"
    
    #either load file or generate file
    if (os.path.exists(filename)):
        print("pulling from file")
        with open(filename, 'rb') as f:
            features = pickle.load(f)
    else:
        if (flag == 0):
            features = generate_transformers(df['text'], "xlm-roberta-base")
        elif (flag == 1):
            features = generate_transformers(df['text'], "cardiffnlp/twitter-roberta-base-hate-latest")
        elif (flag == 2):
            features = generate_transformers(df['text'], "KoalaAI/HateSpeechDetector")

        #save file
        with open(filename, 'wb') as f:
            pickle.dump(features, f)
    
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(features))
    return X_scaled

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,labels = ['NOT', 'OFF']).ravel()
    return fp / (fp + tn)

def generate_model(X: pd.DataFrame, y: pd.Series, flag: int)-> LogisticRegression:
    """Generates a logistic regression model using X and y to train with model selection based on flag.

    Args:
        X (pd.DataFrame): Design matrix for dataset.
        y (pd.Series): Target vector

    Returns:
        LogisticRegression: Logistic regression model. 
    """
    filename = f"model_{flag}.pkl"
    
    #either load file or generate file
    if (os.path.exists(filename)):
        print("pulling from file")
        with open(filename, 'rb') as f:
            model = pickle.load(f)
    else:   
        # Define the parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1], #tighten regularization because models are large
            'penalty': ['l1', 'elasticnet'],
            'solver': ['saga'],
            'max_iter': [1000,2500],
            'l1_ratio': [0.1, 0.5, 0.9],
            'tol': [1e-3]
        }

        # Create the logistic regression model
        log_reg = LogisticRegression(random_state=42)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'fpr': make_scorer(false_positive_rate, greater_is_better=False)
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

        model.fit(X, y)
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
    print("Best parameters:", model.best_estimator_)

    return model.best_estimator_

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

def classify(X: pd.Series, threshold: float) -> pd.Series:
    y = X > threshold

    return y.replace({True: 'OFF', False: 'NOT'})

def select_threshold(model, X, y_true, adjust_threshold):
    global set_threshold
    #get probability of offensive
    y_prob = pd.Series(model.predict_proba(X)[:, 1])
    
    #specify thresholds to try
    thresholds = np.arange(0.0, 1.01, 0.005)
    
    # Lists to store FPR and accuracy values
    fpr_values = []
    f1_values = []
    
    labels = ['NOT', 'OFF']
    for value in thresholds:
        y_pred = classify(y_prob, value)
        fpr = false_positive_rate(y_true, y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels = labels)

        # Store FPR and accuracy for the current threshold
        fpr_values.append(fpr)
        f1_values.append(fscore)
    
    values = pd.DataFrame({'threshold': thresholds, "f1": f1_values, 'fpr': fpr_values})
    
    pd.set_option('display.max_rows', None)  # Disable row truncation
    print(values[values['f1'] > .7])
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
    
    if(adjust_threshold):
        set_threshold = values[values['f1'] > .7]['threshold'].max()
    print(set_threshold)
    return classify(y_prob, set_threshold)
    
if __name__ == "__main__":
    #import data
    df_train = pd.read_csv(TRAIN_FILE, delimiter= "\t")
    df_dev = pd.read_csv(DEV_FILE, delimiter = "\t")
    #df_test = pd.read_csv(TEST_FILE, delimiter = "\t")
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE, delimiter= "\t")
    df_demographic['label'] = "NOT"
    #df_test['label'] = "NOT" # just so classify runs
    
    flag = 0
    #generate features -> scaled
    X_train = generate_features(df_train, flag, "features")
    print("generated train features!")
    
    #train model
    model = generate_model(X_train, df_train['label'], flag)
    
    #generate feature
    X_dev = generate_features(df_dev, flag, "dev")
    X_demographics = generate_features(df_demographic, flag, "demographics")
    #X_test = generate_features(df_test, flag, "test")
    print("generated dev features!")

    #adjust threshold 
    df_dev['pred'] = select_threshold(model, X_dev, df_dev['label'], True)
    df_demographic['pred'] = select_threshold(model, X_demographics, df_demographic['label'], False)
    #df_test['pred'] = select_threshold(model, X_test, df_test['label'], False) #just clasifies based on threshold

    #run evaluations
    metrics_dev = run_metrics(df_dev['label'], df_dev['pred'])
    fpr_demo = fpr_demographic(df_demographic, df_demographic['pred'])  

    #output_file = 'Julie_Lawler_test.tsv'
    #df_test[['pred']].to_csv(output_file, sep='\t', index=False, header=True)

    #print(flag)
    #print(metrics_dev)
    #print(fpr_demo)