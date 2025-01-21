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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV

TRAIN_FILE = "civility_data/train.tsv"
TEST_FILE = "civility_data/test.tsv"
DEMOGRAPHIC_FILE = "civility_data/mini_demographic_dev.tsv"
DEV_FILE = "civility_data/dev.tsv"

#credit for this list goes to better_profanity package
profane_words = pd.read_csv("Profanity_Wordlist.txt", header =None, names = ['word'])
profane_words = profane_words['word'].tolist()

#global var for threshold
set_threshold = 0.5
#build model using train to classify offensive vs. not tweets
#report results on dev 
#report FPR over mini_demographic
#predict on test

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data by tokenizing, lowercasing, and removing stopwords. 

    Args:
        df (pd.Dataframe): Pandas dataframe of containing 'text' column to process.

    Returns:
        pd.Dataframe: Original df with new column 'clean_text'.
    """
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.add("user")

    text = df['text'].str.lower()
    
    def remove_punctuation(tweet):
        return ''.join(char for char in tweet if char not in string.punctuation)

    text = text.apply(remove_punctuation)

    text = text.apply(word_tokenize)
    
    def filter_stopwords(sentence):
        return [w for w in sentence if not w.lower() in stop_words]

    text = text.apply(filter_stopwords)
    
    df['clean_text'] = text
    return df

def count_punctuation(df:pd.DataFrame)-> pd.DataFrame:
    """Counts the number of punctuation marks in each tweet and removes them. 

    Args:
        df (pd.Dataframe): Pandas df containing a column called "text".
        
    Returns:
        pd.Dataframe: Pandas dataframe with punctuation counts
    """
    punctuation = string.punctuation

    def count_tweet(tweet): 
        punct_count =  Counter(char for char in tweet if char in punctuation)
        return {p: punct_count.get(p, 0) for p in punctuation}

    text = df['text']
    
    punctuation_counts = text.apply(count_tweet)
    
    X_counts = pd.DataFrame(punctuation_counts.tolist(), columns=list(punctuation))

    X_feature = pd.merge(df, X_counts, left_index=True, right_index=True)
    
    return X_feature

def capital_ratio(df: pd.DataFrame)->pd.DataFrame:
    """Generates new column with the ratio of capital to non-capital letters.

    Args:
        df (pd.DataFrame): Pandas df containing the columm 'text'

    Returns:
        pd.DataFrame: Same df with a new column 'capitals'
    """
    #remove punctuation to prevent skew
    def remove_punctuation(tweet):
        return ''.join(char for char in tweet if char not in string.punctuation)

    text = df['text'].apply(remove_punctuation)
    
    #remove @USER to prevent skew
    def remove_user(tweet: str):
        return tweet.replace("USER", "")
    
    text = text.apply(remove_user)

    #helper function to calculate capital ratio
    def cap_count(tweet: str) -> float:
        total_chars = len(tweet)
        upper_chars = sum(1 for char in tweet if char.isupper())
        return upper_chars / total_chars if total_chars > 0 else 0

    df['capitals'] = text.apply(cap_count)
    return df

def sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Generates sentiment scores using nltk sid module. 

    Args:
        df (pd.Dataframe): Df with column 'text' to analyze.

    Returns:
        pd.DataFrame: Df with additional columns negative, neutral, positive, and compound sentiment scores. 
    """
    
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

    def get_sentiment_scores(text):
        return sid.polarity_scores(text)
    
    df['sentiment_scores'] = df['text'].apply(get_sentiment_scores)

    # Create separate columns for each sentiment score
    df['negative'] = df['sentiment_scores'].apply(lambda x: x['neg'])
    df['neutral'] = df['sentiment_scores'].apply(lambda x: x['neu'])
    df['positive'] = df['sentiment_scores'].apply(lambda x: x['pos'])
    df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])

    # Drop original sentiment score column
    df = df.drop('sentiment_scores', axis=1)
    
    return df

def generate_topics(df: pd.DataFrame)-> pd.DataFrame:
    """Generates topics use lda module from genism. 

    Args:
        df (pd.DataFrame): Df containing column 'text' to be analyzed.

    Returns:
        pd.DataFrame: Df of features updated with topic column. 
    """
    text = df['clean_text']
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(text) for text in text]
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
    
    def get_topics(tweet):
        bow = dictionary.doc2bow(tweet)
        return max(lda_model[bow], key=lambda x: x[1])[0] if lda_model[bow] else -1
    
    df['topic'] = text.apply(get_topics)
    df['topic'] = df['topic'].astype('category')
    
    return df

def str_length(df: pd.DataFrame)-> pd.DataFrame:
    """Generates length column that describes length of original "text" column. 

    Args:
        df (pd.DataFrame): Pandas df containing column "text".

    Returns:
        pd.DataFrame: Original df with new column "length".
    """
    
    df['length'] = df['text'].apply(len)
    
    return df
 
def count_profanity(df: pd.DataFrame)-> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    def count_profane(tweet: list) -> int:
        return sum(1 for word in tweet if word in profane_words)
    
    df['profanity'] = df['text'].apply(count_profane)
    
    return df

def generate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from tweets

    Args:
        df (pd.Series): Pandas dataframe that contains a column called "text".

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    
    #pre process to remove stopwords, lowercase, and tokenize
    X_feature = preprocess(df)
    
    #create features
    X_feature = count_punctuation(X_feature)        
    X_feature = str_length(X_feature)    
    X_feature = capital_ratio(X_feature)
    X_feature = count_profanity(X_feature)
    X_feature = sentiment_analysis(X_feature)
    X_feature = generate_topics(X_feature)
    
    return X_feature

def generate_features(df: pd.DataFrame, file: str) -> pd.DataFrame:
    """Generates features based on text column in df using multiple frameworks. Saves to file or pulls from file if it exsists.

    Args:
        df (pd.DataFrame): Pandas df containing a column text.
        file (str): String containing the name for save
        
    Returns:
        pd.DataFrame: Dataframe of features. 
    """
    filename = f"{file}_log_reg.pkl"
    
    #either load file or generate file
    if (os.path.exists(filename)):
        print("pulling from file")
        with open(filename, 'rb') as f:
            features = pickle.load(f)
    else:
        features = generate_analysis(df)
        features = features.drop(['text', 'clean_text', 'label', 'category', 'demographic', 'perspective_score'], axis=1, errors='ignore')
        with open(filename, 'wb') as f:
            pickle.dump(features, f)
            
    return features

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales the features in a df using robust scaler from scikit-learn.

    Args:
        df (pd.DataFrame): Pandas df containing features.

    Returns:
        pd.DataFrame: Original df with features scaled. 
    """
    #scale features to improve model performance
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df))
    return X_scaled

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,labels = ['NOT', 'OFF']).ravel()
    return fp / (fp + tn)

def generate_model(X: pd.DataFrame, y: pd.Series)-> LogisticRegression:
    """Generates a logistic regression model using X and y to train with model selection based on flag.

    Args:
        X (pd.DataFrame): Design matrix for dataset.
        y (pd.Series): Target vector

    Returns:
        LogisticRegression: Logistic regression model. 
    """
    filename = f"model_log_reg.pkl"
    
    #either load file or generate file
    if (os.path.exists(filename)):
        print("pulling from file")
        with open(filename, 'rb') as f:
            model = pickle.load(f)
    else:   
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
    """Classifies datapoints based on a threshold

    Args:
        X (pd.Series): Pandas series of data points
        threshold (float): Float threshold to use as the classification boundary

    Returns:
        pd.Series: Data points classified as 'OFF' or 'NOT'
    """
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
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE, delimiter= "\t")
    
    #add target column for demographic file 
    df_demographic['label'] = "NOT"
    print("Data imported successfully!")
    
    #generate features and scales (scaling features improves model performance)
    X_train = generate_features(df_train, "features")
    X_train = scale_features(X_train)
    print("Features generated successfully!")
    
    #train model
    model = generate_model(X_train, df_train['label'])
    print("Model Trained!")
    
    #generate features
    X_dev = generate_features(df_dev, "dev")
    X_dev = scale_features(X_dev)
    X_demographics = generate_features(df_demographic, "demographics")
    X_demographics = scale_features(X_demographics)
    print("Dev features generated")
    
    #adjust threshold 
    df_dev['pred'] = select_threshold(model, X_dev, df_dev['label'], True)
    df_demographic['pred'] = select_threshold(model, X_demographics, df_demographic['label'], False)

    #run evaluations
    metrics_dev = run_metrics(df_dev['label'], df_dev['pred'])
    fpr_demo = fpr_demographic(df_demographic, df_demographic['pred'])  

    #output_file = 'Julie_Lawler_test.tsv'
    #df_test[['pred']].to_csv(output_file, sep='\t', index=False, header=True)

    print(metrics_dev)
    print(fpr_demo)