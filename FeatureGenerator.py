import pandas as pd
import pickle
import os.path
import string
import nltk
import numpy as np
import sklearn
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
from sklearn.preprocessing import RobustScaler

#credit for this list goes to better_profanity package
profane_words = pd.read_csv("Profanity_Wordlist.txt", header =None, names = ['word'])
profane_words = profane_words['word'].tolist()

class FeatureGenerator:
    """
    A class to generate features for tweets. 
    
    The class contains static methods that operate on pandas dataframes. 
    """
    
    def __init__(self, df):
        self.df = df
        self.threshold = 0.5
        self.feature_names = []

    def get_features(self)-> pd.DataFrame:
        self.features = self.df[self.feature_names]
        return self.features

    def preprocess(self):
        """Preprocess data by tokenizing, lowercasing, and removing stopwords. 

        Args:
            df (pd.Dataframe): Pandas dataframe of containing 'text' column to process.

        Returns:
            pd.Dataframe: Original df with new column 'clean_text'.
        """
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        stop_words.add("user")

        text = self.df['text'].str.lower()
        
        def remove_punctuation(tweet):
            return ''.join(char for char in tweet if char not in string.punctuation)

        text = text.apply(remove_punctuation)

        text = text.apply(word_tokenize)
        
        def filter_stopwords(sentence):
            return [w for w in sentence if not w.lower() in stop_words]

        text = text.apply(filter_stopwords)
        
        self.df['clean_text'] = text

    def add_punctuation_count(self):
        """
        Counts the number of punctuation marks in each tweet.
        
        """
        punctuation = string.punctuation

        def count_tweet(tweet): 
            punct_count =  Counter(char for char in tweet if char in punctuation)
            return {p: punct_count.get(p, 0) for p in punctuation}

        text = self.df['text']
        
        punctuation_counts = text.apply(count_tweet)
        
        X_counts = pd.DataFrame(punctuation_counts.tolist(), columns=list(punctuation))

        self.df = pd.merge(self.df, X_counts, left_index=True, right_index=True)
        self.feature_names.extend(X_counts.columns)

    def add_capital_ratio(self):
        """Generates new column with the ratio of capital to non-capital letters.

        Args:
            df (pd.DataFrame): Pandas df containing the columm 'text'

        Returns:
            pd.DataFrame: Same df with a new column 'capitals'
        """
        #remove punctuation to prevent skew
        def remove_punctuation(tweet):
            return ''.join(char for char in tweet if char not in string.punctuation)

        text = self.df['text'].apply(remove_punctuation)
        
        #remove @USER to prevent skew
        def remove_user(tweet: str):
            return tweet.replace("USER", "")
        
        text = text.apply(remove_user)

        #helper function to calculate capital ratio
        def cap_count(tweet: str) -> float:
            total_chars = len(tweet)
            upper_chars = sum(1 for char in tweet if char.isupper())
            return upper_chars / total_chars if total_chars > 0 else 0

        self.df['capitals'] = text.apply(cap_count)
        self.feature_names.append('capitals')

    def add_sentiment_analysis(self):
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
        
        self.df['sentiment_scores'] = self.df['text'].apply(get_sentiment_scores)

        # Create separate columns for each sentiment score
        self.df['negative'] = self.df['sentiment_scores'].apply(lambda x: x['neg'])
        self.df['neutral'] = self.df['sentiment_scores'].apply(lambda x: x['neu'])
        self.df['positive'] = self.df['sentiment_scores'].apply(lambda x: x['pos'])
        self.df['compound'] = self.df['sentiment_scores'].apply(lambda x: x['compound'])

        # Drop original sentiment score column
        self.df = self.df.drop('sentiment_scores', axis=1)
        self.feature_names.extend(['negative', 'neutral', 'positive', 'compound'])

    def add_str_length(self):
        """Generates length column that describes length of original "text" column. 

        Args:
            df (pd.DataFrame): Pandas df containing column "text".

        Returns:
            pd.DataFrame: Original df with new column "length".
        """
        
        self.df['length'] = self.df['text'].apply(len)
        self.feature_names.append('length')
    
    def add_count_profanity(self):
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        def count_profane(tweet: list) -> int:
            return sum(1 for word in tweet if word in profane_words)
        
        self.df['profanity'] = self.df['text'].apply(count_profane)
        self.feature_names.append('profanity')

    def scale_features(self):
        """Scales the features in a df using robust scaler from scikit-learn.

        Args:
            df (pd.DataFrame): Pandas df containing features.

        Returns:
            pd.DataFrame: Original df with features scaled. 
        """
        #scale features to improve model performance
        scaler = RobustScaler()
        features = self.get_features()
        self.features = pd.DataFrame(scaler.fit_transform(features))