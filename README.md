# T4G-Mini-Project-1---PerspectiveAPI-Hate-Speech-Classification

This hate speech classifier is part of the T4G at Pitt Responsible Tech Qickstart Series, a series of mini-projects to explore each pillar of responsible tech. You can use scores generated from PerspectiveAPI and a custom logistic regression model to classify tweets as hate speech. After creating a custom model, you can test your performance and find out if your model is biased.
 
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

Instructions for installing the project. For example:

```bash
# Clone the repository
git clone https://github.com/chase-lahner/T4G-Mini-Project-1---PerspectiveAPI-Hate-Speech-Classification

# Navigate to the project directory
cd T4G-Mini-Project-1---PerspectiveAPI-Hate-Speech-Classification

# Install dependencies
# Using pip
pip install -r requirements/pip.txt
# Using conda
conda create --name HSC_env --file requirements/conda.txt
conda activate HSC_env
```

## How to Use

### Running Main 

```bash
python main
```

### Classes 

#### 1. FeatureGenerator
- **Purpose**: Extracts features from tweet text and prepares data for classification.
- **Key Methods**:
  - `__init__(df)`: Creates FeatureGenerator obejct from df. Note the df must contain the colum 'text'.
  - `preprocess()`: Takes input text and generates feature vectors.
  - `get_features()`: Returns a df with the generated features.
  - `add_punctuation_count()`: Scales features.
  - `add_capital_ratio()`: Ratio of uppercase to lowercase letters.
  - `add_sentiment_analysis()`: Scores tweets for positive, negative, netural, and compound sentiment.
  - `add_str_length()`: Length of tweet
  - `add_count_profanity()`: Counts profane words.
  - `add_word_count('word')`: Counts occurances the given word.
  - `add_emotion_count('emotion')`: Counts occurances of words related to given emotion. Can be 'anger', 'anticipation', 'disgust', 'fear', 'negative', 'positive', 'sadness', 'surprise', 'trust'

- **Usage**: 
```python
  from ScoreClassifier import *

  fg = FeatureGenerator(df) 
  fg.preprocess # Must preprocess before generating features
  fg.add_str_length() # Some methods do not take arguments
  fg.add_emotion_count('anger') # Others accept a str
  fg.scale_features() # Optionally scale features
  X = fg.get_features()
```
#### 2. HateSpeechClassifier
- **Purpose**: Create custom sklearn model to classify tweets as hate speech. 
- **Key Methods**:
  - `__init__(X, y)`: Create HateSpeechClassifier Object. X should be a df of features and y should be series of labels. 
  - `generate_model()`: Returns a LogisticRegression model.
  - `select_threshold()`: Adjusts classification threshold to lowest sensitivity with accuracy above 70%. 
  - `predict()`: Classifies as hate speech. Returns pd.series of predictions.
 - **Usage**: 
```python
  from HateSpeechClassifier import *
  
  classifier = HateSpeechClassifier(X, y)
  model = classifier.generate_model()
  classifier.select_threshold() 
  df_dev['pred'] = classifier.predict(X_dev)
```
#### 3. MetricsGenerator
- **Purpose**: Return key metrics from custom lotistic regression model to evaluate effectiveness
- **Key Methods**:
  - `__init__(df_dev, df_demographic, dev_y_pred, dem_y_pred)`: create MetricsGenerator object, given two dataframes and associated model predictions
  - `fpr()`: calculates false positive rate for a df that contains actual and prediction values
  - `fpr_demographic()`: calculates false postitive rate for each unique demographic in the dataframe
  `run_metrics()`: returns accuracy, precision, recall, and fscore of given model

#### 4. ScoreClassifierClass
- **Purpose**: Create a classifier model to classify messages based off PerspectiveAPI scores
- **Key Methods**:
  - `__init__(df_dev, def_demographic, threshold)`: create a ScoreClassifierClass object, given two dataframes and a classification threshold
  - `classify_dev()`: classifies df_dev dataframe based on threshold
  - `classify_demographic()`: classifies df_demographic dataframe based on threshold
  - `run_metrics_dev()`: returns accuracy, precision, recall, and fscore of dev model
  - `run_metrics_dem()`: returns accuracy, precision, recall, and fscore of dev model
  - `fpr()`: returns fpr of a df that contains the actual and prediction values
  - `fpr_demographic()`: returns fpr for each unique demographic in a dataframe

## Acknowledgements
- **[Professor Lorraine Li](https://lorraine333.github.io/)** - Thank you to Prof Li for providing the datasets and conceptual inspiration for this project.

## Citations
1. **[NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)**
   - Mohammad, Saif M., and Turney, Peter D. (2013)
2. **[Better Profanity Dataset](https://github.com/snguyenthanh/better_profanity)**

## License
This project is licensed under the [MIT License](LICENSE).

## Contact

- Name: Julie Lawler
- Email: jal355@pitt.eud
- GitHub: [jal355](https://github.com/jal355)
modified by Emilie Perman
