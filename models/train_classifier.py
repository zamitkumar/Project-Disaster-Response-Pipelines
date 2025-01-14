import sys
import pandas as pd
import numpy as np
import nltk
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
# Download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.datasets import make_multilabel_classification

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    #print(engine.table_names())
    df = pd.read_sql ('SELECT * FROM messageCat', engine)
    df.head(20)
    X = df['message']
    y = df.iloc[:, 4:40]
    return X, y
    

def tokenize(texts):
# Remove non-alphanumeric characters
    texts = re.sub(r'[^\w\s]', " " , texts.lower())
    
    # Tokenize the text
    tokens = word_tokenize(texts)
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize and clean each token
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    
    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),  
       ('tfidf_transformer', TfidfTransformer()),           
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))  
    ])
    return pipeline
def train_pipeline(X, y ):
    #X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]

    return X_train, X_test, y_train, y_test

def get_evaluation_metrics(actual, predicted, column_names, average='macro'):
    """
    Calculate evaluation metrics for a multi-label machine learning model.

    Args:
        actual (array-like): Ground truth labels (2D array or equivalent structure).
        predicted (array-like): Predicted labels from the model (2D array or equivalent structure).
        column_names (list of str): List of field names corresponding to each label.

    Returns:
        pd.DataFrame: A DataFrame containing Accuracy, Precision, Recall, and F1 Score for each label.
    """

    # Ensure inputs are NumPy arrays for consistent processing
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Calculate metrics for each field and store in a list of dictionaries
    metrics = []
    for i, column in enumerate(column_names):
        metrics.append({
            "Field": column,
            "Accuracy": accuracy_score(actual[:, i], predicted[:, i]),
            "Precision": precision_score(actual[:, i], predicted[:, i], average=average),
            "Recall": recall_score(actual[:, i], predicted[:, i], average=average),
            "F1": f1_score(actual[:, i], predicted[:, i], average=average)
        })

    # Convert metrics into a DataFrame for better presentation
    metrics_df = pd.DataFrame(metrics).set_index("Field")
    return metrics_df

def save_model(model, model_filepath):
    #best_model = build_model()
    """Save the trained model to a file."""
    try:
        pickle.dump(model, open(model_filepath, 'wb'))
        print(f"Model successfully saved to {model_filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")
    

def main():
    

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        X, y = load_data(database_filepath)
        #call pipeline
        X_train, X_test, y_train, y_test=  train_pipeline(X, y)
        
        #call model
        print('Building model...')
        model = build_model()
        #fit 
        print('Training model...')
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        col_names = list(y.columns.values)
        #Train
        print('Evaluating model...')
        train_df = get_evaluation_metrics(y_train, y_train_pred, col_names, average='macro')
        print(train_df)
        #Test
        test_df_first = get_evaluation_metrics(y_test, y_test_pred, col_names, average='macro')
        print(test_df_first)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
         
        
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
