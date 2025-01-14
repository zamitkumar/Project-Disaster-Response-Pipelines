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
    """
    Loads data from a SQLite database and returns features and labels.

    This function connects to a SQLite database file using SQLAlchemy, queries 
    the 'messageCat' table, and loads the data into a pandas DataFrame. The 
    function splits the data into input features (`X`) and target labels (`y`), 
    where `X` is the 'message' column, and `y` contains columns 4 to 40 as the 
    target variables.

    Parameters:
    database_filepath (str): The file path to the SQLite database.

    Returns:
    tuple: A tuple containing:
        - X (pandas.Series): The input features (messages) from the database.
        - y (pandas.DataFrame): The target labels (columns 4 to 40) from the database.
    """
    # Create a connection to the SQLite database
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # Load the data from the 'messageCat' table into a pandas DataFrame
    df = pd.read_sql('SELECT * FROM messageCat', engine)

    # Display the first 20 rows of the dataframe (for inspection)
    df.head(20)

    # Split the dataframe into input features (X) and target labels (y)
    X = df['message']  # 'message' column as the input feature
    y = df.iloc[:, 4:40]  # Columns 4 to 40 as the target labels

    # Return the input features and target labels
    return X, y

def tokenize(texts):
    """
    Tokenizes and processes a given text string by removing non-alphanumeric characters, 
    converting to lowercase, tokenizing, and lemmatizing the words.

    This function performs the following steps:
    1. Removes all non-alphanumeric characters (including punctuation).
    2. Converts the text to lowercase.
    3. Tokenizes the text into individual words (tokens).
    4. Lemmatizes each token to its base form.
    5. Strips any leading/trailing whitespaces from each token.

    Parameters:
    texts (str): The input text string to be tokenized and processed.

    Returns:
    list: A list of cleaned and lemmatized tokens.
    """
    
    # Remove non-alphanumeric characters and convert text to lowercase
    texts = re.sub(r'[^\w\s]', " ", texts.lower())
    
    # Tokenize the text into individual words
    tokens = word_tokenize(texts)
    
    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each token and remove any leading/trailing whitespace
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    
    return clean_tokens

def build_model():
    """
    Function to build a machine learning pipeline with a multi-output random forest classifier 
    and set up a grid search for hyperparameter tuning.

    Steps performed in this function:
        1. Create a pipeline with three steps:
            - CountVectorizer: Tokenizes the text data.
            - TfidfTransformer: Converts term frequency into a normalized form (TF-IDF).
            - MultiOutputClassifier: A wrapper around the Random Forest Classifier to handle multi-output classification.
            
        2. Set up the grid search with hyperparameter options for:
            - ngram range (only unigrams are considered in this case).
            - The number of estimators (trees) for the random forest classifier.
            - The minimum number of samples required to split an internal node in the random forest.

    Returns:
        gcv: A GridSearchCV object ready to tune hyperparameters and fit to training data.
    """
    
    # Define the pipeline steps
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),  # Text vectorization step
        ('tfidf_transformer', TfidfTransformer()),            # TF-IDF transformation step
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output random forest classifier
    ])
    
    # Specify parameters for grid search
    parameters = {
        'vectorizer__ngram_range': [(1, 1)],  # Use unigrams for tokenization
        'classifier__estimator__n_estimators': [50],  # Number of trees in the random forest classifier
        'classifier__estimator__min_samples_split': [2]  # Minimum samples required to split a node in the trees
    }
    
    # Create grid search object for hyperparameter tuning
    gcv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, verbose=3, n_jobs=-1)
    
    # Return the grid search object
    return gcv
   
def train_pipeline(X, y):
    """
    Splits the input features (X) and target labels (y) into training and test sets.
    It also converts the features in the training and test sets to strings.

    Parameters:
    X (array-like or DataFrame): The input features (independent variables).
    y (array-like or Series): The target labels (dependent variable).

    Returns:
    tuple: A tuple containing the following elements:
        - X_train (list): The training set of input features, converted to strings.
        - X_test (list): The test set of input features, converted to strings.
        - y_train (array-like or Series): The training set of target labels.
        - y_test (array-like or Series): The test set of target labels.
    """
    # Split the data into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Convert the features in the training and test sets to strings
    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]

    return X_train, X_test, y_train, y_test

def get_evaluation_metrics(actual, predicted, column_names, average='macro'):
    """
    Calculate evaluation metrics for a multi-label machine learning model, 
    including Accuracy, Precision, Recall, and F1 Score for each label.

    Args:
        actual (array-like): Ground truth labels, typically a 2D array or a structure with the same shape as the predicted labels.
                             Each row represents a sample and each column represents a label.
        predicted (array-like): Predicted labels from the model, in the same shape as `actual`.
                                Each row corresponds to a sample, and each column corresponds to a label.
        column_names (list of str): A list of field names or labels corresponding to each column in the `actual` and `predicted` arrays.
                                    The length of this list should match the number of columns in `actual` and `predicted`.
        average (str, optional): The averaging method to use for precision, recall, and F1 score calculations. 
                                 Options are 'micro', 'macro', 'weighted', and 'samples'. Default is 'macro'.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics (Accuracy, Precision, Recall, and F1 Score) for each label.
                      The index of the DataFrame will be the field names (i.e., the labels), and the columns will include the metrics.

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
    """
    Save the trained model to a file.

    This function serializes the given model object using the `pickle` module
    and writes it to a file specified by `model_filepath`. If successful, 
    it prints a message confirming the model has been saved. If an error occurs 
    during the saving process, it prints the error message.

    Args:
        model (object): The trained machine learning model to be saved.
        model_filepath (str): The file path where the model will be saved.

    Returns:
        None
    """
    try:
        # Serialize and save the model using pickle
        pickle.dump(model, open(model_filepath, 'wb'))
        print(f"Model successfully saved to {model_filepath}")
    except Exception as e:
        # If an error occurs, print the error message
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