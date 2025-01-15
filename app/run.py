import json
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
from starting_verb_extractor import StartingVerbExtractor
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        No fitting needed for this transformer.
        """
        return self

    def transform(self, X):
        """
        Extracts whether the first word in the text is a verb or not.
        """
        def starting_verb(text):
            # Check for empty or whitespace-only strings
            if not text or not text.strip():
                return 0
            words = text.split()
            # Check if the first word is title case
            if len(words) > 0 and words[0].istitle():
                return 1
            return 0

        # Ensure input X is iterable
        if not isinstance(X, (list, np.ndarray)):
            X = X.tolist()

        # Apply the starting_verb function to each element in X
        features = [starting_verb(text) for text in X]

        # Return a 2D NumPy array (scikit-learn requires this format)
        return np.array(features).reshape(-1, 1)
    
def tokenize(text):
    """
    Tokenize, normalize, and lemmatize input text.

    Args:
        text (str): Input text to process.

    Returns:
        list: Cleaned and lemmatized tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
try:
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('messageCat', engine)
except Exception as e:
    print(f"Error loading database or table: {e}")
    df = None

# load model
try:
    model = joblib.load("../models/classifier.pkl")
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    model = None


# Index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Render the main page with visuals and user input form.
    """
    if 'message_length' in df.columns:
        df.drop('message_length', axis=1, inplace=True)
    
        
    # Extract data needed for visuals
    if not df.empty and all(col in df.columns for col in ['related', 'genre', 'message']):
        genre_related = df[df['related'] == 1].groupby('genre').count()['message']
        genre_not_related = df[df['related'] == 0].groupby('genre').count()['message']
        genre_names = list(genre_related.index)
    else:
        genre_related, genre_not_related, genre_names = [], [], []

    # Calculate category proportions
    if not df.empty and all(col in df.columns for col in ['id', 'message', 'original', 'genre']):
        cat_prop = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum() / len(df)
        cat_prop = cat_prop.sort_values(ascending=False)
        cat_names = list(cat_prop.index)
    else:
        cat_prop, cat_names = [], []

    # Create visuals
    graphs = []

    # Graph 1: Distribution of Genres (Related vs Not Related)
    if genre_names:
        graph1 = {
            'data': [
                Bar(x=genre_names, y=genre_related, name='Genre Related'),
                Bar(x=genre_names, y=genre_not_related, name='Genre Not Related')
            ],
            'layout': {
                'title': 'Distribution of Message Genres and Related Status',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"},
                'barmode': 'group',
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
                
            }
        }
        graphs.append(graph1)

    # Graph 2: Proportion of Messages by Category
    if cat_names:
        graph2 = {
            'data': [
                Bar(x=cat_names, y=cat_prop)
            ],
            'layout': {
                'title': 'Proportion of Messages by Category',
                'yaxis': {'title': "Proportion"},
                'xaxis': {'title': "Category", 'tickangle': -45},
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 100}, # Adjust spacing around the graph
            
            }
        }
        graphs.append(graph2)

    # Graph 3: Genre Distribution (Pie Chart)
    if not df.empty:
        genre_counts = df['genre'].value_counts()
        graph3 = {
            'data': [
                {
                    'type': 'pie',
                    'labels': genre_counts.index.tolist(),
                    'values': genre_counts.values.tolist()
                }
            ],
            'layout': {
                'title': 'Genre Distribution',
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},  # Adjust spacing around the graph
            
            }
        }
        graphs.append(graph3)



    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Render the results page with classification for user query.
    """
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    if model:
        try:
            classification_labels = model.predict([query])[0]
            classification_results = dict(zip(df.columns[4:], classification_labels))
        except Exception as e:
            print(f"Error during prediction: {e}")
            classification_results = {}
    else:
        classification_results = {}

    # Render the go.html page
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Run the Flask app.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()