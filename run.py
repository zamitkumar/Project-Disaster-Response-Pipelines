import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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
                'barmode': 'group'
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
            }
        }
        graphs.append(graph3)

    # Graph 4: Top Categories by Message Count
    if not df.empty:
        top_categories = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False).head(10)
        graph4 = {
            'data': [
                Bar(
                    x=top_categories.index.tolist(),
                    y=top_categories.values.tolist(),
                    orientation='h'
                )
            ],
            'layout': {
                'title': 'Top 10 Categories by Message Count',
                'yaxis': {'title': "Category"},
                'xaxis': {'title': "Message Count"},
            }
        }
        graphs.append(graph4)

    # Graph 5: Message Length Distribution (Histogram)
    if 'message' in df.columns:
        df['message_length'] = df['message'].str.len()
        graph5 = {
            'data': [
                {
                    'type': 'histogram',
                    'x': df['message_length'],
                    'nbinsx': 20,
                }
            ],
            'layout': {
                'title': 'Message Length Distribution',
                'yaxis': {'title': "Frequency"},
                'xaxis': {'title': "Message Length"},
            }
        }
        graphs.append(graph5)

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()