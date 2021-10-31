import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_response_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Graph 2 Parameters
    df1 = df[df['genre']=='direct']
    x1 = list(df1.columns[4:])
    y1 = df1.groupby(x1).count()['message']
    
    # Graph 3 Parameters
    df2 = df[df['genre']=='news']
    x2 = list(df2.columns[4:])
    y2 = df2.groupby(x2).count()['message']
    
    # Graph 4 Parameters
    df3 = df[df['genre']=='social']
    x3 = list(df1.columns[4:])
    y3 = df3.groupby(x3).count()['message']
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=x1,
                    y=y1
                )
            ],

            'layout': {
                'title': 'Distribution of Direct Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    # 'title': "Direct Messages",
                    'tickangle': 50
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=x2,
                    y=y2
                )
            ],

            'layout': {
                'title': 'Distribution of News Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    # 'title': "News Messages",
                    'tickangle': 50
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=x3,
                    y=y3
                )
            ],

            'layout': {
                'title': 'Distribution of Social Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    # 'title': "Social Messages",
                    'tickangle': 50
                },
                'margins':  260
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
