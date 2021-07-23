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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_per = round(100*genre_counts/genre_counts.sum(), 1)
    genre_names = list(genre_counts.index)
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    cat_num = cat_num.sort_values(ascending = False)
    cat_per = round(100*cat_num/cat_num.sum(), 1)
    cat = list(cat_num.index)

    colors = ['yellow', 'green', 'red']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                "uid": "aaeeddf",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": genre_per,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                    "#3a2cd4",
                    "#db4a07",
                    "#25a308"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_per
              }
            ],
            "layout": {
              "title": "Pie Chart showing Percentages of Messages by Genre"
            }
        },
        {
            "data": [
              {
                "type": "bar",
                "x": cat,
                "y": cat_per,
                "marker": {
                  "color": '#25a308'}
                }
            ],
            "layout": {
              "title": "Percentage of Messages by Category",
              'yaxis': {
                  'title': "Pecentage"
              },
              'xaxis': {
                  'title': "Genre"
             
              },
              'barmode': 'group'
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