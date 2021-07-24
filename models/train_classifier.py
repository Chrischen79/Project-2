#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
import pickle
import os.path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import re


# In[ ]:


def load_data(database_filepath):
    """
       Function:
       load data

       Args:
       database_filepath: location path of the database

       Return:
       X: Message features
       Y:  target labels
       category (list of str) : target labels list
       """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre','child_alone'], axis = 1)
    category_names = list(Y.columns.values)
    #print(X)
    #print(Y.columns)
    return X, Y, category_names


# In[ ]:


def tokenize(text):
    """
    Function: split text into words and return the root form of the words
    Args:
      text(str): the message
    Return:
      lemm(list of str): a list of the root form of the message words
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]
    
    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemm


# In[ ]:


def build_model():
    """
     Function: build a RF model to classify messages

     Return:
       cv(list of str): the classification model
    """

    # Create a pipeline
    pipeline_rf = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Create Grid search parameters. To avoid too long computation time, only two values of one parameter are used.
    #clf__estimator__n_estimators should have larger values, such as 100 or 500 or 1000.
    parameters = {
        'clf__estimator__n_estimators': [5, 10],
    }

    CV_rfc = GridSearchCV(pipeline_rf, param_grid=parameters)

    return CV_rfc


# In[ ]:


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.
    Args:
    model: the classification model
    X_test: test messages
    Y_test: test target
    """
    Predict_label = model.predict(X_test)
    True_label = np.array(Y_test)
    metrics = []
    
    for i in range(len(category_names)):
        accuracy = accuracy_score(True_label[:, i], Predict_label[:, i])
        precision = precision_score(True_label[:, i], Predict_label[:, i])
        recall = recall_score(True_label[:, i], Predict_label[:, i])
        f1 = f1_score(True_label[:, i], Predict_label[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    metrics = np.array(metrics)
    data_metrics = pd.DataFrame(data = metrics, index = category_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    print(data_metrics)
    return data_metrics
    


# In[ ]:


def save_model(model, model_filepath):
    """
    Function: Save a pickle file of the model
    Args:
    model: the classification model
    model_filepath (str): the path of pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))    
   


# In[ ]:


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


# In[ ]:


if __name__ == '__main__':
    main()

