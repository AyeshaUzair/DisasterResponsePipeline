import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import joblib
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = engine.table_names()
    df = pd.read_sql_table(table_name[0], con=engine)
    
    X = df.message
    y = df.loc[:, 'related':'direct_report']
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokenized_and_stopwords = [words for words in tokenized if not words in stop_words] 
    tokenized_and_stopwords_lemm = [WordNetLemmatizer().lemmatize(w) for w in tokenized_and_stopwords]
    return tokenized_and_stopwords_lemm


def build_model():
        
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])
    
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for cat in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[cat], y_pred[:,num], average='weighted')
        results.at[num+1, 'Category'] = cat
        results.at[num+1, 'f_score'] = f_score
        results.at[num+1, 'precision'] = precision
        results.at[num+1, 'recall'] = recall
        num += 1
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    
    # y_pred = model.predict(X_test)
    # cr = classification_report(Y_test, y_pred , target_names = category_names )
    # print(cr)



def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.values, Y_train.values)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
