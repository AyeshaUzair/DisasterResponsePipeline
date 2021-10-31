import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    # Clean data
    categories_new = df['categories'].str.split(';', expand=True)
    row = list(categories_new.iloc[0])
    category_colnames = list(pd.Series(row).apply(lambda x : x[:-2]))
    categories_new.columns = category_colnames
    
    for column in categories_new:
        # Set each value to be the last character of the string
        categories_new[column] = categories_new[column].apply(lambda x: x[-1])
    
        # Convert column from string to numeric
        categories_new[column] = pd.to_numeric(categories_new[column])
    
    # Drop old categories column and append cleaned one
    df = df.drop('categories', axis=1)
    df = pd.concat([df,categories_new], axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    return df


def save_data(df, database_filename):
    # load to database
    # DB name used: 'DisasterResponseSp.db'
    engine = create_engine('sqlite:///' + database_filename )
    df.to_sql(name='Disaster_response_table', con=engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()