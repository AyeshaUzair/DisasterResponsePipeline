import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function takes the filepath of the disaster messages and 
    department categories in csv formats and merges them with respect to ids.

    Parameters
    ----------
    messages_filepath : TYPE str
        DESCRIPTION. filepath for messages csv file
    categories_filepath : TYPE str
        DESCRIPTION. filepath for categories csv file

    Returns
    -------
    df : TYPE object
        DESCRIPTION. merged pandas dataframe

    """
    # Read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """
    This function cleans the dataframe by performing multiple tasks.
    1. Values in the category column are split on ";" character so 
    each value becomes a seperate column.
    2. Columns are renamed using the values from the first row.
    3. Binary values are assigned to all category columns and replaced in the dataset

    Parameters
    ----------
    df : TYPE object
        DESCRIPTION. Dataframe object

    Returns
    -------
    df : TYPE object
        DESCRIPTION. Cleaned Dataframe object

    """
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
        
    # Few labels of 2 exist so they are replaced with 1
    categories_new.related.replace(2,1,inplace=True) 
    
    # Drop old categories column and append cleaned one
    df = df.drop('categories', axis=1)
    df = pd.concat([df,categories_new], axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    This function saves the cleaned dataframe and creates a database by
    taking the database filename as input and stores the dataframe as a table.

    Parameters
    ----------
    df : TYPE object
        DESCRIPTION. Dataframe object
    database_filename : TYPE str
        DESCRIPTION. Name of the created database

    Returns
    -------
    None.

    """
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
