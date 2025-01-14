import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """Load and merge messages and categories datasets
    
    Args:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """

    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])

    return df



def clean_data(df):
    """Clean dataframe by removing duplicates and converting categories from strings 
    to binary values.
    
    Args:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
       
    Returns:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """

    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: str(x).rstrip ('- 0 1'))
    categories.columns = category_colnames.str.replace(',', '')
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1].astype(int)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df = df[df['strom'] != 2]
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis = 1 )
    df = df.drop_duplicates(subset=['id', 'message'])
    assert len(df[df.duplicated()]) == 0
    return df

def save_data(df, database_filename):
    """Save cleaned data into an SQLite database.
    Args:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messageCat', engine, index=False, if_exists='replace') 
    print(f"Data saved to SQLite database: {database_filename}")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        try:
            df = load_data(messages_filepath, categories_filepath)
        except Exception as e:
            print(f"Error loading data: {e}")

        print('Cleaning data...')
        try:
            df = clean_data(df)
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        try:
            save_data(df, database_filepath)
            print('Cleaned data saved to database!')
        except Exception as e:
            print(f"Error saving data: {e}")
        
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