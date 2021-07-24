#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sqlalchemy import create_engine
import sys


# In[ ]:


def load_data(messages_filepath, categories_filepath):
    """Load & merge messages & categories datasets
    
    inputs:
    messages_filepath: Filepath for csv file containing messages dataset.
    categories_filepath: Filepath for csv file containing categories dataset.
       
    outputs:
    df: Dataframe containing merged content of messages & categories datasets.
    """

    #Load Messages Dataset
    messages = pd.read_csv(messages_filepath)
    
    #Load Categories Dataset
    categories = pd.read_csv(categories_filepath)
    
    #Merge two datasets
    df = messages.merge(categories, on = 'id')
    
    return df


# In[ ]:


def clean_data(df):
    """Clean dataframe by removing duplicates & converting categories from strings 
    to binary values.
    
    Input:
    df: Dataset.
       
    Return:
    df: Dataframe containing cleaned dataset.
    """
    #create a df for each indiviual category 
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
   
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df = df[df['related'] != 2]
    df.drop_duplicates(inplace = True)
    return df


# In[ ]:


def save_data(df, database_filename):
    """Save into  SQLite database.
    
    inputs:
    df: Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


# In[ ]:


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
        print('Please provide the filepaths of the messages and categories '              'datasets as the first and second argument respectively, as '              'well as the filepath of the database to save the cleaned data '              'to as the third argument. \n\nExample: python process_data.py '              'disaster_messages.csv disaster_categories.csv '              'DisasterResponse.db')


# In[ ]:


if __name__ == '__main__':
    main()

