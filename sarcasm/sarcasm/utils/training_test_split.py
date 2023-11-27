import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def combine_data(path1,path2,path3):
    """
    Combine 3 sarcasm datasets into 1 dataset and reorder the id
    :param folder: 3 path to the 3 datasets
    :return: no return, save the combined dataset in the data folder
    """
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    combined_data = pd.concat([df1, df2, df3], axis=0)
    combined_data = combined_data.reset_index()
    combined_data = combined_data.drop(columns=['id','index'])
    combined_data['id'] = range(1, len(combined_data) + 1)
    desired_order = ['class', 'id', 'text'] 
    combined_data = combined_data[desired_order]
    combined_data.to_csv('data/sarcasm.csv', index=False)

def build_dataframe(path):
    """
    Takes as input a directory containing the sarcasm data.
    The function will split the data into training and test using 80/20 split.
    :param folder: a path to the dataset
    :return: a tuple of 4 arguments x_training, y_training, x_test, y_test
    """
    df = pd.read_csv(path)
    X = df['text']
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test

combine_data("data/GEN-sarc-notsarc.csv","data/HYP-sarc-notsarc.csv","data/RQ-sarc-notsarc.csv")