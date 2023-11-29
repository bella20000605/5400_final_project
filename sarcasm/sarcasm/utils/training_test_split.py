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
    :return: no return, save the six datasets x_train, x_test, y_train, y_test, sarcasm_train, sarcasm_test in the data folder
    """
    df = pd.read_csv(path)
    X = df['text']
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    sarcasm_train = pd.concat([X_train, y_train], axis=1)
    sarcasm_test = pd.concat([X_test, y_test], axis=1)

    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    sarcasm_train.to_csv('data/sarcasm_train.csv', index=False)
    sarcasm_test.to_csv('data/sarcasm_test.csv', index=False)


combine_data("data/GEN-sarc-notsarc.csv","data/HYP-sarc-notsarc.csv","data/RQ-sarc-notsarc.csv")
build_dataframe("data/sarcasm.csv")

