import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SarcasmModel:
    """ 
    Interface for sarcasm detection model
    """ 
    def __init__(self,var_smoothing):
        self.var_smoothing = var_smoothing
    def __str__(self):
        f"SarcasmModel(Smoothing Factor='{self.SarcasmModel}')"
    def __repr__(self):
        f"SarcasmModel(Smoothing Factor='{self.SarcasmModel}')"
    def train(self):
        pass
    def test(self):
        pass
    
class Logistic(SarcasmModel):
    """
    Can perform Logistic Regression on a group of text files.
    It will read through each file in the training folder, find the TF-IDF vector space for the text data
    Then it will fit the Logistic Regression model on them.
    """
    def __init__(self,var_smoothing = 0.1):
        self.var_smoothing = var_smoothing

    def train(self,x_train,y_train):
        """
        :param x_train: feature training set, y_train: response training set 
        :return: TF-IDF training vector space and fitted traing model
        """
        tfidf = TfidfVectorizer()
        train_tfidf = tfidf.fit_transform(x_train)
        LogisticModel = LogisticRegression()
        LogisticModel.fit(train_tfidf, y_train)
        return tfidf, LogisticModel
    
    def test(self,training_model,tfidf,x_test):
        """
        :param training_model: fitted traing model, tfidf: TF-IDF training vector space, x_test: feature test set 
        :return: response variable predicition
        """
        test_tfidf = tfidf.transform(x_test)
        y_pred = training_model.predict(test_tfidf)
        return y_pred

class NaiveBayes(SarcasmModel):
    """
    Can perform Naive Bayes classification on a group of text files.
    It will read through each file in the training folder and fit the Naive Bayes model on them.
    """

    def __init__(self,var_smoothing = 0.1):
        self.var_smoothing = var_smoothing
    
    def train(self,x_train,y_train):
        """
        :param x_train: feature training set, y_train: response training set 
        :return: count vectorized training space and fitted traing model
        """
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(x_train)
        classifier = MultinomialNB(alpha=self.var_smoothing)
        classifier.fit(X_train_vectorized, y_train)
        return vectorizer,classifier
    
    def test(self,training_model,vectorizer,x_test):
        """
        :param training_model: fitted traing model, vectorizer: count vectorized traing space, x_test: feature test set 
        :return: response variable predicition
        """
        X_test_vectorized = vectorizer.transform(x_test)
        y_pred = training_model.predict(X_test_vectorized)
        return y_pred

