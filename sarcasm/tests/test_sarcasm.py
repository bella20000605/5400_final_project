import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sarcasm.sarcasm import Logistic, NaiveBayes, RandomForestSarcasm, LSTMDetector

# Laod Data
GEN_data = pd.read_csv('/test/data/GEN-sarc-notsarc.csv')
HYP_data = pd.read_csv('/test/data/HYP-sarc-notsarc.csv')
RQ_data = pd.read_csv('/test/data/RQ-sarc-notsarc.csv')
sarcasm_data = pd.read_csv('/test/data/sarcasm.csv')

datasets = [GEN_data, HYP_data, RQ_data, sarcasm_data]

# logistic regression
@pytest.mark.parametrize("data", datasets, is)
def test_logistic_model(data):
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.2, random_state=42)
    model = Logistic()
    tfidf, trained_model = model.train(x_train, y_train)
    y_pred = model.test(trained_model, tfidf, x_test)

# Naive Bayes
@pytest.mark.parametrize("data", datasets)
def test_naive_bayes_model(data):
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.2, random_state=42)
    model = NaiveBayes()
    vectorizer, trained_model = model.train(x_train, y_train)
    y_pred = model.test(trained_model, vectorizer, x_test)

# Random Forest
@pytest.mark.parametrize("data", datasets)
def test_random_forest_model(data):
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.2, random_state=42)
    model = RandomForestSarcasm()
    tfidf, trained_model = model.train(x_train, y_train)
    y_pred = model.test(trained_model, tfidf, x_test)

# LSTM
@pytest.mark.parametrize("data", datasets)
def test_lstm_model(data):
    x_train, x_test, y_train, y_test = train_test_split(data, test_size=0.2, random_state=42)
    model = LSTMDetector()
    trained_model, tokenizer = model.train(x_train, y_train)
    y_pred = model.test(x_test, trained_model, tokenizer)
  
