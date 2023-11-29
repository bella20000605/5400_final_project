import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    def __init__(self):
        pass

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
    
    def test_prob

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


class LSTM(SarcasmModel):
    def __init__(self, dataset_path, vocab_size=10000, max_length=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None

        # Load and preprocess the dataset
        self.data = pd.read_csv(dataset_path)
        self._preprocess_data()

    def _preprocess_data(self):
        # Tokenize and pad sequences
        self.tokenizer.fit_on_texts(self.data['text'])
        sequences = self.tokenizer.texts_to_sequences(self.data['text'])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')

        # Encode labels
        label_encoding = {'notsarc': 0, 'sarc': 1}
        labels = self.data['class'].map(label_encoding).values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    def build_model(self):
        # Build LSTM model
        self.model = Sequential([
            Embedding(self.vocab_size, 16, input_length=self.max_length),
            LSTM(32),
            Dense(24, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, epochs=10):
        if self.model is None:
            print("Model is not built. Call build_model() first.")
            return

        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test))

    def evaluate(self):
        # Evaluate the model
        if self.model is None:
            print("Model is not built. Call build_model() first.")
            return

        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        return test_acc

    def predict(self, sentence):
        # Make a prediction
        sequence = self.tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        prediction = self.model.predict(padded)
        return 'Sarcastic' if prediction[0][0] > 0.5 else 'Not Sarcastic'

class RandomForestSarcasm(SarcasmModel):
    """
    Can perform Random Forest classification on a group of text files.
    It will read through each file in the training folder, find the TF-IDF vector space for the text data,
    Then it will fit the Random Forest model on them.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """
        Initializes the Random Forest classifier with the specified number of trees and max depth.
        :param n_estimators: Number of trees in the forest.
        :param max_depth: Maximum depth of the tree.
        :param random_state: Random state for reproducibility.
        """
        super().__init__(var_smoothing=0)  # var_smoothing not used in Random Forest
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def train(self, x_train, y_train):
        """
        Trains the Random Forest classifier on the given training data.
        :param x_train: feature training set.
        :param y_train: response training set.
        :return: TF-IDF training vector space and fitted training model.
        """
        tfidf = TfidfVectorizer()
        train_tfidf = tfidf.fit_transform(x_train)
        rf_model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        rf_model.fit(train_tfidf, y_train)
        return tfidf, rf_model

    def test(self, training_model, tfidf, x_test):
        """
        Tests the Random Forest model on the given test data.
        :param training_model: fitted training model.
        :param tfidf: TF-IDF training vector space.
        :param x_test: feature test set.
        :return: response variable prediction.
        """
        test_tfidf = tfidf.transform(x_test)
        y_pred = training_model.predict(test_tfidf)
        return y_pred

class LSTMDetector(SarcasmModel):
    """
    A class for detecting sarcasm in text using an LSTM neural network model.
    building an LSTM model, training the model on provided
    and making predictions on new text instances
    """
    def __init__(self, x_train, y_train, x_test, y_test, vocab_size=10000, max_length=100):
        """
        Initializes the LSTM model with the data set".
        :param x_train: training data of x.
        :param y_train: training data of y.
        :param x_test: testing data of x.
        :param y_test: testing data of y.
        :param vocab_size: maximum number of words to keep in the tokenizer, based on word frequency.
        :param max_length: maximum length of all sequences (number of words per text instance).
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None

        # Load and preprocess the dataset
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self._preprocess_data()

    def _preprocess_data(self):
        """
        Tokenizes and pads the text sequences and encodes the labels for training and testing datasets.
        """
        # Tokenize and pad sequences
        self.tokenizer.fit_on_texts(self.x_train['text'])
        sequences = self.tokenizer.texts_to_sequences(self.x_train['text'])
        self.X_train = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')

        sequences = self.tokenizer.texts_to_sequences(self.x_test['text'])
        self.X_test = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')

        # Encode labels
        label_encoding = {'notsarc': 0, 'sarc': 1}
        self.y_train_encoded = self.y_train['class'].map(label_encoding).values
        self.y_test_encoded = self.y_test['class'].map(label_encoding).values

    def build_model(self):
        """
        Builds an LSTM neural network model for sarcasm detection.
        """
        # Build LSTM model
        self.model = Sequential([
            Embedding(self.vocab_size, 16, input_length=self.max_length),
            LSTM(32),
            Dense(24, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, epochs=10):
        """
        Trains the LSTM model on the training data for a specified number of epochs.
        :param epochs: number of epochs for this model.
        """
        if self.model is None:
            print("Model is not built. Call build_model() first.")
            return

        # Train the model
        self.model.fit(self.X_train, self.y_train_encoded, epochs=epochs, validation_data=(self.X_test, self.y_test_encoded))

    def evaluate(self):
        """
        Evaluates the trained model on the test dataset and returns the accuracy.
        """
        # Evaluate the model
        if self.model is None:
            print("Model is not built. Call build_model() first.")
            return

        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test_encoded)
        return test_acc

    def predict(self, sentence):
        """
        Predicts whether a given sentence is sarcastic or not using the trained model.
        """
        # Make a prediction
        sequence = self.tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        prediction = self.model.predict(padded)
        return 'Sarcastic' if prediction[0][0] > 0.5 else 'Not Sarcastic'
    
# # Usage
# detector = LSTMDetector(x_train,y_train,x_test,y_test)
# detector.build_model()
# detector.train(epochs=10)
# accuracy = detector.evaluate()
# print(f"Model accuracy: {accuracy}")

# # Predict
# print(detector.predict("Oh, what a wonderful day!"))
