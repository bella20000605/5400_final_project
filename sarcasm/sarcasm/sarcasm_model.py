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


class SarcasmDetector(SarcasmModel):
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


# # Usage
# detector = SarcasmDetector('path_to_dataset.csv')
# detector.build_model()
# detector.train(epochs=10)
# accuracy = detector.evaluate()
# print(f"Model accuracy: {accuracy}")

# # Predict
# print(detector.predict("Oh, what a wonderful day!"))
