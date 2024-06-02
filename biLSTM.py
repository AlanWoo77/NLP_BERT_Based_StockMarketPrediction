from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os


class BiLSTM(object):
    def __init__(
        self,
        max_sequence_length,
        embedding_model,
        data_dir,
        model_dir,
        seed,
        embedding_dim,
        nepoch,
    ):
        self.max_sequence_length = max_sequence_length
        self.embedding_model = embedding_model
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.nepoch = nepoch

    def get_data(self):
        # read the data
        file_path = os.path.join(self.data_dir, "fpb.csv")
        df_fpb = pd.read_csv(file_path)[["content", "sentiment"]]
        df_fpb["content"] = self._tokenize(df_fpb["content"])

        # get the vectorisation done
        vectorised = []
        for tweet in df_fpb["content"]:
            embedded = self._vectorize(
                tweet, self.embedding_model, self.max_sequence_length
            )
            vectorised.append(embedded)
        df_fpb["content"] = vectorised
        # split the data
        X = np.array(df_fpb["content"].tolist())
        y = np.array(df_fpb["sentiment"])
        y = to_categorical(y)
        # split the train and test dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.40, random_state=self.seed, stratify=y
        )

        # split the test and validation dataset
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test,
            y_test,
            test_size=0.50,
            random_state=self.seed,
            stratify=y_test,
        )

        # pair them up
        train_dataset = (X_train, y_train)
        valid_dataset = (X_valid, y_valid)
        test_dataset = (X_test, y_test)
        return train_dataset, valid_dataset, test_dataset

    @staticmethod
    def _tokenize(column):
        tweet_tokenizer = TweetTokenizer()
        tokens = []
        for row in column:
            tokens.append(tweet_tokenizer.tokenize(row))
        return tokens

    @staticmethod
    def _vectorize(words, model, fixed_length):
        word_vectors = []
        for word in words:
            try:
                vec = model.vectors[model.key_to_index[word]]
                word_vectors.append(vec)
            except KeyError:
                continue

        # Truncate if too long
        if len(word_vectors) > fixed_length:
            word_vectors = word_vectors[:fixed_length]
        # Pad with zeros if too short
        elif len(word_vectors) < fixed_length:
            padding = [np.zeros(model.vector_size)] * (fixed_length - len(word_vectors))
            word_vectors.extend(padding)
        return word_vectors

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Bidirectional(
                LSTM(128),
                input_shape=(self.max_sequence_length, self.embedding_dim),
            )
        )
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(3, activation="softmax"))
        return

    def train(self):
        # first, get the data
        train_dataset, valid_dataset, test_dataset = self.get_data()
        X_train, y_train = train_dataset
        X_valid, y_valid = valid_dataset
        X_test, y_test = test_dataset
        self.build_model()
        optimizer = Adam(learning_rate=1e-4)
        loss = CategoricalCrossentropy()
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.nepoch,
            batch_size=32,
            validation_data=(X_valid, y_valid),
            validation_batch_size=32,
        )
        return self.history.history

    def evaluate(self):
        self.metrics = dict()
        # get the data to be evaluated
        _, _, test_dataset = self.get_data()
        X_test, y_test = test_dataset
        y_prediction = self.model.predict(X_test)
        # keep the raw one for Categorical Cross Entropy
        cce = CategoricalCrossentropy()
        self.metrics["loss"] = cce(y_test, y_prediction).numpy()
        # after cross entropy is computed, convert them to 1D array
        y_prediction = y_prediction.argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        self.metrics["accuracy"] = accuracy_score(y_test, y_prediction)
        self.metrics["cm"] = confusion_matrix(y_test, y_prediction)
        self.metrics["f1"] = f1_score(y_test, y_prediction, average="macro")

        return self.metrics
