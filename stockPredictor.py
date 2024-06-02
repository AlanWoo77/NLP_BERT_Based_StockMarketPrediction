from utils import preprocess, tokenize, vectorize, rescale
from keras.models import Sequential
from datetime import datetime
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from keras.optimizers.schedules import PolynomialDecay
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import Precision
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import os
import copy


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


class Config_Stock(object):
    """The configuration class for training."""

    def __init__(
        self,
        data_dir,
        model_dir,
        embedding_model,
        # for tokenization
        tickers=["AAPL", "AMZN", "TSLA", "MSFT", "GOOG"],
        max_seq_length=128,
        max_embedding_length=64,
        eval_batch_size=64,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.embedding_model = embedding_model
        self.tickers = tickers
        self.max_seq_length = max_seq_length
        self.max_embedding_length = max_embedding_length
        self.eval_batch_size = eval_batch_size


class StockPredictor(object):
    ## initialise the class ##
    def __init__(self, config, tweets, company_tweets, stocks):
        self.config = config
        self.tweets = tweets
        self.company_tweets = company_tweets
        self.stocks = stocks

    def predict_sentiment(self, engine=None, ticker=None):
        # get all the data needed first
        dataset = self.prepare_data_for_NLP(engine=engine, ticker=ticker)
        self.sentiment_dataset = copy.deepcopy(self.raw_dataset)
        # predict the sentiment using FinBERT model
        if engine == "finbert":
            path_FinBERT = os.path.join(self.config.model_dir, "FurtherTrain+FineTune")
            path_FinBERT_legacy = os.path.join(self.config.model_dir, "legacy")
            FinBERT = TFDistilBertForSequenceClassification.from_pretrained(
                path_FinBERT
            )
            FinBERT_legacy = TFDistilBertForSequenceClassification.from_pretrained(
                path_FinBERT_legacy
            )
            sentiment_logits_finBERT = FinBERT.predict(dataset).logits
            sentiment_logits_legacy = FinBERT_legacy.predict(dataset).logits
            sentiment_finBERT = tf.nn.softmax(sentiment_logits_finBERT, axis=-1)
            sentiment_legacy = tf.nn.softmax(sentiment_logits_legacy, axis=-1)

            self.sentiment_dataset["neutral"] = sentiment_finBERT[:, 0]
            self.sentiment_dataset["positive"] = sentiment_finBERT[:, 1]
            self.sentiment_dataset["negative"] = sentiment_finBERT[:, 2]
            self.sentiment_dataset["negative_legacy"] = sentiment_legacy[:, 0]
            self.sentiment_dataset["positive_legacy"] = sentiment_legacy[:, 1]
            return self.sentiment_dataset

        elif engine == "distilbert":
            path_DistilBERT = os.path.join(self.config.model_dir, "FineTune")
            DistilBERT = TFDistilBertForSequenceClassification.from_pretrained(
                path_DistilBERT
            )
            sentiment_logits_DistilBERT = DistilBERT.predict(dataset).logits
            sentiment_DistilBERT = tf.nn.softmax(sentiment_logits_DistilBERT, axis=-1)

            self.sentiment_dataset["neutral"] = sentiment_DistilBERT[:, 0]
            self.sentiment_dataset["positive"] = sentiment_DistilBERT[:, 1]
            self.sentiment_dataset["negative"] = sentiment_DistilBERT[:, 2]
            return self.sentiment_dataset

        elif engine == "lstm":
            path_BiLSTM = os.path.join(self.config.model_dir, "BiLSTM/model.h5")
            BiLSTM = tf.keras.models.load_model(path_BiLSTM)

            # predict the sentiment using the BiLSTM
            sentiment_BiLSTM = BiLSTM.predict(dataset)

            self.sentiment_dataset["neutral"] = sentiment_BiLSTM[:, 0]
            self.sentiment_dataset["positive"] = sentiment_BiLSTM[:, 1]
            self.sentiment_dataset["negative"] = sentiment_BiLSTM[:, 2]

            return self.sentiment_dataset

    def prepare_data_for_NLP(self, engine=None, ticker=None):
        ## the value of `model` has to be one of the list elements: ["finbert","distilbert",bilstm]
        ## `tweet_id` stands for the column of id, `post_date` stands for the date, `body` being the actual post

        ## two columns contained, one being `tweet_id`, corresponding to the same column in the tweets dataframe. Another one is `ticker_symbol`
        tweets_ticker = self.tweets.merge(
            self.company_tweets, on="tweet_id", how="left"
        )
        tweets_ticker.loc[
            tweets_ticker["ticker_symbol"] == "GOOGL", "ticker_symbol"
        ] = "GOOG"
        tweets_ticker = tweets_ticker.dropna(subset=["body"])
        # filter by the ticker
        tweets_ticker = tweets_ticker.loc[tweets_ticker.ticker_symbol == ticker]
        ################################################################################
        ################################################################################
        ## split the dataset according to different companies
        self.raw_dataset = tweets_ticker
        # for each ticker, create something simialr to store the tensor which can be used to train
        ## get the data prepared for finBERT and DistilBERT prediction
        # Get the text
        ############################################################################
        # first get the tensors for FinDistilBERT and DistilBERT
        X = tweets_ticker["body"].to_list()
        if engine == "distilbert" or engine == "finbert":
            # Tokenize
            encodings = tokenizer(
                X,
                truncation=True,
                padding=True,
                max_length=self.config.max_seq_length,
            )
            #
            dataset = tf.data.Dataset.from_tensor_slices(dict(encodings)).batch(
                self.config.eval_batch_size
            )
            self.dataset = dataset
            return self.dataset
            ###########################################################################
            ###########################################################################
            # second, get the embeddings for the BiLSTM Classifier
        elif engine == "lstm":
            X = tokenize(X)

            # get the vectorisation done
            vectorised = []
            for tweet in X:
                embedded = vectorize(
                    tweet,
                    self.config.embedding_model,
                    self.config.max_embedding_length,
                )
                vectorised.append(embedded)
            X = vectorised
            # split the data
            X = np.array(X)
            self.dataset = X
        ## now, store all the raw datasets
        return self.dataset

    def prepare_data_for_LSTM(self, engine=None, ticker=None, baseline=False):
        stock = self.stocks.loc[self.stocks.ticker_symbol == ticker]
        if not baseline:
            if engine == "finbert":
                ## get the tweet-sentiment dataset and merge them according to the date
                df_sentiment = (
                    self.predict_sentiment(engine=engine, ticker=ticker)[
                        [
                            "neutral",
                            "positive",
                            "negative",
                            "negative_legacy",
                            "positive_legacy",
                            "post_date",
                        ]
                    ]
                    .groupby(["post_date"])
                    .mean()
                )
            else:
                ## get the tweet-sentiment dataset and merge them according to the date
                df_sentiment = (
                    self.predict_sentiment(engine=engine, ticker=ticker)[
                        [
                            "neutral",
                            "positive",
                            "negative",
                            "post_date",
                        ]
                    ]
                    .groupby(["post_date"])
                    .mean()
                )
            df_sentiment.index = pd.to_datetime(df_sentiment.index)
            self.stock_sentiment = stock.merge(
                df_sentiment, left_on="day_date", right_index=True, how="left"
            )
        elif baseline:
            self.stock_sentiment = stock

        ## slice the dataset
        self.stock_sentiment = self.stock_sentiment.loc[
            (self.stock_sentiment["day_date"] >= datetime(2015, 1, 1))
            & (self.stock_sentiment["day_date"] <= datetime(2019, 12, 31))
        ].sort_values(by="day_date")
        ## add the features which can be used
        momentum_window = -10
        self.stock_sentiment["furtureReturn"] = -self.stock_sentiment[
            "close_value"
        ].pct_change(periods=momentum_window)
        # get the threshold for comparing
        buy_point = self.stock_sentiment["furtureReturn"].describe()["75%"]
        self.stock_sentiment["buySignal"] = (
            self.stock_sentiment["furtureReturn"] >= buy_point
        ).astype("int")

        n_timesteps = 30

        # split the train / validation / testing dataset
        train_dataset = self.stock_sentiment.loc[
            (self.stock_sentiment["day_date"] >= datetime(2015, 1, 1))
            & (self.stock_sentiment["day_date"] <= datetime(2017, 12, 31))
        ]
        train_dataset = train_dataset.drop(["ticker_symbol", "day_date"], axis=1)

        valid_dataset = self.stock_sentiment.loc[
            (self.stock_sentiment["day_date"] >= datetime(2017, 12, 1))
            & (self.stock_sentiment["day_date"] <= datetime(2018, 12, 31))
        ]
        valid_dataset = valid_dataset.drop(["ticker_symbol", "day_date"], axis=1)

        test_dataset = self.stock_sentiment.loc[
            (self.stock_sentiment["day_date"] >= datetime(2018, 12, 1))
            & (self.stock_sentiment["day_date"] <= datetime(2019, 12, 31))
        ]
        test_dataset = test_dataset.drop(["ticker_symbol", "day_date"], axis=1)

        train_features = []
        train_labels = []
        for i in range(n_timesteps, train_dataset.shape[0]):
            train_features.append(train_dataset.iloc[(i - n_timesteps) : i, :-1])
            train_labels.append(train_dataset.iloc[i, -1])

        train_features, train_labels = np.array(train_features), np.array(train_labels)

        valid_features = []
        valid_labels = []
        for i in range(n_timesteps, valid_dataset.shape[0]):
            valid_features.append(valid_dataset.iloc[(i - n_timesteps) : i, :-1])
            valid_labels.append(valid_dataset.iloc[i, -1])

        valid_features, valid_labels = np.array(valid_features), np.array(valid_labels)

        test_features = []
        test_labels = []
        for i in range(n_timesteps, test_dataset.shape[0]):
            test_features.append(test_dataset.iloc[(i - n_timesteps) : i, :-1])
            test_labels.append(test_dataset.iloc[i, -1])

        test_features, test_labels = np.array(test_features), np.array(test_labels)

        # do the re-scaling
        scaler = MinMaxScaler()
        train_features, valid_features, test_features = rescale(
            scaler=scaler,
            X_train=train_features,
            X_valid=valid_features,
            X_test=test_features,
        )

        return (
            (train_features, train_labels),
            (valid_features, valid_labels),
            (test_features, test_labels),
        )

    def build_model(self, sequence_length=30, n_features=10):
        model = Sequential()
        model.add(
            LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features))
        )
        model.add(Dropout(0.25))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(25, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        return model

    def train(
        self,
        engine=None,
        ticker=None,
        train_dataset=None,
        valid_dataset=None,
        lr_start=5e-4,
        lr_end=5e-5,
        training_epochs=15,
    ):
        X_train, y_train = train_dataset
        X_valid, y_valid = valid_dataset
        ##################################
        X_train = np.nan_to_num(X_train)
        X_valid = np.nan_to_num(X_valid)
        #################################
        self.model = self.build_model(
            sequence_length=X_train.shape[1], n_features=X_train.shape[2]
        )
        # compute the total steps
        num_train_steps = -(-X_train.shape[0] // training_epochs) * training_epochs
        lr_scheduler = PolynomialDecay(
            initial_learning_rate=lr_start,
            end_learning_rate=lr_end,
            decay_steps=num_train_steps,
        )
        # set the optimizer
        optimizer = Adam(learning_rate=lr_scheduler)
        loss = BinaryCrossentropy()
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy", Precision()],
        )
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=training_epochs,
            batch_size=32,
            validation_data=(X_valid, y_valid),
            validation_batch_size=32,
        )
        return self.history.history

    def evaluate(self, engine=None, ticker=None, test_dataset=None):
        X_test, y_test = test_dataset
        self.metrics = dict()
        # get the data to be evaluated
        ## convert all the NaN to 0s
        X_test = np.nan_to_num(X_test)
        y_prediction = self.model.predict(X_test)
        # keep the raw one for Categorical Cross Entropy
        bce = BinaryCrossentropy()
        self.metrics["loss"] = bce(y_test, y_prediction).numpy()
        # after cross entropy is computed, convert them to 1D array
        y_prediction = (y_prediction >= 0.5).astype("int")
        self.metrics["accuracy"] = accuracy_score(y_test, y_prediction)
        self.metrics["precision"] = precision_score(y_test, y_prediction)
        self.metrics["cm"] = confusion_matrix(y_test, y_prediction)
        self.metrics["f1"] = f1_score(y_test, y_prediction)

        return self.metrics, y_prediction
