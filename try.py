from transformers import DistilBertForMaskedLM
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import torch
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import emoji
import re
from finDistilBert import *
from biLSTM import BiLSTM
from stockPredictor import *
import gensim.downloader

current_path = os.getcwd()
data_path = os.path.join(current_path, "Data")
model_path = os.path.join(current_path, "models")
log_path = os.path.join(current_path, "trainingLog")
distilbert_mask = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

config = Config(
    data_dir=data_path,
    distilbert_mask=distilbert_mask,
    model_dir=model_path,
    log_dir=log_path,
)

embeddings = gensim.downloader.load("word2vec-google-news-300")

# classifier = BiLSTM(
#     max_sequence_length=64,
#     embedding_model=embeddings,
#     data_dir=data_path,
#     seed=42,
#     embedding_dim=300,
#     nepoch=6,
# )

config_predictor = Config_Stock(
    data_dir=data_path,
    model_dir=model_path,
    embedding_model=embeddings,
)

## read the data and wash for a bit
path_tweets = os.path.join(config_predictor.data_dir, "Tweet.csv")
path_companyTweets = os.path.join(config_predictor.data_dir, "Company_Tweet.csv")
tweets = pd.read_csv(path_tweets)[:100]
tweets["post_date"] = pd.to_datetime(tweets["post_date"], unit="s")
tweets["post_date"] = tweets["post_date"].dt.strftime("%Y-%m-%d")
company_tweets = pd.read_csv(path_companyTweets)


path_stock = os.path.join(config_predictor.data_dir, "CompanyValues.csv")
stocks = pd.read_csv(path_stock)
stocks["day_date"] = pd.to_datetime(stocks["day_date"], format="%Y-%m-%d")
stocks.loc[stocks["ticker_symbol"] == "GOOGL", "ticker_symbol"] = "GOOG"


predictor = StockPredictor(
    config=config_predictor, tweets=tweets, company_tweets=company_tweets, stocks=stocks
)

# get the data
train_dataset, valid_dataset, test_dataset = predictor.prepare_data_for_LSTM(
    engine="finbert", ticker="AAPL"
)

# train
history = predictor.train(
    engine="finbert",
    ticker="AAPL",
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    lr_start=5e-4,
    lr_end=5e-5,
    training_epochs=15,
)

metrics_finbert_AAPL, predictions_finbert_AAPL = predictor.evaluate(
    engine="finbert", ticker="AAPL", test_dataset=test_dataset
)

print(metrics_finbert_AAPL)
