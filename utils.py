import emoji
import re
import numpy as np
from nltk.tokenize import TweetTokenizer


def lowercase_text(text):
    return text.lower()


def remove_dollar_sign_words(text):
    # This regular expression matches words that start with $ followed by letters
    pattern = r"\$\w+(\.\w+)?"
    # Replace the pattern with an empty string
    return re.sub(pattern, "", text)


def remove_urls_and_emails(text):
    # Regex pattern to match URLs
    url_pattern = r"https?://\S+|www\.\S+"
    # Regex pattern to match email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    # Remove URLs
    text = re.sub(url_pattern, "", text)
    # Remove email addresses
    text = re.sub(email_pattern, "", text)
    return text


def remove_newlines(text):
    return text.replace("\n", "")


def translate_emojis_to_text(text):
    return emoji.demojize(text)


def remove_extra_symbols(text):
    pattern = r"[:#_;&@]"
    text = re.sub(pattern, " ", text)
    return text


def remove_extra_space(text):
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess(text):
    text = lowercase_text(text)
    text = remove_dollar_sign_words(text)
    text = remove_urls_and_emails(text)
    text = remove_newlines(text)
    text = translate_emojis_to_text(text)
    text = remove_extra_symbols(text)
    text = remove_extra_space(text)
    return text


def unfreezer(epoch, step, model):
    if epoch == 0 & step == 0:
        model.layers[-2].trainable = True
    elif epoch == 0 & step == int(len(train_dataset) / 2):
        model.layers[-2].trainable = True
    elif epoch == 1 & step == 0:
        model.distilbert.transformer.layer[-1].trainable = True
    elif epoch == 2 & step == 0:
        model.distilbert.transformer.layer[-2].trainable = True
    elif epoch == 3 & step == 0:
        model.distilbert.transformer.layer[-3].trainable = True
    elif epoch == 4 & step == 0:
        model.distilbert.transformer.layer[-4].trainable = True
    elif epoch == 5 & step == 0:
        model.distilbert.transformer.layer[-5].trainable = True
    return None


def tokenize(column):
    tweet_tokenizer = TweetTokenizer()
    tokens = []
    for row in column:
        tokens.append(tweet_tokenizer.tokenize(row))
    return tokens


def vectorize(words, model, fixed_length):
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


# define the rescale function for preparing the input for LSTM
def rescale(scaler, X_train, X_valid, X_test):
    # Reshape the data to 2D for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_valid_2d = X_valid.reshape(-1, X_valid.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])

    # Fit the scaler on the training data and transform all datasets
    scaler.fit(X_train_2d)
    X_train_scaled_2d = scaler.transform(X_train_2d)
    X_valid_scaled_2d = scaler.transform(X_valid_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # Reshape the scaled data back to its original 3D shape
    X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
    X_valid_scaled = X_valid_scaled_2d.reshape(X_valid.shape)
    X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

    return X_train_scaled, X_valid_scaled, X_test_scaled
