import os
import pickle
import re
from bs4 import BeautifulSoup

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sklearn
# import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords
import json

MAX_SEQUENCE_LENGTH = 300


def run_detect(text):
    processed_text = denoise_text(text)
    tokenizer = load_tokenizer()
    txt_padded = pad_sequences(tokenizer.texts_to_sequences(processed_text), maxlen=MAX_SEQUENCE_LENGTH)

    return model_predict(txt_padded)


def model_predict(txt_padded):
    model = load_model()
    y_pred = model.predict(txt_padded)[0][0]
    return "Fake News" if y_pred >= 0.5 else "Real News"


def load_model():
    path_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(path_dir, "model.h5")

    model = tf.keras.models.load_model(filename)

    return model


def load_stop_json():
    with open('stop_words.json', 'r') as f:
        json_object = json.load(f)

    return json_object['stop']


def load_tokenizer():
    path_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(path_dir, "tokenizer.pickle")
    print("filename: ", filename)

    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def strip_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing URL's
def remove_url(text):
    return re.sub(r'http\S+', '', text)


# Removing the stopwords from text
def remove_stopwords(text):
    stop = load_stop_json()
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_url(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)

    text1 = [text]
    return text1
