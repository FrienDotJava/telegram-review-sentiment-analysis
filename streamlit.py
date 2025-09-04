import numpy as np
import streamlit as st
import json
import pickle
import re
import string

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------------------------------------------------------

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text

def casefoldingText(text):
    result = text.lower()
    return result

slangwords = {"gw": "aku", "abis": "habis", "udh": "sudah", 
              "masi": "masih", "udah": "sudah", "aja": "saja", 
              "bgt": "banget", "maks": "maksimal", "trus":"terus", 
              "bgus":"bagus", 'yg':'yang', 'apk':'aplikasi', 'tele':'telegram',
              "gk":"tidak", "gabisa":"tidak bisa", "nomer":"nomor","kalo":"kalau",
              "gua":"aku"
              }
def fix_slangwords(text):
    words = text.split()
    fixed_words = []

    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)

    fixed_text = ' '.join(fixed_words)
    return fixed_text

def tokenizingText(text):
    result = word_tokenize(text)
    return result

NEGATORS = {"tidak","tak","bukan","enggak","gak","ga","nggak","kurang","ndak","ndak","ora","ngga","tdk","gk"}

def filteringText(text_token):
    listStopwords = set(stopwords.words('indonesian')) | set(stopwords.words('english'))
    listStopwords.update(["sih", "yah"])
    listStopwords = listStopwords - NEGATORS

    return [t for t in text_token if t not in listStopwords]

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    words = text.split()

    stemmed_words = [stemmer.stem(word) for word in words]

    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

def toSentence(list_words): # Mengubah daftar kata menjadi kalimat
    sentence = ' '.join(word for word in list_words)
    return sentence

# ---------------------------------------------------------------------------------------------------------------------

# Load word_index
with open("word_index.json", "r", encoding="utf-8") as f:
    word_index = json.load(f)

# Load max_length and class_names
with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

max_length = meta["max_length"]
class_names = meta["class_names"]

RNN = load_model("rnn_model.h5")

def preprocess_to_tokens_inference(text: str):
    t = cleaningText(text)
    t = casefoldingText(t)
    t = fix_slangwords(t)
    tokens = tokenizingText(t)
    tokens = filteringText(tokens)
    return tokens

def text_to_padded_sequence_inference(text: str, word_index: dict, max_length: int = 200):
    tokens = preprocess_to_tokens_inference(text)
    seq = [word_index.get(token, 0) for token in tokens]
    X = pad_sequences([seq], maxlen=max_length, padding='post', truncating='post')
    return X, tokens

def predict_sentence(text: str, model, word_index: dict, max_length: int = 200, class_names=None):
    X_new, tokens = text_to_padded_sequence_inference(text, word_index, max_length)
    probs = model.predict(X_new, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = class_names[pred_idx] if class_names else pred_idx
    return {"text": text, "tokens": tokens, "pred_idx": pred_idx, "label": label, "probs": probs}

class_names = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------------------------------------------------

st.title('Sentimen Analysis Telegram Review in Indonesia')
review = st.text_input('What do you think about Telegram app? (in Indonesia)')

if st.button("Predict"):
    out = predict_sentence(review, RNN, word_index, max_length=200, class_names=class_names)
    st.success(f"\nText: {out['text']}")
    st.success(f"Tokens: {out['tokens']}")
    st.success(f"Predicted: {out['label']}")
    st.success(f"Probabilities: {out['probs']}")