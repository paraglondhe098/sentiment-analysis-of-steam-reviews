import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords

maxlen =50
estopwords=stopwords.words('english')
estopwords.remove('not')
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)
def preprocess_text(sen, stopwords=estopwords, singleremover=True):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''
    sentence = sen.lower()
    # Remove html tags
    sentence = remove_tags(sentence)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    if singleremover:
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    sentence = re.sub(r'\s+', ' ',sentence)
    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence
def preprocess_and_tokenize(unprocessed_texts):
    '''
    :param unprocessed_texts: List of Reviews(String)
    :return: Tokenized sequences with padding upto length 50
    '''
    processed = []
    for review in unprocessed_texts:
        processed.append(preprocess_text(review))
    file = open("tokenizer.pkl", 'rb')
    # tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer = pickle.load(file)
    file.close()
    tokenized = tokenizer.texts_to_sequences(processed)
    tokenized_padded= tf.keras.preprocessing.sequence.pad_sequences(tokenized, padding='post', maxlen=maxlen)
    return tokenized_padded

if __name__ == '__main__':
    # model = tf.keras.Sequential()
    model = tf.keras.models.load_model('Models/c1_lstm_model_acc_highaccr0.827.keras')
    #Add Reviews in input_reviews.csv file
    input_reviews = pd.read_csv('input_reviews.csv')

    list_inputs = input_reviews['Reviews']
    preprocessed_inputs = preprocess_and_tokenize(list_inputs)
    prediction = np.round(model.predict(preprocessed_inputs),2)
    print(prediction)