#1-importing  packages
import numpy
import pycaret
import random
import json
import pickle
import nltk
import tensorflow as tf


#2-Loading data
from nltk.stem.lancaster import LancasterStemmer
word_stemmer = LancasterStemmer()
with open("training.json") as data:
    contents = json.load(data)

#3-Preprocessing
try:
    # load pre-processed data from the pickle file.
    with open("cache.pickle", "rb") as file:
        vocab, label, train, test = pickle.load(file)

except:

    
    vocab = []
    label = []
    x = []
    y = []

    # Extract words from the patterns, extract labels (tags)

    for i in contents["intents"]:
        for j in i["patterns"]:
        #Tokenization
            word_tokenized = nltk.word_tokenize(j) 
            vocab.extend(word_tokenized) 
            x.append(word_tokenized)
            y.append(i["tag"])

        if i["tag"] not in label:
            label.append(i["tag"])

    #Stemming
    vocab = [word_stemmer.stem(w.lower()) for w in vocab if w != "?"]
    vocab = sorted(list(set(vocab)))
    label = sorted(label)

    #BAG OF WORDS- One hot encoding
    train = []
    test = []

    output = [0 for _ in range(len(label))]

    for k, doc in enumerate(x):
        bag_words = []

        wrds = [word_stemmer.stem(w) for w in doc]

        for w in vocab:
            if w in wrds:
                bag_words.append(1)
            else:
                bag_words.append(0)

        line = output[:]
        line[label.index(y[k])] = 1

        train.append(bag_words)
        test.append(line)


    train= numpy.array(train)
    test = numpy.array(test)

    # Write the pre-processed data into the pickle file.
    with open("cache.pickle", "wb") as m:
        pickle.dump((vocab, label, train, test), m)

#4-BAG OF WORDS
def bag_of_words(question, vocab):
    bagofwords = [0 for _ in range(len(vocab))]

    word = nltk.word_tokenize(question)
    word = [word_stemmer.stem(j.lower()) for j in word]

    for v in word:
        for i, w in enumerate(vocab):
            if v == w:
                bagofwords[i] = 1

    return numpy.array(bagofwords)


#5-CONVOLUTIONAL NEURAL NETWORK
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
cnn_model = Sequential()
cnn_model.add(Embedding(507, 32, input_length=507))
cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size = 3))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(220,activation='softmax'))
cnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


