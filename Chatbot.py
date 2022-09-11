#importing  packages
import numpy
import pycaret
import random
import json
import pickle
import nltk
import tensorflow as tf


#Loading data
from nltk.stem.lancaster import LancasterStemmer
word_stemmer = LancasterStemmer()
with open("training.json") as data:
    contents = json.load(data)


