

import nltk #Importing NLP library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

messages = pd.read_csv("./data/train.csv",nrows=60000)