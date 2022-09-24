
#LIBRARIES
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#DATA IMPORT
messages = pd.read_csv("C:/Users/nivedhan/Downloads/Natural-Language-Processing-main/Natural-Language-Processing-main/News Classification/data/train.csv",nrows=60000)
messages.head(5)