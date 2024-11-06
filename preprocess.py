import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# 1. Load and preprocess data
def load_and_preprocess_data(filepath, num_samples=1000):
    data = pd.read_csv(filepath).head(num_samples)
    questions = data['Question'].values
    answers = data['Answer'].values
    questions = ['<start> ' + q for q in questions]
    answers = ['<start> ' + a + ' <end>' for a in answers]
    return questions, answers
