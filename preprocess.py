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

# 2. Tokenization
def tokenize_and_pad_sequences(questions, answers, max_seq_length=150):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(questions + answers)
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

    # Convert texts to sequences
    question_sequences = tokenizer.texts_to_sequences(questions)
    answer_sequences = tokenizer.texts_to_sequences(answers)
    
    # Pad sequences
    question_sequences = pad_sequences(question_sequences, maxlen=max_seq_length, padding='post')
    answer_sequences = pad_sequences(answer_sequences, maxlen=max_seq_length, padding='post')

    return question_sequences, answer_sequences, tokenizer, vocab_size

# Visualization : Sequence Length Distribution
def plot_sequence_length_distribution(sequences, title="Sequence Length Distribution"):
    sequence_lengths = [len(seq) for seq in sequences]
 
    plt.figure(figsize=(12, 6))
    sns.histplot(sequence_lengths, bins=30, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.show()

# 4. Plotting functions
def plot_word_frequency(tokenizer, top_n=30):
    word_counts = tokenizer.word_counts
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, frequencies = zip(*sorted_word_counts)
 
