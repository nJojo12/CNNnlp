


import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dot, Reshape
from tensorflow.keras.optimizers import Adam
import re
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add these imports at the top
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ------------------------
# Preprocessing
# ------------------------


import re

def preprocess(text):
    words = text.lower()
    new_book =""

    for char in words:
        ascii_val = ord(char)
        if ((48 <= ascii_val <= 57) or (97 <= ascii_val <= 122)) or char == " ":
            new_book+= char


    words = new_book.split(" ")
    words= [w for w in words if w != '']
    
    return words


def create_vocabulary(words):
    unique_words = list(OrderedDict.fromkeys(words))
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return unique_words, word_to_idx, idx_to_word

def generate_training_data(words, word_to_idx, window_size=2):
    target_words = []
    context_words = []
    
    for i, target_word in enumerate(words):
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if j != i:
                target_words.append(word_to_idx[target_word])
                context_words.append(word_to_idx[words[j]])
    
    return np.array(target_words), np.array(context_words)





def create_cnn_model(max_sequence_length, vocab_size, num_books=7):
    # Input layer
    inputs = Input(shape=(max_sequence_length,))
    
    # Embedding layer
    embedding = Embedding(input_dim=vocab_size + 1,  # +1 for padding
                        output_dim=100,
                        input_length=max_sequence_length)(inputs)
    
    # CNN layers
    conv1 = Conv1D(128, 5, activation='relu')(embedding)
    pool1 = GlobalMaxPooling1D()(conv1)
    
    # Dense layers
    dense = Dense(64, activation='relu')(pool1)
    outputs = Dense(num_books, activation='softmax')(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def prepare_training_data(processed_books, vocabularies, max_sequence_length=50):
    X = []
    y = []
    
    # Create a master vocabulary
    master_vocab = set()
    for book_vocab in vocabularies.values():
        master_vocab.update(book_vocab[2])  # book_vocab[2] contains unique_words
    
    word_to_idx = {word: idx + 1 for idx, word in enumerate(master_vocab)}  # +1 for padding

    print(word_to_idx)
    
    # Convert text to sequences
    for book_num, words in processed_books.items():
        sequence = [word_to_idx[word] for word in words if word in word_to_idx]
        X.append(sequence)
        y.append(int(book_num[2]) - 1)  # Convert HP1 to 0, HP2 to 1, etc.

    
    # Pad sequences
    X = pad_sequences(X, maxlen=max_sequence_length, padding='post')
    y = to_categorical(y)

    print(X)
    print(y)
    
    return X, y, word_to_idx



def main():
    # Load text
    # Dictionary to store processed lines from each book
    processed_books = {}
    vocabularies = {}
    
    # Process one line from each book
    for book_num in range(1, 8):  # HP books 1-7
        filename = f'HP{book_num}.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            # Read first non-empty line
            line = f.readline()
            processed_words = preprocess(line)
            processed_books[f'HP{book_num}'] = processed_words

            unique_words, word_to_idx, idx_to_word = create_vocabulary(processed_words)

            vocabularies[f'HP{book_num}'] = [word_to_idx, idx_to_word, unique_words]
            print(vocabularies[f'HP{book_num}'])

    max_sequence_length = 10
    X, y, master_word_to_idx = prepare_training_data(processed_books, vocabularies, max_sequence_length)
    
    # Create and train the model
    model = create_cnn_model(max_sequence_length, len(master_word_to_idx))
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    history = model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       epochs=10,
                       batch_size=32)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    return model, master_word_to_idx, history

    

if __name__ == "__main__":
    main()
