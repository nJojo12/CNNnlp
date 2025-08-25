# import numpy as np
# from collections import OrderedDict
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Dense, Dot, Reshape
# from tensorflow.keras.optimizers import Adam
# import re
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt


# def preprocess(text):
#     words = re.findall(r'\w+', text.lower())
#     return words

# def create_vocabulary(words):
#     unique_words = list(OrderedDict.fromkeys(words))
#     word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
#     idx_to_word = {idx: word for word, idx in word_to_idx.items()}
#     return unique_words, word_to_idx, idx_to_word

# def generate_training_data(words, word_to_idx, window_size=2):
#     target_words = []
#     context_words = []
    
#     for i, target_word in enumerate(words):
#         start = max(0, i - window_size)
#         end = min(len(words), i + window_size + 1)
        
#         for j in range(start, end):
#             if j != i:
#                 target_words.append(word_to_idx[target_word])
#                 context_words.append(word_to_idx[words[j]])
    
#     return np.array(target_words), np.array(context_words)

# def build_word2vec_model(vocab_size, embedding_dim=100):
#     input_target = Input(shape=(1,))
#     input_context = Input(shape=(1,))
    
#     embedding = Embedding(vocab_size, embedding_dim, name='embedding')
    
#     target_embedding = embedding(input_target)
#     context_embedding = embedding(input_context)
    
#     dot_product = Dot(axes=2)([target_embedding, context_embedding])
#     dot_product = Reshape((1,))(dot_product)
    
#     output = Dense(1, activation='sigmoid')(dot_product)
    
#     model = Model(inputs=[input_target, input_context], outputs=output)
#     model.compile(loss='binary_crossentropy', optimizer=Adam(0.001))
    
#     return model

# def find_similar_words(embedding_weights, word_to_idx, idx_to_word, query_word, top_n=5):
#     if query_word not in word_to_idx:
#         print(f"'{query_word}' not in vocabulary")
#         return
    
#     query_embedding = embedding_weights[word_to_idx[query_word]]
    
#     similarities = []
#     for word, idx in word_to_idx.items():
#         if word == query_word:
#             continue
#         sim = cosine_similarity([query_embedding], [embedding_weights[idx]])[0][0]
#         similarities.append((word, sim))
    
#     similarities.sort(key=lambda x: x[1], reverse=True)
    
#     print(f"Words similar to '{query_word}':")
#     for word, sim in similarities[:top_n]:
#         print(f"{word}: {sim:.3f}")


# def main():
#     # Step 1: Read text file
#     with open('HP1.txt', 'r', encoding='utf-8') as f:
#         text = f.read(50000)
    
#     # Step 2: Preprocess text
#     words = preprocess(text)
    
#     # Step 3: Create vocabulary
#     unique_words, word_to_idx, idx_to_word = create_vocabulary(words)
#     vocab_size = len(unique_words)
#     print(f"Vocabulary size: {vocab_size}")
    
#     # Generate training data
#     target_words, context_words = generate_training_data(words, word_to_idx)
#     print(f"Generated {len(target_words)} training pairs")
    
#     if len(target_words) == 0:
#         print("Not enough training data generated")
#         return
    
#     # Create negative samples
#     np.random.seed(42)
#     negative_samples = np.random.randint(0, vocab_size, len(target_words))
    
#     # Combine positive and negative samples
#     X_target = np.concatenate([target_words, target_words])
#     X_context = np.concatenate([context_words, negative_samples])
#     y = np.concatenate([np.ones(len(target_words)), np.zeros(len(target_words))])
    
#     # Build and train model
#     model = build_word2vec_model(vocab_size, embedding_dim=100)
    
#     model.fit(
#         x=[X_target, X_context],
#         y=y,
#         batch_size=128,
#         epochs=10,
#         verbose=1
#     )
    
#     # Get embeddings
#     embedding_layer = model.get_layer('embedding')
#     weights = embedding_layer.get_weights()[0]
    
#     # Example embedding
#     example_word = "harry"
#     if example_word in word_to_idx:
#         embedding = weights[word_to_idx[example_word]]
#         print(f"Embedding for '{example_word}': {embedding[:10]}...")
#     else:
#         print(f"Word '{example_word}' not in vocabulary")
    
#     # Find similar words
#     find_similar_words(weights, word_to_idx, idx_to_word, "harry")
#     find_similar_words(weights, word_to_idx, idx_to_word, "dursley")
#     find_similar_words(weights, word_to_idx, idx_to_word, "vernon")
    

# if __name__ == "__main__":
#     main()


import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dot, Reshape
from tensorflow.keras.optimizers import Adam
import re
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------------
# Preprocessing
# ------------------------
def preprocess(text):
    words = re.findall(r'\w+', text.lower())
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

# ------------------------
# Model
# ------------------------
def build_word2vec_model(vocab_size, embedding_dim=100):
    input_target = Input(shape=(1,))
    input_context = Input(shape=(1,))
    
    embedding = Embedding(vocab_size, embedding_dim, name='embedding')
    
    target_embedding = embedding(input_target)
    context_embedding = embedding(input_context)
    
    dot_product = Dot(axes=2)([target_embedding, context_embedding])
    dot_product = Reshape((1,))(dot_product)
    
    output = Dense(1, activation='sigmoid')(dot_product)
    
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001))
    
    return model

# ------------------------
# Similarity Experiments
# ------------------------
def find_similar_words_metrics(embedding_weights, word_to_idx, query_word, top_n=5):
    if query_word not in word_to_idx:
        print(f"'{query_word}' not in vocabulary")
        return
    
    query_vec = embedding_weights[word_to_idx[query_word]].reshape(1, -1)
    all_vecs = embedding_weights

    metrics = {
        "Cosine Similarity": cosine_similarity(query_vec, all_vecs)[0],
        "Negative Euclidean Distance": -euclidean_distances(query_vec, all_vecs)[0],
        "Negative Manhattan Distance": -manhattan_distances(query_vec, all_vecs)[0],
        "Dot Product": np.dot(all_vecs, query_vec.T).flatten()
    }

    print(f"\n=== Similar words to '{query_word}' under different metrics ===")
    for metric_name, scores in metrics.items():
        sorted_indices = np.argsort(scores)[::-1]
        sorted_words = [(w, scores[word_to_idx[w]]) for w in [list(word_to_idx.keys())[i] for i in sorted_indices if i != word_to_idx[query_word]]]
        print(f"\n{metric_name}:")
        for word, score in sorted_words[:top_n]:
            print(f"  {word}: {score:.4f}")

# ------------------------
# Visualization Function
# ------------------------
def plot_metric_neighbors(embedding_weights, word_to_idx, query_word, top_n=5):
    if query_word not in word_to_idx:
        print(f"'{query_word}' not in vocabulary")
        return
    
    query_vec = embedding_weights[word_to_idx[query_word]].reshape(1, -1)
    all_vecs = embedding_weights

    metrics = {
        "Cosine Similarity": cosine_similarity(query_vec, all_vecs)[0],
        "Negative Euclidean Distance": -euclidean_distances(query_vec, all_vecs)[0],
        "Negative Manhattan Distance": -manhattan_distances(query_vec, all_vecs)[0],
        "Dot Product": np.dot(all_vecs, query_vec.T).flatten()
    }

    # Fit PCA once on the whole vocab
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_vecs)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Top {top_n} neighbors of '{query_word}' by different metrics", fontsize=16)

    for ax, (metric_name, scores) in zip(axes, metrics.items()):
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = [i for i in sorted_indices if i != word_to_idx[query_word]][:top_n]

        # Plot all words lightly in gray
        ax.scatter(reduced[:, 0], reduced[:, 1], c='lightgray', alpha=0.3, s=10)

        # Plot query word in red
        qx, qy = reduced[word_to_idx[query_word]]
        ax.scatter(qx, qy, c='red', s=80, label=query_word)

        # Plot neighbors in blue + arrows
        for idx in top_indices:
            nx, ny = reduced[idx]
            ax.scatter(nx, ny, c='blue', s=50)
            ax.text(nx + 0.01, ny + 0.01, list(word_to_idx.keys())[idx], fontsize=8)
            ax.arrow(qx, qy, nx - qx, ny - qy, color='blue', alpha=0.5, head_width=0.02, length_includes_head=True)

        ax.set_title(metric_name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ------------------------
# Main
# ------------------------
def main():
    # Load text
    with open('HP1.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    words = preprocess(text)
    unique_words, word_to_idx, idx_to_word = create_vocabulary(words)
    vocab_size = len(unique_words)
    print(f"Vocabulary size: {vocab_size}")
    
    target_words, context_words = generate_training_data(words, word_to_idx)
    print(f"Generated {len(target_words)} training pairs")
    if len(target_words) == 0:
        print("Not enough training data generated")
        return
    
    np.random.seed(42)
    negative_samples = np.random.randint(0, vocab_size, len(target_words))
    
    X_target = np.concatenate([target_words, target_words])
    X_context = np.concatenate([context_words, negative_samples])
    y = np.concatenate([np.ones(len(target_words)), np.zeros(len(target_words))])
    
    model = build_word2vec_model(vocab_size, embedding_dim=100)
    model.fit([X_target, X_context], y, batch_size=128, epochs=10, verbose=1)
    
    embedding_layer = model.get_layer('embedding')
    weights = embedding_layer.get_weights()[0]
    
    # Run experiments + plots
    test_words = ["harry", "dursley", "vernon","“voldemort","wizard","“hermione","“ron","“snape","“malfoy","“hagrid"]
    for word in test_words:
        find_similar_words_metrics(weights, word_to_idx, word, top_n=5)
        plot_metric_neighbors(weights, word_to_idx, word, top_n=5)

if __name__ == "__main__":
    main()
