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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, Dropout, LSTM, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch

# ------------------------
# Preprocessing
# ------------------------

def preprocess(text):
    """Improved preprocessing function"""
    words = text.lower()
    new_book = ""

    for char in words:
        ascii_val = ord(char)
        if ((48 <= ascii_val <= 57) or (97 <= ascii_val <= 122)) or char == " ":
            new_book += char

    words = new_book.split(" ")
    words = [w for w in words if w != '' and len(w) > 1]  # Filter out single characters
    
    return words

def create_vocabulary(words):
    unique_words = list(OrderedDict.fromkeys(words))
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return unique_words, word_to_idx, idx_to_word

def load_pretrained_word2vec(model_path=''):
    """Load pre-trained Word2Vec model from Lab 1"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        word_to_idx = checkpoint['word_to_idx']
        idx_to_word = checkpoint['idx_to_word']
        
        # Extract embeddings from the model state dict
        model_state = checkpoint['model_state_dict']
        embeddings = model_state['center_embeddings.weight'].numpy()
        
        print(f"Loaded pre-trained embeddings: {embeddings.shape}")
        return embeddings, word_to_idx, idx_to_word
    except FileNotFoundError:
        print("Pre-trained Word2Vec model not found. Training new embeddings...")
        return None, None, None

def create_improved_cnn_model(max_sequence_length, vocab_size, embedding_matrix=None, 
                             embedding_dim=100, num_books=7, trainable_embeddings=False,filter_sizes = [2,3,4]):
    """Improved CNN model with pre-trained embeddings"""
    
    # Input layer
    inputs = Input(shape=(max_sequence_length,))
    
    # Embedding layer - use pre-trained if available

    if embedding_matrix is not None:
        embedding = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=max_sequence_length,
                            weights=[embedding_matrix],
                            trainable=trainable_embeddings)(inputs)
        print("Using pre-trained embeddings")
    else:
        embedding = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=max_sequence_length,
                            trainable=True)(inputs)
        print("Training embeddings from scratch")
    
    x = Dropout(0.2)(embedding)
    
    # Multiple CNN layers with different filter sizes
    conv_layers = []
    
    for filter_size in filter_sizes:
        conv = Conv1D(128, filter_size, activation='relu', padding='same')(x)
        conv = GlobalMaxPooling1D()(conv)
        conv_layers.append(conv)
    
    print(conv_layers)
    # Concatenate all conv layers
    if len(conv_layers) > 1:
        from tensorflow.keras.layers import concatenate
        merged = concatenate(conv_layers)
    else:
        merged = conv_layers[0]
    
    # Add LSTM for sequential patterns
    # lstm = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)
    
    # Concatenate CNN and LSTM features
    from tensorflow.keras.layers import concatenate
    combined = concatenate([merged])
    
    # Dense layers with dropout and regularization
    dense1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(combined)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(dense1)
    dense2 = Dropout(0.3)(dense2)
    
    outputs = Dense(num_books, activation='softmax')(dense2)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a more sophisticated optimizer
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
    
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def prepare_improved_training_data(processed_books, pretrained_word_to_idx=None, max_sequence_length=200):
    """Improved training data preparation with better text chunking and fixed indexing"""
    X = []
    y = []
    
    # Use pre-trained vocabulary if available, otherwise create new one
    if pretrained_word_to_idx is not None:
        # Create mapping: reserve index 0 for padding, map words to indices 1, 2, 3, ...
        master_word_to_idx = {}
        master_word_to_idx['<PAD>'] = 0  # Explicit padding token
        
        for word, original_idx in pretrained_word_to_idx.items():
            # Map to new indices starting from 1
            master_word_to_idx[word] = original_idx + 1
            
        vocab_size = len(pretrained_word_to_idx) + 1  # +1 for padding token
        print(f"Using pre-trained vocabulary: {len(pretrained_word_to_idx)} words + padding = {vocab_size} total")
        print(f"Index range: 0 (padding) to {vocab_size - 1}")
    else:
        # Create master vocabulary from all books
        master_vocab = set()
        for book_vocab in processed_books.values():
            for words_list in book_vocab:
                master_vocab.update(words_list)
        
        # Create mapping: reserve index 0 for padding
        master_word_to_idx = {}
        master_word_to_idx['<PAD>'] = 0
        
        for idx, word in enumerate(sorted(master_vocab), 1):  # Start from 1
            master_word_to_idx[word] = idx
            
        vocab_size = len(master_vocab) + 1  # +1 for padding token
        print(f"Created new vocabulary: {len(master_vocab)} words + padding = {vocab_size} total")
        print(f"Index range: 0 (padding) to {vocab_size - 1}")

    # Convert text to sequences with improved chunking
    for book_num, list_of_word_lists in processed_books.items():
        book_label = int(book_num[2]) - 1  # Convert HP1 to 0, HP2 to 1, etc.
        
        # Process each page/line
        for words in list_of_word_lists:
            if len(words) < 10:  # Skip very short sequences
                continue
                
            # Create overlapping windows for longer sequences
            if len(words) > max_sequence_length:
                # Create multiple samples with sliding window
                step_size = max_sequence_length // 2
                for i in range(0, len(words) - max_sequence_length + 1, step_size):
                    window_words = words[i:i + max_sequence_length]
                    sequence = [master_word_to_idx.get(word, 0) for word in window_words]
                    X.append(sequence)
                    y.append(book_label)
            else:
                # Use the full sequence
                sequence = [master_word_to_idx.get(word, 0) for word in words]
                if len(sequence) > 5:  # Only add sequences with reasonable length
                    X.append(sequence)
                    y.append(book_label)

    # Pad sequences
    X = pad_sequences(X, maxlen=max_sequence_length, padding='post', truncating='post', value=0)
    y = to_categorical(y, num_classes=7)

    print(f"Created {len(X)} training samples")
    print(f"Sequence shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Vocabulary size for embedding layer: {vocab_size}")
    
    # Verify indices are within bounds
    max_index = np.max(X) if len(X) > 0 else 0
    print(f"Max index in sequences: {max_index}")
    print(f"Min index in sequences: {np.min(X) if len(X) > 0 else 0}")
    
    if max_index >= vocab_size:
        print(f"ERROR: Max index {max_index} is >= vocab_size {vocab_size}")
        print("This will cause the embedding error!")
        # Find problematic indices
        problem_indices = np.where(X >= vocab_size)
        if len(problem_indices[0]) > 0:
            print(f"Found {len(problem_indices[0])} problematic indices")
            # Set them to 0 (padding)
            X[problem_indices] = 0
            print("Fixed by setting problematic indices to 0")
    else:
        print("All indices are within bounds - should work correctly!")
    
    return X, y, master_word_to_idx, vocab_size

def create_embedding_matrix(pretrained_embeddings, pretrained_word_to_idx, new_word_to_idx, embedding_dim, vocab_size):
    """Create embedding matrix mapping from new vocabulary to pre-trained embeddings"""
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    found_words = 0
    
    for word, new_idx in new_word_to_idx.items():
        if new_idx == 0:  # Skip padding token
            continue
            
        original_word = word if isinstance(word, str) else str(word)
        
        if original_word in pretrained_word_to_idx:
            pretrained_idx = pretrained_word_to_idx[original_word]
            if pretrained_idx < len(pretrained_embeddings):
                embedding_matrix[new_idx] = pretrained_embeddings[pretrained_idx]
                found_words += 1
        else:
            # Initialize unknown words randomly
            embedding_matrix[new_idx] = np.random.normal(0, 0.1, embedding_dim)
    
    print(f"Found pre-trained embeddings for {found_words}/{len(new_word_to_idx)} words")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix
import time
import datetime

# Add this import at the top with your other imports
# ...existing imports...

def format_time(seconds):
    """Convert seconds to readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

def main():
    print("Improved CNN Text Classifier for Harry Potter Books")
    print("=" * 60)
    
    # Try to load pre-trained Word2Vec embeddings
    pretrained_embeddings, pretrained_word_to_idx, pretrained_idx_to_word = load_pretrained_word2vec()
    
    # Load text from Harry Potter books
    processed_books = {}
    
    # Process all books
    for book_num in range(1, 8):
        filename = f'HP{book_num}.txt'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                processed_lines = []
                
                for line in lines:
                    line = line.strip()
                    if len(line) > 50:  # Only process substantial lines
                        processed_words = preprocess(line)
                        if len(processed_words) > 10:  # Only add meaningful sequences
                            processed_lines.append(processed_words)
                
                processed_books[f'HP{book_num}'] = processed_lines
                print(f"Loaded HP{book_num}: {len(processed_lines)} pages")
                
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            continue

    if len(processed_books) == 0:
        print("Error: No Harry Potter books found!")
        return None

    # Prepare training data
    max_sequence_length = 100 # Increased for better context
    X, y, master_word_to_idx, vocab_size = prepare_improved_training_data(
        processed_books, pretrained_word_to_idx, max_sequence_length
    )
    
    # Print class distribution
    print("\nClass distribution:")
    class_counts = np.sum(y, axis=0)
    for i, count in enumerate(class_counts):
        print(f"  HP{i+1}: {int(count)} samples")
    
    # Create embedding matrix if we have pre-trained embeddings
    embedding_matrix = None
    embedding_dim = 100
    
    if pretrained_embeddings is not None:
        embedding_dim = pretrained_embeddings.shape[1]
        embedding_matrix = create_embedding_matrix(
            pretrained_embeddings, pretrained_word_to_idx, 
            master_word_to_idx, embedding_dim, vocab_size
        )
    
    # Create improved model configurations
    fs = [[2,3,4],[10,15,20],[50,75,100],[150,200,250]]
    config_names = ["Small kernels (2,3,4)", "Medium kernels (10,15,20)", 
                   "Large kernels (50,75,100)", "Very large kernels (150,200,250)"]
    
    models = []
    timing_results = []
    
    # Build all models first
    print("\n" + "="*60)
    print("BUILDING MODELS")
    print("="*60)
    
    for i, filter_config in enumerate(fs):
        print(f"\nBuilding model {i+1}: {config_names[i]}")
        model_build_start = time.time()
        
        model = create_improved_cnn_model(
            max_sequence_length=max_sequence_length,
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            embedding_dim=embedding_dim,
            num_books=7,
            trainable_embeddings=False,
            filter_sizes=filter_config
        )
        
        model_build_time = time.time() - model_build_start
        print(f"Model built in {format_time(model_build_time)}")
        
        models.append(model)
        timing_results.append({
            'config_name': config_names[i],
            'filter_sizes': filter_config,
            'build_time': model_build_time,
            'param_count': model.count_params()
        })

    # Split data once for all models
    print(f"\n{'='*60}")
    print("PREPARING DATA SPLITS")
    print("="*60)
    
    data_split_start = time.time()
    
    from sklearn.model_selection import train_test_split
    
    # Convert one-hot back to labels for stratification
    y_labels = np.argmax(y, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, 
        stratify=np.argmax(y_train, axis=1)
    )
    
    data_split_time = time.time() - data_split_start
    
    print(f"Data splits prepared in {format_time(data_split_time)}")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples") 
    print(f"  Testing: {X_test.shape[0]} samples")

    # Train each model and track timing
    count = 1
    for model_idx, model in enumerate(models):
        config_name = config_names[model_idx]
        filter_config = fs[model_idx]
        
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL {count}: {config_name}")
        print(f"Filter sizes: {filter_config}")
        print(f"Parameters: {model.count_params():,}")
        print("="*80)
        
        # Model summary
        print(f"\nModel architecture:")
        model.summary()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Train the model with timing
        print(f"\nStarting training at {datetime.datetime.now().strftime('%H:%M:%S')}")
        training_start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        
        print(f"\nTraining completed at {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"Training time: {format_time(training_time)}")
        
        # Evaluation timing
        eval_start_time = time.time()
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        eval_time = time.time() - eval_start_time
        
        # Store timing results
        timing_results[model_idx].update({
            'training_time': training_time,
            'eval_time': eval_time,
            'total_time': training_time + eval_time,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'epochs_trained': len(history.history['loss']),
            'best_val_accuracy': max(history.history['val_accuracy']),
            'time_per_epoch': training_time / len(history.history['loss'])
        })
        
        print(f"\nFinal Results for {config_name}:")
        print(f"  Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Training time: {format_time(training_time)}")
        print(f"  Evaluation time: {format_time(eval_time)}")
        print(f"  Total time: {format_time(training_time + eval_time)}")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        print(f"  Time per epoch: {format_time(training_time / len(history.history['loss']))}")
        
        # Detailed evaluation
        print(f"\nGenerating detailed analysis...")
        analysis_start = time.time()
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        print("\nClassification Report:")
        book_names = [f'HP{i+1}' for i in range(7)]
        print(classification_report(y_test_classes, y_pred_classes, 
                                target_names=book_names))
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=book_names, yticklabels=book_names)
        plt.title(f'Confusion Matrix - {config_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{count}_{config_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
          
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy - {config_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - {config_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_history_{count}_{config_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
        
        analysis_time = time.time() - analysis_start
        print(f"Analysis completed in {format_time(analysis_time)}")
        
        count += 1
    
    # Print comprehensive timing comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TIMING COMPARISON")
    print("="*80)
    
    print(f"{'Model':<25} {'Filters':<20} {'Params':<10} {'Accuracy':<10} {'Train Time':<12} {'Time/Epoch':<12}")
    print("-" * 100)
    
    for result in timing_results:
        print(f"{result['config_name'][:24]:<25} "
              f"{str(result['filter_sizes']):<20} "
              f"{result['param_count']:,<10} "
              f"{result['test_accuracy']:.4f}    "
              f"{format_time(result['training_time']):<12} "
              f"{format_time(result['time_per_epoch']):<12}")
    
    # Find best performers
    best_accuracy = max(timing_results, key=lambda x: x['test_accuracy'])
    fastest_training = min(timing_results, key=lambda x: x['training_time'])
    best_efficiency = min(timing_results, key=lambda x: x['training_time'] / x['test_accuracy'])
    
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"🏆 Best Accuracy: {best_accuracy['config_name']} ({best_accuracy['test_accuracy']:.4f})")
    print(f"⚡ Fastest Training: {fastest_training['config_name']} ({format_time(fastest_training['training_time'])})")
    print(f"🎯 Best Efficiency (time/accuracy): {best_efficiency['config_name']}")
    
    return model, master_word_to_idx, history, timing_results

if __name__ == "__main__":
    main()