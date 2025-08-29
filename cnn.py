import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dot, Reshape
from tensorflow.keras.optimizers import Adam
import re
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
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

def create_comprehensive_experiment_configs():
    """Create extensive experiment configurations with kernel sizes and sequence lengths"""
    
    # Define comprehensive configurations
    configs = [
        # Small kernels with different sequence lengths
        {"name": "Tiny-Short", "kernels": [1, 2], "seq_len": 50, "filters": 128},
        {"name": "Tiny-Medium", "kernels": [1, 2], "seq_len": 100, "filters": 128},
        {"name": "Tiny-Long", "kernels": [1, 2], "seq_len": 150, "filters": 128},
        {"name": "Tiny-XLong", "kernels": [1, 2], "seq_len": 200, "filters": 128},
        
        # Small kernels
        {"name": "Small-Short", "kernels": [2, 3, 4], "seq_len": 50, "filters": 128},
        {"name": "Small-Medium", "kernels": [2, 3, 4], "seq_len": 100, "filters": 128},
        {"name": "Small-Long", "kernels": [2, 3, 4], "seq_len": 150, "filters": 128},
        {"name": "Small-XLong", "kernels": [2, 3, 4], "seq_len": 200, "filters": 128},
        
        # Medium kernels  
        {"name": "Medium-Short", "kernels": [5, 7, 9], "seq_len": 50, "filters": 96},
        {"name": "Medium-Medium", "kernels": [5, 7, 9], "seq_len": 100, "filters": 96},
        {"name": "Medium-Long", "kernels": [5, 7, 9], "seq_len": 150, "filters": 96},
        {"name": "Medium-XLong", "kernels": [5, 7, 9], "seq_len": 200, "filters": 96},
        
        # Large kernels
        {"name": "Large-Short", "kernels": [10, 15, 20], "seq_len": 50, "filters": 64},
        {"name": "Large-Medium", "kernels": [10, 15, 20], "seq_len": 100, "filters": 64},
        {"name": "Large-Long", "kernels": [10, 15, 20], "seq_len": 150, "filters": 64},
        {"name": "Large-XLong", "kernels": [10, 15, 20], "seq_len": 200, "filters": 64},
        
        # Very large kernels
        {"name": "XLarge-Medium", "kernels": [25, 35, 45], "seq_len": 100, "filters": 48},
        {"name": "XLarge-Long", "kernels": [25, 35, 45], "seq_len": 150, "filters": 48},
        {"name": "XLarge-XLong", "kernels": [25, 35, 45], "seq_len": 200, "filters": 48},
        {"name": "XLarge-XXLong", "kernels": [25, 35, 45], "seq_len": 250, "filters": 48},
        
        # Extreme kernels (testing limits)
        {"name": "Extreme-Long", "kernels": [50, 75, 100], "seq_len": 200, "filters": 32},
        {"name": "Extreme-XLong", "kernels": [50, 75, 100], "seq_len": 250, "filters": 32},
        {"name": "Extreme-XXLong", "kernels": [50, 75, 100], "seq_len": 300, "filters": 32},
        
        # Single kernel experiments for pure comparison
        {"name": "Single-1", "kernels": [1], "seq_len": 100, "filters": 256},
        {"name": "Single-3", "kernels": [3], "seq_len": 100, "filters": 256},
        {"name": "Single-5", "kernels": [5], "seq_len": 100, "filters": 256},
        {"name": "Single-10", "kernels": [10], "seq_len": 100, "filters": 256},
        {"name": "Single-20", "kernels": [20], "seq_len": 100, "filters": 256},
        {"name": "Single-50", "kernels": [50], "seq_len": 100, "filters": 256},
        
        # Efficiency optimized
        {"name": "Fast-Small", "kernels": [2, 3], "seq_len": 50, "filters": 64},
        {"name": "Fast-Medium", "kernels": [3, 4], "seq_len": 75, "filters": 64},
        {"name": "Fast-Large", "kernels": [4, 5], "seq_len": 100, "filters": 64},
        
        # Accuracy optimized (based on literature)
        {"name": "Optimal-A", "kernels": [3, 4, 5], "seq_len": 128, "filters": 100},
        {"name": "Optimal-B", "kernels": [3, 4, 5, 6], "seq_len": 128, "filters": 80},
        {"name": "Optimal-C", "kernels": [2, 3, 4, 5, 6], "seq_len": 128, "filters": 64},
    ]
    
    return configs

def create_custom_cnn_model(max_sequence_length, vocab_size, embedding_matrix=None,
                           embedding_dim=100, filter_sizes=[3, 4, 5], num_filters=128):
    """Create CNN model with custom filter count"""
    
    inputs = Input(shape=(max_sequence_length,))
    
    if embedding_matrix is not None:
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                            weights=[embedding_matrix], trainable=False)(inputs)
    else:
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    
    x = Dropout(0.2)(embedding)
    
    # Multiple CNN layers with specified filter count
    conv_layers = []
    for filter_size in filter_sizes:
        conv = Conv1D(num_filters, filter_size, activation='relu', padding='same')(x)
        conv = GlobalMaxPooling1D()(conv)
        conv_layers.append(conv)
    
    # Concatenate all conv layers
    if len(conv_layers) > 1:
        from tensorflow.keras.layers import concatenate
        merged = concatenate(conv_layers)
    else:
        merged = conv_layers[0]
    
    # Dense layers
    dense1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(merged)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(dense1)
    dense2 = Dropout(0.3)(dense2)
    
    outputs = Dense(7, activation='softmax')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    print("Comprehensive CNN Text Classifier Experiment for Harry Potter Books")
    print("=" * 80)
    
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

    # Get comprehensive experiment configurations
    experiment_configs = create_comprehensive_experiment_configs()
    
    print(f"\n{'='*80}")
    print(f"TOTAL EXPERIMENTS: {len(experiment_configs)}")
    print(f"This will test {len(experiment_configs)} different configurations!")
    print(f"{'='*80}")
    
    # Prepare different datasets for different sequence lengths
    datasets = {}
    sequence_lengths = list(set([config['seq_len'] for config in experiment_configs]))
    
    print(f"\nPreparing datasets for sequence lengths: {sorted(sequence_lengths)}")
    
    for seq_len in sequence_lengths:
        print(f"\nPreparing data for sequence length: {seq_len}")
        X, y, master_word_to_idx, vocab_size = prepare_improved_training_data(
            processed_books, pretrained_word_to_idx, seq_len
        )
        
        # Create embedding matrix if we have pre-trained embeddings
        embedding_matrix = None
        embedding_dim = 100
        
        if pretrained_embeddings is not None:
            embedding_dim = pretrained_embeddings.shape[1]
            embedding_matrix = create_embedding_matrix(
                pretrained_embeddings, pretrained_word_to_idx, 
                master_word_to_idx, embedding_dim, vocab_size
            )
        
        # Split data
        from sklearn.model_selection import train_test_split
        y_labels = np.argmax(y, axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, 
            stratify=np.argmax(y_train, axis=1)
        )
        
        datasets[seq_len] = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'vocab_size': vocab_size, 'embedding_matrix': embedding_matrix,
            'embedding_dim': embedding_dim, 'master_word_to_idx': master_word_to_idx
        }
        
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Testing: {X_test.shape[0]} samples")
    
    # Run all experiments
    all_results = []
    experiment_start_time = time.time()
    
    print(f"\n{'='*80}")
    print("STARTING COMPREHENSIVE EXPERIMENTS")
    print(f"{'='*80}")
    
    for exp_idx, config in enumerate(experiment_configs, 1):
        config_name = config['name']
        kernels = config['kernels']
        seq_len = config['seq_len']
        num_filters = config['filters']
        
        print(f"\n{'='*100}")
        print(f"EXPERIMENT {exp_idx}/{len(experiment_configs)}: {config_name}")
        print(f"Kernels: {kernels}, Sequence Length: {seq_len}, Filters: {num_filters}")
        print(f"{'='*100}")
        
        # Get dataset for this sequence length
        dataset = datasets[seq_len]
        
        try:
            # Create model
            model_build_start = time.time()
            model = create_custom_cnn_model(
                max_sequence_length=seq_len,
                vocab_size=dataset['vocab_size'],
                embedding_matrix=dataset['embedding_matrix'],
                embedding_dim=dataset['embedding_dim'],
                filter_sizes=kernels,
                num_filters=num_filters
            )
            
            model_build_time = time.time() - model_build_start
            param_count = model.count_params()
            
            print(f"Model built in {format_time(model_build_time)}")
            print(f"Parameters: {param_count:,}")
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ]
            
            # Train model
            print(f"\nStarting training at {datetime.datetime.now().strftime('%H:%M:%S')}")
            training_start_time = time.time()
            
            history = model.fit(
                dataset['X_train'], dataset['y_train'],
                validation_data=(dataset['X_val'], dataset['y_val']),
                epochs=15,  # Reduced for faster experimentation
                batch_size=64,
                callbacks=callbacks
            )
            
            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            
            # Evaluate
            eval_start_time = time.time()
            test_loss, test_accuracy = model.evaluate(dataset['X_test'], dataset['y_test'], verbose=0)
            eval_time = time.time() - eval_start_time
            
            # Store results
            result = {
                'experiment_id': exp_idx,
                'config_name': config_name,
                'kernels': kernels,
                'sequence_length': seq_len,
                'num_filters': num_filters,
                'param_count': param_count,
                'build_time': model_build_time,
                'training_time': training_time,
                'eval_time': eval_time,
                'total_time': training_time + eval_time,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'epochs_trained': len(history.history['loss']),
                'best_val_accuracy': max(history.history['val_accuracy']),
                'final_val_accuracy': history.history['val_accuracy'][-1],
                'time_per_epoch': training_time / len(history.history['loss']),
                'convergence_epoch': history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1,
                'status': 'SUCCESS'
            }
            
            all_results.append(result)
            
            print(f"\n🎯 EXPERIMENT {exp_idx} RESULTS:")
            print(f"  ✅ Status: SUCCESS")
            print(f"  📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"  ⏱️  Training Time: {format_time(training_time)}")
            print(f"  🔢 Parameters: {param_count:,}")
            print(f"  📈 Best Val Acc: {max(history.history['val_accuracy']):.4f}")
            print(f"  🎯 Converged at Epoch: {result['convergence_epoch']}")
            
            # Memory cleanup
            del model, history
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"❌ EXPERIMENT {exp_idx} FAILED: {str(e)}")
            
            error_result = {
                'experiment_id': exp_idx,
                'config_name': config_name,
                'kernels': kernels,
                'sequence_length': seq_len,
                'num_filters': num_filters,
                'status': f'FAILED: {str(e)}',
                'test_accuracy': 0.0,
                'training_time': 0.0
            }
            all_results.append(error_result)
            continue
        
        # Progress update
        elapsed_time = time.time() - experiment_start_time
        avg_time_per_exp = elapsed_time / exp_idx
        remaining_experiments = len(experiment_configs) - exp_idx
        estimated_remaining_time = avg_time_per_exp * remaining_experiments
        
        print(f"\n📊 PROGRESS UPDATE:")
        print(f"  Completed: {exp_idx}/{len(experiment_configs)} ({exp_idx/len(experiment_configs)*100:.1f}%)")
        print(f"  Elapsed: {format_time(elapsed_time)}")
        print(f"  Estimated Remaining: {format_time(estimated_remaining_time)}")
        print(f"  Estimated Total: {format_time(elapsed_time + estimated_remaining_time)}")
    
    # Comprehensive analysis
    print(f"\n{'='*100}")
    print("COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("="*100)
    
    successful_results = [r for r in all_results if r['status'] == 'SUCCESS']
    failed_results = [r for r in all_results if r['status'] != 'SUCCESS']
    
    print(f"\n📊 EXPERIMENT SUMMARY:")
    print(f"  Total Experiments: {len(all_results)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    print(f"  Success Rate: {len(successful_results)/len(all_results)*100:.1f}%")
    
    if successful_results:
        # Sort by accuracy
        successful_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        print(f"\n🏆 TOP 10 PERFORMERS (by accuracy):")
        print(f"{'Rank':<5} {'Config':<20} {'Kernels':<15} {'SeqLen':<7} {'Accuracy':<10} {'Time':<12}")
        print("-" * 85)
        
        for i, result in enumerate(successful_results[:10], 1):
            print(f"{i:<5} {result['config_name']:<20} {str(result['kernels']):<15} "
                  f"{result['sequence_length']:<7} {result['test_accuracy']:.4f}    "
                  f"{format_time(result['training_time']):<12}")
        
        # Analysis by categories
        print(f"\n📈 ANALYSIS BY KERNEL SIZE:")
        kernel_analysis = {}
        for result in successful_results:
            max_kernel = max(result['kernels'])
            if max_kernel <= 5:
                category = 'Small (≤5)'
            elif max_kernel <= 15:
                category = 'Medium (6-15)'
            elif max_kernel <= 50:
                category = 'Large (16-50)'
            else:
                category = 'XLarge (>50)'
            
            if category not in kernel_analysis:
                kernel_analysis[category] = []
            kernel_analysis[category].append(result['test_accuracy'])
        
        for category, accuracies in kernel_analysis.items():
            avg_acc = np.mean(accuracies)
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            std_acc = np.std(accuracies)
            print(f"  {category}: Avg={avg_acc:.4f}, Max={max_acc:.4f}, Min={min_acc:.4f}, Std={std_acc:.4f} (n={len(accuracies)})")
        
        print(f"\n📏 ANALYSIS BY SEQUENCE LENGTH:")
        seq_analysis = {}
        for result in successful_results:
            seq_len = result['sequence_length']
            if seq_len not in seq_analysis:
                seq_analysis[seq_len] = []
            seq_analysis[seq_len].append(result['test_accuracy'])
        
        for seq_len, accuracies in sorted(seq_analysis.items()):
            avg_acc = np.mean(accuracies)
            max_acc = max(accuracies)
            std_acc = np.std(accuracies)
            print(f"  Length {seq_len}: Avg={avg_acc:.4f}, Max={max_acc:.4f}, Std={std_acc:.4f} (n={len(accuracies)})")
        
        # Efficiency analysis
        print(f"\n⚡ EFFICIENCY ANALYSIS (Accuracy per Training Hour):")
        efficiency_results = []
        for result in successful_results:
            efficiency = result['test_accuracy'] / (result['training_time'] / 3600)  # accuracy per hour
            efficiency_results.append((result['config_name'], efficiency, result['test_accuracy'], result['training_time']))
        
        efficiency_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<5} {'Config':<20} {'Efficiency':<12} {'Accuracy':<10} {'Time':<12}")
        print("-" * 70)
        for i, (config, eff, acc, train_time) in enumerate(efficiency_results[:10], 1):
            print(f"{i:<5} {config:<20} {eff:.3f}      {acc:.4f}    {format_time(train_time):<12}")
    
    # Save comprehensive results
    import json
    results_filename = f"comprehensive_experiment_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy types for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, (np.integer, np.floating)):
                json_result[key] = float(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
                json_result[key] = [float(v) for v in value]
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    with open(results_filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_filename}")
    
    total_experiment_time = time.time() - experiment_start_time
    print(f"\n🏁 TOTAL EXPERIMENT TIME: {format_time(total_experiment_time)}")
    
    return all_results

if __name__ == "__main__":
    main()