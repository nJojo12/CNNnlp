#!/usr/bin/env python3
"""
Word2Vec Implementation with Comprehensive Metric Analysis
COMS4054A/COMS7062A/COMS7066A: Natural Language Processing Lab 1

Subtopic: How does the metric you use to interpret the embedding space affect your conclusions?

This implementation trains a Word2Vec model and then analyzes the embedding space
using multiple metrics to demonstrate how different metrics lead to different conclusions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import random
from typing import List, Tuple, Dict, Set
import json
import pandas as pd
from scipy.spatial.distance import cdist, jaccard
from scipy.stats import spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class TextPreprocessor:
    """Handles text preprocessing with configurable options"""
    
    def __init__(self, min_word_freq=2, lowercase=True):
        self.min_word_freq = min_word_freq
        self.lowercase = lowercase
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text"""
        if self.lowercase:
            text = text.lower()
        
        text = re.sub(r"[^\w\s'-]", " ", text)
        
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'s": " is",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        words = [w for w in text.split() if w]
        return words
    
    def build_vocabulary(self, words: List[str]) -> None:
        """Build vocabulary from word list"""
        self.word_freq = Counter(words)
        
        vocab_words = [w for w, freq in self.word_freq.items() 
                      if freq >= self.min_word_freq]
        
        seen = set()
        ordered_vocab = []
        for w in words:
            if w in vocab_words and w not in seen:
                ordered_vocab.append(w)
                seen.add(w)
        
        self.word_to_idx = {w: i for i, w in enumerate(ordered_vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(ordered_vocab)}
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        print(f"Total words: {len(words)}")
        print(f"Unique words: {len(self.word_freq)}")

class SkipGramDataset:
    """Creates skip-gram training pairs"""
    
    def __init__(self, words: List[str], word_to_idx: Dict[str, int], 
                 window_size: int = 2, dynamic_window: bool = True):
        self.words = words
        self.word_to_idx = word_to_idx
        self.window_size = window_size
        self.dynamic_window = dynamic_window
        self.pairs = []
        self._create_pairs()
    
    def _create_pairs(self) -> None:
        """Create (center_word, context_word) pairs"""
        for i, center_word in enumerate(self.words):
            if center_word not in self.word_to_idx:
                continue
                
            center_idx = self.word_to_idx[center_word]
            
            if self.dynamic_window:
                actual_window = random.randint(1, self.window_size)
            else:
                actual_window = self.window_size
            
            for j in range(max(0, i - actual_window), 
                          min(len(self.words), i + actual_window + 1)):
                if i != j and self.words[j] in self.word_to_idx:
                    context_idx = self.word_to_idx[self.words[j]]
                    self.pairs.append((center_idx, context_idx))
        
        print(f"Created {len(self.pairs)} training pairs")
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of training pairs"""
        batch_indices = random.sample(range(len(self.pairs)), 
                                    min(batch_size, len(self.pairs)))
        
        center_words = []
        context_words = []
        
        for idx in batch_indices:
            center, context = self.pairs[idx]
            center_words.append(center)
            context_words.append(context)
        
        return (torch.tensor(center_words, dtype=torch.long),
                torch.tensor(context_words, dtype=torch.long))

class SkipGramModel(nn.Module):
    """Skip-gram model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 init_std: float = 0.01, sparse_gradients: bool = True):
        super(SkipGramModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.center_embeddings = nn.Embedding(
            vocab_size, embedding_dim, sparse=sparse_gradients
        )
        self.context_embeddings = nn.Embedding(
            vocab_size, embedding_dim, sparse=sparse_gradients
        )
        
        nn.init.normal_(self.center_embeddings.weight, std=init_std)
        nn.init.normal_(self.context_embeddings.weight, std=init_std)
    
    def forward(self, center_words: torch.Tensor, context_words: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        center_embeds = self.center_embeddings(center_words)
        context_embeds = self.context_embeddings(context_words)
        scores = torch.sum(center_embeds * context_embeds, dim=1)
        return scores
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get embedding for a word index"""
        with torch.no_grad():
            embed = self.center_embeddings.weight[word_idx].cpu().numpy()
        return embed

class Word2VecTrainer:
    """Handles training with negative sampling"""
    
    def __init__(self, model: SkipGramModel, vocab_size: int, 
                 neg_samples: int = 5, neg_sample_power: float = 0.75):
        self.model = model
        self.vocab_size = vocab_size
        self.neg_samples = neg_samples
        self.neg_sample_power = neg_sample_power
        self.neg_sampling_dist = None
        self.loss_history = []
    
    def setup_negative_sampling(self, word_freq: Counter, word_to_idx: Dict[str, int]):
        """Setup negative sampling distribution"""
        freq_array = np.zeros(self.vocab_size)
        
        for word, freq in word_freq.items():
            if word in word_to_idx:
                freq_array[word_to_idx[word]] = freq
        
        freq_array = np.power(freq_array, self.neg_sample_power)
        self.neg_sampling_dist = freq_array / freq_array.sum()
    
    def train(self, dataset: SkipGramDataset, epochs: int = 10, 
              batch_size: int = 512, lr: float = 0.025, lr_decay: bool = True):
        """Train the model"""
        optimizer = optim.SparseAdam(self.model.parameters(), lr=lr)
        
        steps_per_epoch = len(dataset.pairs) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            if lr_decay:
                current_lr = lr * (1 - (epoch / epochs))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            for step in range(steps_per_epoch):
                center_words, context_words = dataset.get_batch(batch_size)
                
                pos_scores = self.model(center_words, context_words)
                pos_loss = -torch.log(torch.sigmoid(pos_scores)).mean()
                
                neg_loss = 0
                for _ in range(self.neg_samples):
                    neg_words = torch.tensor(
                        np.random.choice(self.vocab_size, size=batch_size, 
                                       p=self.neg_sampling_dist),
                        dtype=torch.long
                    )
                    neg_scores = self.model(center_words, neg_words)
                    neg_loss += -torch.log(torch.sigmoid(-neg_scores)).mean()
                
                loss = pos_loss + neg_loss
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}/{steps_per_epoch}, "
                          f"Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / steps_per_epoch
            self.loss_history.append(avg_epoch_loss)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

class MetricAnalyzer:
    """Comprehensive analysis of embeddings using different metrics"""
    
    def __init__(self, model: SkipGramModel, idx_to_word: Dict[int, str], 
                 word_to_idx: Dict[str, int], word_freq: Counter):
        self.model = model
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.word_freq = word_freq
        self.embeddings = None
        self._extract_embeddings()
        
    def _extract_embeddings(self):
        """Extract embeddings as numpy array"""
        with torch.no_grad():
            self.embeddings = self.model.center_embeddings.weight.cpu().numpy()
    
    def compute_similarities(self, metric: str = 'cosine') -> np.ndarray:
        """Compute pairwise similarities/distances using specified metric"""
        if metric == 'cosine':
            # Normalize embeddings for cosine similarity
            normalized = normalize(self.embeddings, axis=1, norm='l2')
            similarities = np.dot(normalized, normalized.T)
            return similarities
        
        elif metric == 'euclidean':
            distances = pairwise_distances(self.embeddings, metric='euclidean')
            return -distances  # Negative so higher = more similar
        
        elif metric == 'manhattan':
            distances = pairwise_distances(self.embeddings, metric='manhattan')
            return -distances
        
        elif metric == 'dot':
            return np.dot(self.embeddings, self.embeddings.T)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_nearest_neighbors(self, word: str, metric: str = 'cosine', k: int = 10) -> List[Tuple[str, float]]:
        """Find k nearest neighbors using specified metric"""
        if word not in self.word_to_idx:
            return []
        
        word_idx = self.word_to_idx[word]
        
        if metric == 'cosine':
            normalized = normalize(self.embeddings, axis=1, norm='l2')
            similarities = np.dot(normalized[word_idx], normalized.T)
            indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self
            scores = similarities[indices]
        
        elif metric == 'euclidean':
            distances = np.linalg.norm(self.embeddings - self.embeddings[word_idx], axis=1)
            indices = np.argsort(distances)[1:k+1]  # Exclude self
            scores = -distances[indices]  # Negative for consistency
        
        elif metric == 'manhattan':
            distances = np.sum(np.abs(self.embeddings - self.embeddings[word_idx]), axis=1)
            indices = np.argsort(distances)[1:k+1]
            scores = -distances[indices]
        
        elif metric == 'dot':
            dot_products = np.dot(self.embeddings[word_idx], self.embeddings.T)
            indices = np.argsort(dot_products)[::-1][1:k+1]
            scores = dot_products[indices]
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        neighbors = [(self.idx_to_word[idx], scores[i]) for i, idx in enumerate(indices)]
        return neighbors
    
    def compute_jaccard_similarity(self, word1: str, word2: str, k: int = 10) -> float:
        """Compute Jaccard similarity based on k-nearest neighbors"""
        if word1 not in self.word_to_idx or word2 not in self.word_to_idx:
            return 0.0
        
        # Get k-nearest neighbors for both words
        neighbors1 = set([w for w, _ in self.find_nearest_neighbors(word1, 'cosine', k)])
        neighbors2 = set([w for w, _ in self.find_nearest_neighbors(word2, 'cosine', k)])
        
        # Compute Jaccard index
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0.0
    
    def compare_metrics_on_word(self, word: str, k: int = 10) -> pd.DataFrame:
        """Compare nearest neighbors across different metrics"""
        if word not in self.word_to_idx:
            return pd.DataFrame()
        
        metrics = ['cosine', 'euclidean', 'manhattan', 'dot']
        results = {}
        
        for metric in metrics:
            neighbors = self.find_nearest_neighbors(word, metric, k)
            results[metric] = [f"{w} ({s:.3f})" for w, s in neighbors]
        
        # Pad lists to same length
        max_len = max(len(v) for v in results.values())
        for metric in metrics:
            results[metric] += [''] * (max_len - len(results[metric]))
        
        return pd.DataFrame(results)
    
    def analyze_metric_agreement(self, test_words: List[str], k: int = 10) -> Dict:
        """Analyze how much different metrics agree on nearest neighbors"""
        metrics = ['cosine', 'euclidean', 'manhattan', 'dot']
        agreements = {}
        
        # Initialize all possible metric pairs
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                agreements[f"{metrics[i]}-{metrics[j]}"] = []
        
        for word in test_words:
            if word not in self.word_to_idx:
                continue
            
            # Get neighbors for each metric
            neighbors_by_metric = {}
            for metric in metrics:
                neighbors = self.find_nearest_neighbors(word, metric, k)
                neighbors_by_metric[metric] = set([w for w, _ in neighbors])
            
            # Compute pairwise Jaccard similarities
            for i in range(len(metrics)):
                for j in range(i+1, len(metrics)):
                    m1, m2 = metrics[i], metrics[j]
                    set1 = neighbors_by_metric[m1]
                    set2 = neighbors_by_metric[m2]
                    jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
                    agreements[f"{m1}-{m2}"].append(jaccard)
        
        # Compute average agreements
        avg_agreements = {k: np.mean(v) if v else 0 for k, v in agreements.items()}
        return avg_agreements
    
    def perform_analogy_test(self, a: str, b: str, c: str, metric: str = 'cosine', top_k: int = 5) -> List[Tuple[str, float]]:
        """Perform analogy test: a is to b as c is to ?"""
        if not all(w in self.word_to_idx for w in [a, b, c]):
            return []
        
        # Get embeddings
        a_vec = self.embeddings[self.word_to_idx[a]]
        b_vec = self.embeddings[self.word_to_idx[b]]
        c_vec = self.embeddings[self.word_to_idx[c]]
        
        # Compute target vector
        target_vec = b_vec - a_vec + c_vec
        
        # Find nearest neighbors based on metric
        exclude_words = {a, b, c}
        
        if metric == 'cosine':
            target_norm = target_vec / np.linalg.norm(target_vec)
            embeddings_norm = normalize(self.embeddings, axis=1, norm='l2')
            similarities = np.dot(embeddings_norm, target_norm)
            
            candidates = []
            for idx, sim in enumerate(similarities):
                word = self.idx_to_word[idx]
                if word not in exclude_words:
                    candidates.append((word, sim))
        
        elif metric == 'euclidean':
            distances = np.linalg.norm(self.embeddings - target_vec, axis=1)
            
            candidates = []
            for idx, dist in enumerate(distances):
                word = self.idx_to_word[idx]
                if word not in exclude_words:
                    candidates.append((word, -dist))  # Negative for consistency
        
        elif metric == 'manhattan':
            distances = np.sum(np.abs(self.embeddings - target_vec), axis=1)
            
            candidates = []
            for idx, dist in enumerate(distances):
                word = self.idx_to_word[idx]
                if word not in exclude_words:
                    candidates.append((word, -dist))
        
        elif metric == 'dot':
            dot_products = np.dot(self.embeddings, target_vec)
            
            candidates = []
            for idx, dot in enumerate(dot_products):
                word = self.idx_to_word[idx]
                if word not in exclude_words:
                    candidates.append((word, dot))
        
        # Sort and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def visualize_metric_comparison(self, test_words: List[str], save_path: str = None):
        """Create comprehensive visualization comparing metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Heatmap of metric agreement
        agreement = self.analyze_metric_agreement(test_words)
        metrics = ['cosine', 'euclidean', 'manhattan', 'dot']
        agreement_matrix = np.zeros((len(metrics), len(metrics)))
        
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                elif i < j:
                    agreement_matrix[i, j] = agreement.get(f"{m1}-{m2}", 0)
                    agreement_matrix[j, i] = agreement_matrix[i, j]
        
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', 
                   xticklabels=metrics, yticklabels=metrics,
                   cmap='coolwarm', center=0.5, ax=axes[0])
        axes[0].set_title('Metric Agreement on Nearest Neighbors', fontsize=14)
        
        # # 2. PCA visualization colored by different metrics
        # pca = PCA(n_components=2, random_state=42)
        # pca_embeddings = pca.fit_transform(self.embeddings)
        
        # # Get word frequencies for coloring
        # freq_values = np.array([self.word_freq.get(self.idx_to_word[i], 0) 
        #                        for i in range(len(self.embeddings))])
        
        # scatter = axes[1].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], 
        #                         c=np.log1p(freq_values), cmap='viridis', 
        #                         alpha=0.6, s=30)
        # axes[1].set_title('PCA: Colored by Log Word Frequency', fontsize=14)
        # axes[1].set_xlabel('PC1')
        # axes[1].set_ylabel('PC2')
        # plt.colorbar(scatter, ax=axes[1])
        
        # # 3. t-SNE visualization
        # if len(test_words) > 5:
        #     # Get indices for test words
        #     test_indices = [self.word_to_idx[w] for w in test_words[:30] 
        #                    if w in self.word_to_idx]
            
        #     if len(test_indices) > 5:
        #         tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_indices)-1))
        #         tsne_embeddings = tsne.fit_transform(self.embeddings[test_indices])
                
        #         axes[2].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0.7)
        #         for i, idx in enumerate(test_indices):
        #             axes[2].annotate(self.idx_to_word[idx], 
        #                            (tsne_embeddings[i, 0], tsne_embeddings[i, 1]),
        #                            fontsize=8, alpha=0.8)
        #         axes[2].set_title('t-SNE: Test Words', fontsize=14)
        
        # # 5. Analogy performance across metrics
        # analogies = [
        #     ('man', 'woman', 'king'),
        #     ('good', 'bad', 'happy'),
        #     ('boy', 'girl', 'man')
        # ]
        
        # analogy_results = defaultdict(list)
        # for a, b, c in analogies:
        #     if all(w in self.word_to_idx for w in [a, b, c]):
        #         for metric in ['cosine', 'euclidean', 'manhattan', 'dot']:
        #             results = self.perform_analogy_test(a, b, c, metric, top_k=1)
        #             if results:
        #                 analogy_results[metric].append(results[0][0])
        
        # # Plot analogy results
        # if analogy_results:
        #     y_pos = np.arange(len(analogies))
        #     width = 0.2
            
        #     for i, (metric, results) in enumerate(analogy_results.items()):
        #         axes[4].text(i, 0.5, f"{metric}\n" + "\n".join(results[:3]), 
        #                    ha='center', va='center', fontsize=10)
            
        #     axes[4].set_xlim(-0.5, 3.5)
        #     axes[4].set_ylim(0, 1)
        #     axes[4].set_title('Top Analogy Results by Metric', fontsize=14)
        #     axes[4].axis('off')
        
        # # 6. Neighbor overlap visualization
        # if test_words:
        #     test_word = test_words[0]
        #     if test_word in self.word_to_idx:
        #         neighbor_sets = {}
        #         for metric in ['cosine', 'euclidean', 'manhattan', 'dot']:
        #             neighbors = self.find_nearest_neighbors(test_word, metric, k=10)
        #             neighbor_sets[metric] = set([w for w, _ in neighbors])
                
        #         # Create Venn diagram manually
        #         axes[5].text(0.5, 0.9, f'Nearest Neighbors of "{test_word}"', 
        #                    ha='center', fontsize=12, weight='bold')
                
        #         y_offset = 0.7
        #         for metric, neighbors in neighbor_sets.items():
        #             axes[5].text(0.1, y_offset, f"{metric}:", fontsize=10, weight='bold')
        #             axes[5].text(0.3, y_offset, ", ".join(list(neighbors)[:5]), 
        #                        fontsize=9, wrap=True)
        #             y_offset -= 0.15
                
        #         axes[5].set_xlim(0, 1)
        #         axes[5].set_ylim(0, 1)
        #         axes[5].axis('off')
        
        # plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_comparison_report(self, test_words: List[str], save_path: str = None):
        """Create detailed comparison report across metrics"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE METRIC COMPARISON REPORT")
        report.append("=" * 80)
        
        # 1. Nearest neighbor comparison
        report.append("\n1. NEAREST NEIGHBOR COMPARISON")
        report.append("-" * 40)
        
        for word in test_words[:5]:  # Limit to first 5 words
            if word not in self.word_to_idx:
                continue
                
            report.append(f"\nWord: {word}")
            df = self.compare_metrics_on_word(word, k=5)
            report.append(df.to_string())
        
        # 2. Metric agreement analysis
        report.append("\n\n2. METRIC AGREEMENT ANALYSIS")
        report.append("-" * 40)
        
        agreement = self.analyze_metric_agreement(test_words, k=10)
        report.append("\nAverage Jaccard similarity between metrics:")
        for pair, score in sorted(agreement.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {pair}: {score:.3f}")
        
        # 3. Analogy performance
        report.append("\n\n3. ANALOGY PERFORMANCE BY METRIC")
        report.append("-" * 40)
        
        analogies = [
            ('man', 'woman', 'king', 'queen'),
            ('boy', 'girl', 'man', 'woman'),
            ('good', 'bad', 'happy', 'sad'),
            ('big', 'small', 'bigger', 'smaller')
        ]
        
        for a, b, c, expected in analogies:
            if all(w in self.word_to_idx for w in [a, b, c]):
                report.append(f"\nAnalogy: {a} is to {b} as {c} is to ?")
                report.append(f"Expected: {expected}")
                
                for metric in ['cosine', 'euclidean', 'manhattan', 'dot']:
                    results = self.perform_analogy_test(a, b, c, metric, top_k=3)
                    if results:
                        top_word = results[0][0]
                        is_correct = "✓" if top_word == expected else "✗"
                        report.append(f"  {metric}: {top_word} {is_correct} (score: {results[0][1]:.3f})")
        
        # 4. Frequency bias analysis
        report.append("\n\n4. FREQUENCY BIAS ANALYSIS")
        report.append("-" * 40)
        
        # Get high and low frequency words
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        high_freq_words = [w for w, _ in sorted_words[:20] if w in self.word_to_idx]
        low_freq_words = [w for w, _ in sorted_words[-100:] if w in self.word_to_idx and self.word_freq[w] >= 5]
        
        report.append("\nAverage distance to high-frequency words:")
        for metric in ['euclidean', 'manhattan']:
            distances = []
            for word in test_words[:10]:
                if word not in self.word_to_idx:
                    continue
                for hf_word in high_freq_words[:5]:
                    if metric == 'euclidean':
                        dist = np.linalg.norm(
                            self.embeddings[self.word_to_idx[word]] - 
                            self.embeddings[self.word_to_idx[hf_word]]
                        )
                    else:  # manhattan
                        dist = np.sum(np.abs(
                            self.embeddings[self.word_to_idx[word]] - 
                            self.embeddings[self.word_to_idx[hf_word]]
                        ))
                    distances.append(dist)
            
            avg_dist = np.mean(distances) if distances else 0
            report.append(f"  {metric}: {avg_dist:.3f}")
        
        # Save report
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        
        return report_text

def analyze_character_relationships_with_metrics(analyzer: MetricAnalyzer):
    """Analyze character relationships using different metrics"""
    print("\n" + "="*80)
    print("CHARACTER RELATIONSHIP ANALYSIS ACROSS METRICS")
    print("="*80)
    
    characters = ['harry', 'ron', 'hermione', 'voldemort', 'dumbledore', 
                 'snape', 'draco', 'hagrid', 'neville', 'mcgonagall']
    
    # Find available characters
    available_chars = [c for c in characters if c in analyzer.word_to_idx]
    print(f"\nCharacters found: {', '.join(available_chars)}")
    
    if len(available_chars) < 2:
        print("Not enough characters found in vocabulary")
        return
    
    # Analyze relationships with each metric
    metrics = ['cosine', 'euclidean', 'manhattan', 'dot']
    
    for metric in metrics:
        print(f"\n\n{metric.upper()} SIMILARITY")
        print("-" * 40)
        
        # Create similarity matrix
        n_chars = len(available_chars)
        similarity_matrix = np.zeros((n_chars, n_chars))
        
        for i, char1 in enumerate(available_chars):
            for j, char2 in enumerate(available_chars):
                if i == j:
                    similarity_matrix[i, j] = 1.0 if metric == 'cosine' else 0.0
                else:
                    idx1 = analyzer.word_to_idx[char1]
                    idx2 = analyzer.word_to_idx[char2]
                    
                    if metric == 'cosine':
                        vec1_norm = analyzer.embeddings[idx1] / np.linalg.norm(analyzer.embeddings[idx1])
                        vec2_norm = analyzer.embeddings[idx2] / np.linalg.norm(analyzer.embeddings[idx2])
                        similarity_matrix[i, j] = np.dot(vec1_norm, vec2_norm)
                    
                    elif metric == 'euclidean':
                        dist = np.linalg.norm(analyzer.embeddings[idx1] - analyzer.embeddings[idx2])
                        similarity_matrix[i, j] = 1 / (1 + dist)  # Convert to similarity
                    
                    elif metric == 'manhattan':
                        dist = np.sum(np.abs(analyzer.embeddings[idx1] - analyzer.embeddings[idx2]))
                        similarity_matrix[i, j] = 1 / (1 + dist)
                    
                    elif metric == 'dot':
                        similarity_matrix[i, j] = np.dot(analyzer.embeddings[idx1], analyzer.embeddings[idx2])
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=[c.capitalize() for c in available_chars],
                   yticklabels=[c.capitalize() for c in available_chars],
                   annot=True, fmt='.2f', cmap='coolwarm',
                   center=0 if metric != 'dot' else None)
        plt.title(f'Character Relationships - {metric.capitalize()} Metric', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'character_relationships_{metric}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top relationships
        print(f"\nTop character relationships ({metric}):")
        relationships = []
        for i in range(n_chars):
            for j in range(i+1, n_chars):
                relationships.append((available_chars[i], available_chars[j], similarity_matrix[i, j]))
        
        relationships.sort(key=lambda x: x[2], reverse=True)
        for char1, char2, sim in relationships[:5]:
            print(f"  {char1.capitalize()}-{char2.capitalize()}: {sim:.3f}")

def create_metric_comparison_visualizations(analyzer: MetricAnalyzer, test_words: List[str]):
    """Create comprehensive visualizations comparing all metrics"""
    
    # 1. Create a figure showing how different metrics rank words differently
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select a focal word
    focal_word = 'harry' if 'harry' in analyzer.word_to_idx else test_words[0]
    
    metrics = ['cosine', 'euclidean', 'manhattan', 'dot']
    colors = ['blue', 'red', 'green', 'purple']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        neighbors = analyzer.find_nearest_neighbors(focal_word, metric, k=15)
        
        words = [w for w, _ in neighbors]
        scores = [s for _, s in neighbors]
        
        # Normalize scores for visualization
        if metric in ['euclidean', 'manhattan']:
            scores = [-s for s in scores]  # Make positive for visualization
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        axes[idx].barh(range(len(words)), scores, color=color, alpha=0.7)
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words)
        axes[idx].set_xlabel('Normalized Similarity Score')
        axes[idx].set_title(f'{metric.capitalize()} - Nearest Neighbors of "{focal_word}"', fontsize=14)
        axes[idx].invert_yaxis()
        
        # Add frequency information
        for i, word in enumerate(words):
            freq = analyzer.word_freq.get(word, 0)
            axes[idx].text(scores[i] + 0.01, i, f'(f={freq})', 
                          fontsize=8, va='center', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('metric_comparison_neighbors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Create scatter plot showing how metrics differ
    print("\nCreating metric correlation analysis...")
    
    # Sample random words for analysis
    sample_size = min(100, len(analyzer.idx_to_word))
    sample_indices = random.sample(range(len(analyzer.idx_to_word)), sample_size)
    
    # Compute pairwise distances for sampled words using different metrics
    metric_distances = {}
    
    for metric in metrics:
        distances = []
        for i in range(len(sample_indices)):
            for j in range(i+1, len(sample_indices)):
                idx1, idx2 = sample_indices[i], sample_indices[j]
                
                if metric == 'cosine':
                    vec1_norm = analyzer.embeddings[idx1] / np.linalg.norm(analyzer.embeddings[idx1])
                    vec2_norm = analyzer.embeddings[idx2] / np.linalg.norm(analyzer.embeddings[idx2])
                    dist = 1 - np.dot(vec1_norm, vec2_norm)
                elif metric == 'euclidean':
                    dist = np.linalg.norm(analyzer.embeddings[idx1] - analyzer.embeddings[idx2])
                elif metric == 'manhattan':
                    dist = np.sum(np.abs(analyzer.embeddings[idx1] - analyzer.embeddings[idx2]))
                else:  # dot
                    dist = -np.dot(analyzer.embeddings[idx1], analyzer.embeddings[idx2])
                
                distances.append(dist)
        
        metric_distances[metric] = distances
    
    # Create correlation plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics[i+1:], i+1):
            if plot_idx < 6:
                axes[plot_idx].scatter(metric_distances[m1], metric_distances[m2], 
                                      alpha=0.5, s=10)
                axes[plot_idx].set_xlabel(f'{m1.capitalize()} Distance')
                axes[plot_idx].set_ylabel(f'{m2.capitalize()} Distance')
                
                # Compute correlation
                corr, _ = spearmanr(metric_distances[m1], metric_distances[m2])
                axes[plot_idx].set_title(f'Spearman correlation: {corr:.3f}')
                
                plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('metric_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training and analysis pipeline"""
    # Configuration
    config = {
        'text_files': ['HP1.txt', 'HP2.txt', 'HP3.txt', 'HP4.txt', 'HP5.txt', 'HP6.txt', 'HP7.txt'],
        'min_word_freq': 5,
        'max_words_per_file': 100000,
        'embedding_dim': 100,
        'window_size': 5,
        'epochs': 20,
        'batch_size': 512,
        'learning_rate': 0.025,
        'neg_samples': 5
    }
    
    print("Word2Vec Skip-gram Implementation with Metric Analysis")
    print("=" * 80)
    print("Subtopic: How does the metric you use to interpret the embedding space")
    print("          affect your conclusions?")
    print("=" * 80)
    
    # 1. Read and preprocess text
    print("\n1. Reading and preprocessing text from multiple books...")
    
    all_text = ""
    books_loaded = []
    
    for text_file in config['text_files']:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                book_text = f.read()[:config['max_words_per_file']]
                all_text += " " + book_text
                books_loaded.append(text_file)
                print(f"  ✓ Loaded {text_file} ({len(book_text)} characters)")
        except FileNotFoundError:
            print(f"  ✗ Warning: {text_file} not found, continuing with other books...")
    
    if not books_loaded:
        print("Error: No Harry Potter text files found.")
        return None, None, None
    
    text = all_text
    
    # 2. Preprocess text
    preprocessor = TextPreprocessor(min_word_freq=config['min_word_freq'])
    words = preprocessor.preprocess_text(text)
    
    # 3. Build vocabulary
    print("\n2. Building vocabulary...")
    preprocessor.build_vocabulary(words)
    
    # 4. Create dataset
    print("\n3. Creating training dataset...")
    dataset = SkipGramDataset(
        words, 
        preprocessor.word_to_idx, 
        window_size=config['window_size']
    )
    
    # 5. Initialize model
    print("\n4. Initializing model...")
    model = SkipGramModel(
        vocab_size=len(preprocessor.word_to_idx),
        embedding_dim=config['embedding_dim']
    )
    
    # 6. Setup trainer
    trainer = Word2VecTrainer(
        model, 
        vocab_size=len(preprocessor.word_to_idx),
        neg_samples=config['neg_samples']
    )
    trainer.setup_negative_sampling(preprocessor.word_freq, preprocessor.word_to_idx)
    
    # 7. Train model
    print("\n5. Training model...")
    trainer.train(
        dataset,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['learning_rate']
    )
    
    # 8. Metric Analysis
    print("\n6. Performing comprehensive metric analysis...")
    analyzer = MetricAnalyzer(model, preprocessor.idx_to_word, 
                             preprocessor.word_to_idx, preprocessor.word_freq)
    
    # Define test words for analysis
    test_words = ['harry', 'ron', 'hermione', 'wizard', 'magic', 'wand', 
                 'spell', 'dark', 'love', 'friend', 'school', 'professor',
                 'voldemort', 'dumbledore', 'good', 'evil', 'the', 'and', 'a']
    
    # Filter to available words
    test_words = [w for w in test_words if w in preprocessor.word_to_idx]
    
    # Add more common words
    common_words = [w for w, _ in preprocessor.word_freq.most_common(50) 
                   if w in preprocessor.word_to_idx]
    test_words.extend(common_words[:20])
    test_words = list(set(test_words))  # Remove duplicates
    
    # 9. Generate comprehensive analysis
    print("\n7. Generating metric comparison visualizations...")
    
    # Character relationship analysis
    analyze_character_relationships_with_metrics(analyzer)
    
    # Metric comparison visualizations
    create_metric_comparison_visualizations(analyzer, test_words)
    
    # Main comparison visualization
    # analyzer.visualize_metric_comparison(test_words, 'metric_analysis_overview.png')
    

    # 10. Save model and analysis results
    print("\n9. Saving model and analysis results...")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_idx': preprocessor.word_to_idx,
        'idx_to_word': preprocessor.idx_to_word,
        'config': config,
        'loss_history': trainer.loss_history
    }, 'word2vec_model.pth')
    
    # Save metric comparison data
    metric_comparison_data = {
        'test_words': test_words,
        'metric_agreement': analyzer.analyze_metric_agreement(test_words),
        'config': config
    }
    
    with open('metric_comparison_data.json', 'w') as f:
        json.dump(metric_comparison_data, f, indent=2)
    
    print("\nAnalysis complete!")
    print(f"Generated files:")
    print(f"  - word2vec_model_with_metrics.pth")
    print(f"  - metric_comparison_report.txt")
    print(f"  - metric_analysis_overview.png")
    print(f"  - metric_comparison_neighbors.png")
    print(f"  - metric_correlations.png")
    print(f"  - character_relationships_*.png (for each metric)")
    print(f"  - metric_comparison_data.json")
    
    return model, preprocessor, analyzer

if __name__ == "__main__":
    model, preprocessor, analyzer = main()
