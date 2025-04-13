"""
Stylometric Analysis Module for AI-Generated Content Detection

This module implements stylometric analysis techniques to detect patterns
in writing style that may indicate AI-generated content.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class StylometricAnalyzer:
    """
    Analyzes text using stylometric features to detect AI-generated content.
    
    Stylometric analysis examines writing style characteristics such as:
    - Sentence length distribution
    - Vocabulary richness
    - Function word usage
    - Punctuation patterns
    - Readability metrics
    """
    
    def __init__(self):
        """Initialize the stylometric analyzer with required resources."""
        self.stop_words = set(stopwords.words('english'))
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive stylometric analysis on the provided text.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary containing various stylometric features and metrics
        """
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        # Basic text statistics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        words_lower = [word.lower() for word in words if word.isalpha()]
        
        # Calculate features
        sentence_lengths = self._get_sentence_lengths(sentences)
        vocabulary_richness = self._calculate_vocabulary_richness(words_lower)
        function_word_stats = self._analyze_function_words(words_lower)
        punctuation_stats = self._analyze_punctuation(text)
        readability_metrics = self._calculate_readability(text, sentences, words)
        
        # Combine all features
        features = {
            "basic_stats": {
                "num_sentences": len(sentences),
                "num_words": len(words_lower),
                "avg_word_length": np.mean([len(word) for word in words_lower]) if words_lower else 0,
            },
            "sentence_structure": sentence_lengths,
            "vocabulary_richness": vocabulary_richness,
            "function_word_usage": function_word_stats,
            "punctuation_patterns": punctuation_stats,
            "readability": readability_metrics
        }
        
        # Add AI detection indicators
        features["ai_indicators"] = self._extract_ai_indicators(features)
        
        return features
    
    def _get_sentence_lengths(self, sentences: List[str]) -> Dict[str, Any]:
        """Calculate statistics about sentence lengths."""
        if not sentences:
            return {"avg_length": 0, "std_dev": 0, "distribution": {}}
        
        lengths = [len(word_tokenize(s)) for s in sentences]
        
        # Create distribution buckets
        distribution = Counter()
        for length in lengths:
            bucket = (length // 5) * 5  # Group by 5s (0-4, 5-9, etc.)
            distribution[f"{bucket}-{bucket+4}"] += 1
        
        return {
            "avg_length": np.mean(lengths),
            "std_dev": np.std(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "distribution": dict(distribution)
        }
    
    def _calculate_vocabulary_richness(self, words: List[str]) -> Dict[str, float]:
        """Calculate vocabulary richness metrics."""
        if not words:
            return {"ttr": 0, "hapax_legomena_ratio": 0}
        
        # Type-Token Ratio (TTR)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # Hapax Legomena (words that appear exactly once)
        word_counts = Counter(words)
        hapax_legomena = sum(1 for word, count in word_counts.items() if count == 1)
        hapax_ratio = hapax_legomena / len(words)
        
        return {
            "ttr": ttr,
            "hapax_legomena_ratio": hapax_ratio,
            "unique_words_count": len(unique_words)
        }
    
    def _analyze_function_words(self, words: List[str]) -> Dict[str, Any]:
        """Analyze the usage of function words (stopwords)."""
        if not words:
            return {"function_word_ratio": 0, "top_function_words": {}}
        
        # Calculate ratio of function words to total words
        function_words = [word for word in words if word in self.stop_words]
        function_word_ratio = len(function_words) / len(words)
        
        # Get most common function words
        function_word_counts = Counter(function_words)
        top_function_words = dict(function_word_counts.most_common(10))
        
        return {
            "function_word_ratio": function_word_ratio,
            "top_function_words": top_function_words
        }
    
    def _analyze_punctuation(self, text: str) -> Dict[str, Any]:
        """Analyze punctuation patterns in the text."""
        if not text:
            return {"punctuation_ratio": 0, "punctuation_counts": {}}
        
        # Count punctuation marks
        punctuation_marks = re.findall(r'[.,!?;:"\'\(\)\[\]\{\}]', text)
        punctuation_ratio = len(punctuation_marks) / len(text)
        
        # Count specific punctuation types
        punctuation_counts = Counter(punctuation_marks)
        
        return {
            "punctuation_ratio": punctuation_ratio,
            "punctuation_counts": dict(punctuation_counts)
        }
    
    def _calculate_readability(self, text: str, sentences: List[str], words: List[str]) -> Dict[str, float]:
        """Calculate readability metrics."""
        if not text or not sentences or not words:
            return {"flesch_reading_ease": 0, "automated_readability_index": 0}
        
        # Count syllables (simplified approach)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count = 1
            return count
        
        syllable_count = sum(count_syllables(word) for word in words if word.isalpha())
        
        # Flesch Reading Ease
        if len(sentences) == 0 or len(words) == 0:
            flesch = 0
        else:
            flesch = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
        
        # Automated Readability Index
        char_count = len(text) - text.count(" ")
        if len(sentences) == 0 or len(words) == 0:
            ari = 0
        else:
            ari = 4.71 * (char_count / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43
        
        return {
            "flesch_reading_ease": flesch,
            "automated_readability_index": ari,
            "syllables_per_word": syllable_count / len(words) if words else 0
        }
    
    def _extract_ai_indicators(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract indicators that might suggest AI-generated content.
        
        These are heuristics based on common patterns observed in AI text:
        - Very consistent sentence lengths (low std dev)
        - Unusually high or low vocabulary richness
        - Abnormal function word usage
        - Overly consistent readability
        """
        indicators = {}
        
        # Sentence length consistency (AI often has more uniform sentence lengths)
        sent_std_dev = features["sentence_structure"]["std_dev"]
        indicators["sentence_length_consistency"] = {
            "value": sent_std_dev,
            "is_suspicious": sent_std_dev < 2.0,  # Threshold determined empirically
            "explanation": "AI-generated text often has very consistent sentence lengths"
        }
        
        # Vocabulary richness (can be unnaturally high in some AI models)
        ttr = features["vocabulary_richness"]["ttr"]
        indicators["vocabulary_richness"] = {
            "value": ttr,
            "is_suspicious": ttr > 0.8 or ttr < 0.3,  # Suspicious if too high or too low
            "explanation": "AI text may have abnormally high or low vocabulary diversity"
        }
        
        # Function word usage
        func_word_ratio = features["function_word_usage"]["function_word_ratio"]
        indicators["function_word_usage"] = {
            "value": func_word_ratio,
            "is_suspicious": func_word_ratio < 0.25 or func_word_ratio > 0.6,
            "explanation": "AI text may use function words in unusual proportions"
        }
        
        # Overall suspicion score (simple average of binary indicators)
        suspicious_count = sum(1 for ind in indicators.values() if ind["is_suspicious"])
        indicators["overall_suspicion_score"] = suspicious_count / len(indicators)
        
        return indicators


# Example usage
if __name__ == "__main__":
    analyzer = StylometricAnalyzer()
    sample_text = """
    This is a sample text to demonstrate stylometric analysis. 
    The analyzer will extract various features from this text.
    These features can help determine if the text was written by a human or generated by AI.
    Stylometric analysis looks at patterns in writing style, vocabulary usage, and sentence structure.
    """
    results = analyzer.analyze(sample_text)
    print(results)
