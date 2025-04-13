"""
Perplexity Analysis Module for AI-Generated Content Detection

This module implements perplexity-based techniques to detect AI-generated content.
Perplexity measures how well a probability model predicts a sample, with lower
perplexity indicating the model finds the text more predictable.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Tuple, Optional


class PerplexityAnalyzer:
    """
    Analyzes text using perplexity scores to detect AI-generated content.
    
    Perplexity is a measurement of how well a probability model predicts a sample.
    Lower perplexity indicates the model finds the text more predictable.
    AI-generated text often has different perplexity patterns compared to human text.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the perplexity analyzer with a language model.
        
        Args:
            model_name: The name of the pretrained model to use for perplexity calculation
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def analyze(self, text: str, stride: int = 512) -> Dict[str, Any]:
        """
        Calculate perplexity and related metrics for the provided text.
        
        Args:
            text: The input text to analyze
            stride: Stride length for processing long texts
            
        Returns:
            Dictionary containing perplexity scores and related metrics
        """
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        # Calculate overall perplexity
        perplexity, log_probs = self._calculate_perplexity(text, stride)
        
        # Calculate sliding window perplexities to detect inconsistencies
        window_perplexities = self._calculate_window_perplexities(text)
        
        # Calculate predictability metrics
        predictability_metrics = self._calculate_predictability_metrics(log_probs)
        
        # Combine all metrics
        results = {
            "overall_perplexity": perplexity,
            "window_perplexities": window_perplexities,
            "predictability_metrics": predictability_metrics
        }
        
        # Add AI detection indicators
        results["ai_indicators"] = self._extract_ai_indicators(results)
        
        return results
    
    def _calculate_perplexity(self, text: str, stride: int = 512) -> Tuple[float, List[float]]:
        """
        Calculate the perplexity of the text using the loaded language model.
        
        For long texts, we use a sliding window approach with the given stride.
        
        Args:
            text: The input text
            stride: Stride length for processing long texts
            
        Returns:
            Tuple of (perplexity_score, token_log_probs)
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        # For short texts that fit in the model's context window
        if input_ids.size(1) <= self.tokenizer.model_max_length:
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss.item()
                
            # Calculate perplexity
            perplexity = np.exp(neg_log_likelihood)
            
            # Get token-level log probabilities
            log_probs = []
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
                
                # Calculate log probabilities for each token
                for i in range(input_ids.size(1) - 1):
                    next_token_logits = logits[0, i, :]
                    next_token_id = input_ids[0, i + 1]
                    log_prob = -torch.nn.functional.cross_entropy(
                        next_token_logits.unsqueeze(0), 
                        next_token_id.unsqueeze(0)
                    ).item()
                    log_probs.append(log_prob)
            
            return perplexity, log_probs
        
        # For longer texts, use a sliding window approach
        else:
            max_length = self.tokenizer.model_max_length
            nlls = []
            log_probs = []
            
            # Process the text in chunks with overlap
            for i in range(0, input_ids.size(1), stride):
                end_idx = min(i + max_length, input_ids.size(1))
                chunk_input_ids = input_ids[:, i:end_idx]
                
                # Skip chunks that are too small
                if chunk_input_ids.size(1) < 2:
                    continue
                
                with torch.no_grad():
                    # Calculate loss for this chunk
                    outputs = self.model(chunk_input_ids, labels=chunk_input_ids)
                    neg_log_likelihood = outputs.loss.item()
                    nlls.append(neg_log_likelihood)
                    
                    # Calculate log probabilities for each token in this chunk
                    outputs = self.model(chunk_input_ids)
                    logits = outputs.logits
                    
                    for j in range(chunk_input_ids.size(1) - 1):
                        next_token_logits = logits[0, j, :]
                        next_token_id = chunk_input_ids[0, j + 1]
                        log_prob = -torch.nn.functional.cross_entropy(
                            next_token_logits.unsqueeze(0), 
                            next_token_id.unsqueeze(0)
                        ).item()
                        log_probs.append(log_prob)
            
            # Calculate average perplexity across all chunks
            avg_nll = np.mean(nlls)
            perplexity = np.exp(avg_nll)
            
            return perplexity, log_probs
    
    def _calculate_window_perplexities(self, text: str, window_size: int = 100, step: int = 50) -> Dict[str, Any]:
        """
        Calculate perplexity in sliding windows to detect inconsistencies.
        
        Args:
            text: The input text
            window_size: Size of each window in characters
            step: Step size for sliding the window
            
        Returns:
            Dictionary with window perplexity statistics
        """
        if len(text) < window_size:
            return {
                "windows": [{"start": 0, "end": len(text), "perplexity": self._calculate_perplexity(text)[0]}],
                "mean": 0,
                "std_dev": 0,
                "min": 0,
                "max": 0
            }
        
        # Calculate perplexity for each window
        windows = []
        for i in range(0, len(text) - window_size + 1, step):
            window_text = text[i:i+window_size]
            if len(window_text.strip()) > 0:
                perplexity = self._calculate_perplexity(window_text)[0]
                windows.append({
                    "start": i,
                    "end": i + window_size,
                    "perplexity": perplexity
                })
        
        # Calculate statistics
        perplexities = [w["perplexity"] for w in windows]
        
        return {
            "windows": windows,
            "mean": np.mean(perplexities) if perplexities else 0,
            "std_dev": np.std(perplexities) if perplexities else 0,
            "min": min(perplexities) if perplexities else 0,
            "max": max(perplexities) if perplexities else 0
        }
    
    def _calculate_predictability_metrics(self, log_probs: List[float]) -> Dict[str, float]:
        """
        Calculate metrics related to text predictability from token log probabilities.
        
        Args:
            log_probs: List of log probabilities for each token
            
        Returns:
            Dictionary with predictability metrics
        """
        if not log_probs:
            return {
                "mean_log_prob": 0,
                "std_dev_log_prob": 0,
                "predictability_score": 0
            }
        
        # Calculate statistics of log probabilities
        mean_log_prob = np.mean(log_probs)
        std_dev_log_prob = np.std(log_probs)
        
        # Calculate a predictability score (higher means more predictable)
        predictability_score = -mean_log_prob  # Negative because log probs are negative
        
        return {
            "mean_log_prob": mean_log_prob,
            "std_dev_log_prob": std_dev_log_prob,
            "predictability_score": predictability_score
        }
    
    def _extract_ai_indicators(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract indicators that might suggest AI-generated content based on perplexity.
        
        Args:
            results: Dictionary with perplexity analysis results
            
        Returns:
            Dictionary with AI indicators
        """
        indicators = {}
        
        # Overall perplexity (AI text often has lower perplexity with certain models)
        perplexity = results["overall_perplexity"]
        indicators["overall_perplexity"] = {
            "value": perplexity,
            "is_suspicious": perplexity < 30.0,  # Threshold determined empirically for GPT-2
            "explanation": "AI-generated text often has lower perplexity (is more predictable)"
        }
        
        # Perplexity consistency across windows
        window_std_dev = results["window_perplexities"]["std_dev"]
        indicators["perplexity_consistency"] = {
            "value": window_std_dev,
            "is_suspicious": window_std_dev < 5.0,  # Threshold determined empirically
            "explanation": "AI-generated text often has consistent perplexity across different sections"
        }
        
        # Log probability standard deviation
        log_prob_std_dev = results["predictability_metrics"]["std_dev_log_prob"]
        indicators["log_prob_variation"] = {
            "value": log_prob_std_dev,
            "is_suspicious": log_prob_std_dev < 2.0,  # Threshold determined empirically
            "explanation": "AI-generated text often has less variation in token predictability"
        }
        
        # Overall suspicion score (simple average of binary indicators)
        suspicious_count = sum(1 for ind in indicators.values() if ind["is_suspicious"])
        indicators["overall_suspicion_score"] = suspicious_count / len(indicators)
        
        return indicators


# Example usage
if __name__ == "__main__":
    analyzer = PerplexityAnalyzer()
    sample_text = """
    This is a sample text to demonstrate perplexity analysis. 
    The analyzer will calculate how predictable this text is according to a language model.
    Lower perplexity scores indicate the text is more predictable.
    AI-generated text often has different perplexity patterns compared to human-written text.
    """
    results = analyzer.analyze(sample_text)
    print(f"Overall perplexity: {results['overall_perplexity']}")
    print(f"AI indicators: {results['ai_indicators']}")
