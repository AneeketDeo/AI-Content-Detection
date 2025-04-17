"""
Text Analysis Module for AI-Generated Content Detection

This module integrates multiple analysis techniques to detect AI-generated text content:
1. Stylometric Analysis - examines writing style characteristics
2. Perplexity Analysis - measures text predictability using language models
3. Transformer-based Classification - directly classifies text as human or AI-generated

The module provides a unified interface for analyzing text and determining
whether it was likely written by a human or generated by AI.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
import os
import torch
import gc

# Import individual analyzers
from .stylometric import StylometricAnalyzer
from .perplexity import PerplexityAnalyzer
from .transformer_classifier import TransformerClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import streamlit as st

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Set environment variable to use local cache
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'models')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'models')

# Disable PyTorch's custom class registration
torch._C._disable_custom_class_registration()

token = st.secrets["HF_TOKEN"]


class TextAnalysisModule:
    """
    Unified module for detecting AI-generated text content using multiple techniques.
    
    This module combines stylometric analysis, perplexity scoring, and transformer-based
    classification to provide comprehensive detection of AI-generated text.
    """
    
    def __init__(self, use_stylometric: bool = True, use_perplexity: bool = True, 
                 use_transformer: bool = True, perplexity_model: str = "distilgpt2",
                 transformer_model: str = "distilroberta-base"):
        """
        Initialize the text analysis module with selected analyzers.
        
        Args:
            use_stylometric: Whether to use stylometric analysis
            use_perplexity: Whether to use perplexity analysis
            use_transformer: Whether to use transformer-based classification
            perplexity_model: Model name for perplexity analyzer
            transformer_model: Model name for transformer classifier
        """
        self.analyzers = {}
        self.use_stylometric = use_stylometric
        self.use_perplexity = use_perplexity
        self.use_transformer = use_transformer
        self.perplexity_model = perplexity_model
        self.transformer_model = transformer_model
        
        # Initialize stylometric analyzer immediately as it doesn't use PyTorch
        if use_stylometric:
            logger.info("Initializing stylometric analyzer")
            self.analyzers['stylometric'] = StylometricAnalyzer()
    
    def _initialize_perplexity_analyzer(self):
        """Initialize the perplexity analyzer on demand."""
        if self.use_perplexity and 'perplexity' not in self.analyzers:
            logger.info(f"Initializing perplexity analyzer with model: {self.perplexity_model}")
            try:
                model_path = os.path.join('models', self.perplexity_model)
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                self.analyzers['perplexity'] = PerplexityAnalyzer(model_name=self.perplexity_model)
            except Exception as e:
                logger.warning(f"Failed to initialize perplexity analyzer: {e}")
                self.analyzers['perplexity'] = None
    
    def _initialize_transformer_classifier(self):
        """Initialize the transformer classifier on demand."""
        if self.use_transformer and 'transformer' not in self.analyzers:
            logger.info(f"Initializing transformer classifier with model: {self.transformer_model}")
            try:
                model_path = os.path.join('models', self.transformer_model)
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                self.analyzers['transformer'] = TransformerClassifier(model_name=self.transformer_model)
            except Exception as e:
                logger.warning(f"Failed to initialize transformer classifier: {e}")
                self.analyzers['transformer'] = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using all available analyzers and combine results.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary containing combined analysis results and final determination
        """
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        logger.info(f"Analyzing text of length {len(text)}")
        
        # Run each analyzer and collect results
        results = {}
        
        # Run stylometric analysis if available
        if 'stylometric' in self.analyzers:
            try:
                logger.info("Running stylometric analysis")
                results['stylometric'] = self.analyzers['stylometric'].analyze(text)
            except Exception as e:
                logger.error(f"Error in stylometric analysis: {e}")
                results['stylometric'] = {"error": str(e)}
        
        # Initialize and run perplexity analysis
        if self.use_perplexity:
            try:
                self._initialize_perplexity_analyzer()
                if 'perplexity' in self.analyzers and self.analyzers['perplexity'] is not None:
                    logger.info("Running perplexity analysis")
                    results['perplexity'] = self.analyzers['perplexity'].analyze(text)
            except Exception as e:
                logger.error(f"Error in perplexity analysis: {e}")
                results['perplexity'] = {"error": str(e)}
            finally:
                # Clean up perplexity analyzer
                if 'perplexity' in self.analyzers:
                    del self.analyzers['perplexity']
                gc.collect()
                torch.cuda.empty_cache()
        
        # Initialize and run transformer analysis
        if self.use_transformer:
            try:
                self._initialize_transformer_classifier()
                if 'transformer' in self.analyzers and self.analyzers['transformer'] is not None:
                    logger.info("Running transformer analysis")
                    results['transformer'] = self.analyzers['transformer'].analyze(text)
            except Exception as e:
                logger.error(f"Error in transformer analysis: {e}")
                results['transformer'] = {"error": str(e)}
            finally:
                # Clean up transformer classifier
                if 'transformer' in self.analyzers:
                    del self.analyzers['transformer']
                gc.collect()
                torch.cuda.empty_cache()
        
        # Combine results and make final determination
        combined_result = self._combine_results(results)
        
        return combined_result
    
    def _combine_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple analyzers and make a final determination.
        
        Args:
            results: Dictionary of results from each analyzer
            
        Returns:
            Dictionary with combined results and final determination
        """
        # Extract AI indicators from each analyzer
        ai_indicators = {}
        for analyzer_name, analyzer_results in results.items():
            if "ai_indicators" in analyzer_results:
                ai_indicators[analyzer_name] = analyzer_results["ai_indicators"]
        
        # Calculate weighted scores for final determination
        weights = {
            "transformer": 0.5,  # Transformer-based classification has highest weight
            "perplexity": 0.3,   # Perplexity analysis has medium weight
            "stylometric": 0.2   # Stylometric analysis has lowest weight
        }
        
        # Calculate weighted suspicion score
        weighted_score = 0.0
        total_weight = 0.0
        
        for analyzer_name, indicators in ai_indicators.items():
            if "overall_suspicion_score" in indicators and analyzer_name in weights:
                score = indicators["overall_suspicion_score"]
                weight = weights[analyzer_name]
                weighted_score += score * weight
                total_weight += weight
        
        # Normalize the weighted score
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0
        
        # Determine the final classification based on the score
        if final_score >= 0.7:
            classification = "ai-generated"
            confidence = final_score
        elif final_score <= 0.3:
            classification = "human-written"
            confidence = 1.0 - final_score
        else:
            classification = "uncertain"
            confidence = 0.5
        
        # Generate explanation for the determination
        explanation = self._generate_explanation(results, classification, final_score)
        
        # Combine all results
        combined_result = {
            "classification": classification,
            "confidence": confidence,
            "suspicion_score": final_score,
            "explanation": explanation,
            "detailed_results": results,
            "ai_indicators": ai_indicators
        }
        
        return combined_result
    
    def _generate_explanation(self, results: Dict[str, Dict[str, Any]], 
                             classification: str, score: float) -> str:
        """
        Generate a human-readable explanation for the final determination.
        
        Args:
            results: Dictionary of results from each analyzer
            classification: Final classification (ai-generated, human-written, uncertain)
            score: Final suspicion score
            
        Returns:
            String with explanation
        """
        explanation_parts = []
        
        # Add classification-specific introduction
        if classification == "ai-generated":
            explanation_parts.append(
                f"This text is likely AI-generated (confidence: {score:.2f}). "
                f"The analysis detected several indicators of AI-generated content:"
            )
        elif classification == "human-written":
            explanation_parts.append(
                f"This text is likely human-written (confidence: {1-score:.2f}). "
                f"The analysis found few or no indicators of AI-generated content:"
            )
        else:
            explanation_parts.append(
                f"The analysis is uncertain about whether this text is AI-generated or human-written. "
                f"Some indicators were detected, but they are not conclusive:"
            )
        
        # Add transformer classifier results if available
        if "transformer" in results and "classification" in results["transformer"]:
            transformer_result = results["transformer"]
            explanation_parts.append(
                f"- Transformer classifier: {transformer_result['classification']} "
                f"(confidence: {transformer_result['confidence']:.2f})"
            )
        
        # Add perplexity results if available
        if "perplexity" in results and "overall_perplexity" in results["perplexity"]:
            perplexity_result = results["perplexity"]
            explanation_parts.append(
                f"- Perplexity analysis: score {perplexity_result['overall_perplexity']:.2f} "
                f"({'lower' if perplexity_result['overall_perplexity'] < 60 else 'higher'} "
                f"perplexity {'suggests AI generation' if perplexity_result['overall_perplexity'] < 60 else 'is typical of human writing'})"
            )
        
        # Add stylometric results if available
        if "stylometric" in results and "ai_indicators" in results["stylometric"]:
            stylometric_indicators = results["stylometric"]["ai_indicators"]
            suspicious_indicators = [
                k for k, v in stylometric_indicators.items() 
                if isinstance(v, dict) and v.get("is_suspicious", False)
            ]
            
            if suspicious_indicators and len(suspicious_indicators) > 0:
                explanation_parts.append(
                    f"- Stylometric analysis: detected {len(suspicious_indicators)} suspicious patterns "
                    f"({', '.join(suspicious_indicators)})"
                )
            else:
                explanation_parts.append(
                    f"- Stylometric analysis: no suspicious patterns detected"
                )
        
        # Add conclusion
        if classification == "ai-generated":
            explanation_parts.append(
                "Based on these indicators, this text exhibits characteristics "
                "commonly found in AI-generated content."
            )
        elif classification == "human-written":
            explanation_parts.append(
                "Based on these indicators, this text exhibits characteristics "
                "more typical of human-written content."
            )
        else:
            explanation_parts.append(
                "The mixed indicators make it difficult to conclusively determine "
                "whether this text is AI-generated or human-written."
            )
        
        return "\n".join(explanation_parts)


# Example usage
if __name__ == "__main__":
    analyzer = TextAnalysisModule(
        use_stylometric=True,
        use_perplexity=True,
        use_transformer=True
    )
    
    sample_text = """
    This is a sample text to demonstrate the text analysis module.
    It combines multiple techniques to detect AI-generated content.
    The module will analyze this text and determine if it was written by a human or generated by AI.
    """
    
    results = analyzer.analyze(sample_text)
    print(f"Classification: {results['classification']} (confidence: {results['confidence']:.2f})")
    print(f"Explanation: {results['explanation']}")
