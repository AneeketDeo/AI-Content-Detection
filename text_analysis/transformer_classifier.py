"""
Transformer-Based Classifier for AI-Generated Content Detection

This module implements transformer-based classifiers to detect AI-generated content.
It uses pre-trained models specifically fine-tuned for distinguishing between
human-written and AI-generated text.
"""

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Any, Tuple, Optional


class TransformerClassifier:
    """
    Uses transformer-based models to classify text as human-written or AI-generated.
    
    This classifier leverages pre-trained models that have been fine-tuned on datasets
    containing both human-written and AI-generated text samples.
    """
    
    def __init__(self, model_name: str = "roberta-base-openai-detector"):
        """
        Initialize the transformer classifier with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use for classification
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            # Fallback to a more general model if the specific detector isn't available
            print(f"Failed to load {model_name}, falling back to RoBERTa base: {e}")
            self.model_name = "roberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            
    def analyze(self, text: str, chunk_size: int = 512) -> Dict[str, Any]:
        """
        Classify the provided text as human-written or AI-generated.
        
        For long texts, the text is split into chunks and each chunk is classified
        separately, then the results are aggregated.
        
        Args:
            text: The input text to classify
            chunk_size: Maximum number of tokens per chunk for long texts
            
        Returns:
            Dictionary containing classification results and confidence scores
        """
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        # Process the text in chunks if it's too long
        chunks = self._split_into_chunks(text, chunk_size)
        chunk_results = [self._classify_chunk(chunk) for chunk in chunks]
        
        # Aggregate results from all chunks
        aggregated_results = self._aggregate_chunk_results(chunk_results)
        
        # Add detailed analysis
        detailed_analysis = self._generate_detailed_analysis(chunk_results, text)
        
        # Combine all results
        results = {
            "classification": aggregated_results["classification"],
            "confidence": aggregated_results["confidence"],
            "chunk_results": chunk_results,
            "detailed_analysis": detailed_analysis
        }
        
        # Add AI detection indicators
        results["ai_indicators"] = self._extract_ai_indicators(results)
        
        return results
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split long text into chunks that can be processed by the model.
        
        Args:
            text: The input text
            chunk_size: Maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # For very short texts, return as a single chunk
        if len(text) < 100:
            return [text]
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        # If the text fits within the chunk size, return as a single chunk
        if len(tokens) <= chunk_size:
            return [text]
        
        # Split into chunks based on sentences to maintain context
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            test_tokens = self.tokenizer.encode(test_chunk)
            
            if len(test_tokens) <= chunk_size:
                current_chunk = test_chunk
            else:
                # If the current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start a new chunk with this sentence
                # If the sentence itself is too long, truncate it
                sentence_tokens = self.tokenizer.encode(sentence)
                if len(sentence_tokens) > chunk_size:
                    # Truncate the sentence to fit within chunk_size
                    truncated_tokens = sentence_tokens[:chunk_size - 2]  # Leave room for special tokens
                    truncated_sentence = self.tokenizer.decode(truncated_tokens)
                    chunks.append(truncated_sentence)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _classify_chunk(self, chunk: str) -> Dict[str, Any]:
        """
        Classify a single chunk of text.
        
        Args:
            chunk: Text chunk to classify
            
        Returns:
            Dictionary with classification results for this chunk
        """
        # Tokenize the chunk
        inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # Determine the predicted class
            # For most AI detectors, class 1 is AI-generated, class 0 is human
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Map the predicted class to a label
            classification = "ai-generated" if predicted_class == 1 else "human-written"
            
            # For some models, the labels might be reversed, so we check the model name
            if "openai-detector" in self.model_name.lower():
                # For OpenAI detector, class 0 is real (human), class 1 is fake (AI)
                classification = "ai-generated" if predicted_class == 1 else "human-written"
            
            return {
                "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,  # Truncate for readability
                "classification": classification,
                "confidence": float(confidence),
                "human_probability": float(probabilities[0]),
                "ai_probability": float(probabilities[1])
            }
    
    def _aggregate_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple chunks.
        
        Args:
            chunk_results: List of classification results for each chunk
            
        Returns:
            Dictionary with aggregated classification results
        """
        if not chunk_results:
            return {"classification": "unknown", "confidence": 0.0}
        
        # Count the number of chunks classified as AI-generated
        ai_count = sum(1 for result in chunk_results if result["classification"] == "ai-generated")
        human_count = len(chunk_results) - ai_count
        
        # Calculate the average confidence for each class
        ai_confidence = np.mean([result["ai_probability"] for result in chunk_results])
        human_confidence = np.mean([result["human_probability"] for result in chunk_results])
        
        # Determine the overall classification based on majority vote
        if ai_count > human_count:
            classification = "ai-generated"
            confidence = ai_confidence
        else:
            classification = "human-written"
            confidence = human_confidence
        
        # If the confidence is very close to 0.5, mark as uncertain
        if 0.45 <= confidence <= 0.55:
            classification = "uncertain"
        
        return {
            "classification": classification,
            "confidence": float(confidence),
            "ai_chunks_ratio": ai_count / len(chunk_results),
            "human_chunks_ratio": human_count / len(chunk_results)
        }
    
    def _generate_detailed_analysis(self, chunk_results: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """
        Generate a detailed analysis of the classification results.
        
        Args:
            chunk_results: List of classification results for each chunk
            text: The original input text
            
        Returns:
            Dictionary with detailed analysis
        """
        # Calculate consistency of classifications across chunks
        classifications = [result["classification"] for result in chunk_results]
        unique_classifications = set(classifications)
        consistency = 1.0 if len(unique_classifications) == 1 else len(unique_classifications) / len(chunk_results)
        
        # Find the chunks with highest AI probability
        sorted_by_ai_prob = sorted(chunk_results, key=lambda x: x["ai_probability"], reverse=True)
        most_ai_like_chunks = sorted_by_ai_prob[:min(3, len(sorted_by_ai_prob))]
        
        # Find the chunks with highest human probability
        sorted_by_human_prob = sorted(chunk_results, key=lambda x: x["human_probability"], reverse=True)
        most_human_like_chunks = sorted_by_human_prob[:min(3, len(sorted_by_human_prob))]
        
        return {
            "classification_consistency": 1.0 - consistency,  # Higher means more consistent
            "most_ai_like_chunks": most_ai_like_chunks,
            "most_human_like_chunks": most_human_like_chunks,
            "chunk_count": len(chunk_results),
            "text_length": len(text)
        }
    
    def _extract_ai_indicators(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract indicators that might suggest AI-generated content based on transformer classification.
        
        Args:
            results: Dictionary with classification results
            
        Returns:
            Dictionary with AI indicators
        """
        indicators = {}
        
        # Overall classification
        classification = results["classification"]
        confidence = results["confidence"]
        
        indicators["transformer_classification"] = {
            "value": classification,
            "confidence": confidence,
            "is_suspicious": classification == "ai-generated" and confidence > 0.7,
            "explanation": "Transformer model directly classified this as AI-generated with high confidence"
        }
        
        # Classification consistency across chunks
        consistency = results["detailed_analysis"]["classification_consistency"]
        indicators["classification_consistency"] = {
            "value": consistency,
            "is_suspicious": consistency > 0.8,  # High consistency can be suspicious
            "explanation": "AI-generated text often has consistent classification across different sections"
        }
        
        # Ratio of AI-classified chunks
        if "ai_chunks_ratio" in results:
            ai_ratio = results["ai_chunks_ratio"]
            indicators["ai_chunks_ratio"] = {
                "value": ai_ratio,
                "is_suspicious": ai_ratio > 0.7,
                "explanation": "High percentage of text chunks classified as AI-generated"
            }
        
        # Overall suspicion score (weighted average of binary indicators)
        weights = {"transformer_classification": 0.6, "classification_consistency": 0.2, "ai_chunks_ratio": 0.2}
        suspicion_score = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in indicators:
                suspicion_score += indicators[key]["is_suspicious"] * weight
                total_weight += weight
        
        if total_weight > 0:
            indicators["overall_suspicion_score"] = suspicion_score / total_weight
        else:
            indicators["overall_suspicion_score"] = 0.0
        
        return indicators


# Example usage
if __name__ == "__main__":
    classifier = TransformerClassifier()
    sample_text = """
    This is a sample text to demonstrate transformer-based classification. 
    The classifier will determine if this text was written by a human or generated by AI.
    Transformer models have been fine-tuned on datasets containing both human and AI text.
    These models can identify subtle patterns that distinguish AI-generated content.
    """
    results = classifier.analyze(sample_text)
    print(f"Classification: {results['classification']} (confidence: {results['confidence']:.2f})")
    print(f"AI indicators: {results['ai_indicators']}")
