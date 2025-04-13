"""
Image Analysis Module for AI-Generated Content Detection

This module integrates multiple analysis techniques to detect AI-generated images:
1. Frequency Domain Analysis - examines artifacts and patterns in the frequency domain
2. Texture and Noise Analysis - analyzes noise patterns and texture consistency
3. Deep Learning Classification - uses pre-trained models to classify images

The module provides a unified interface for analyzing images and determining
whether they were likely created by AI or are real photographs.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# Import individual analyzers
from .frequency_domain import FrequencyDomainAnalyzer
from .texture_noise import TextureNoiseAnalyzer
from .deep_learning import DeepLearningClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageAnalysisModule:
    """
    Unified module for detecting AI-generated image content using multiple techniques.
    
    This module combines frequency domain analysis, texture and noise analysis, and
    deep learning classification to provide comprehensive detection of AI-generated images.
    """
    
    def __init__(self, use_frequency: bool = True, use_texture: bool = True, 
                 use_deep_learning: bool = True, deep_learning_model: str = "resnet"):
        """
        Initialize the image analysis module with selected analyzers.
        
        Args:
            use_frequency: Whether to use frequency domain analysis
            use_texture: Whether to use texture and noise analysis
            use_deep_learning: Whether to use deep learning classification
            deep_learning_model: Model type for deep learning classifier
        """
        self.analyzers = {}
        
        # Initialize selected analyzers
        if use_frequency:
            logger.info("Initializing frequency domain analyzer")
            self.analyzers['frequency'] = FrequencyDomainAnalyzer()
        
        if use_texture:
            logger.info("Initializing texture and noise analyzer")
            self.analyzers['texture'] = TextureNoiseAnalyzer()
        
        if use_deep_learning:
            logger.info(f"Initializing deep learning classifier with model: {deep_learning_model}")
            try:
                self.analyzers['deep_learning'] = DeepLearningClassifier(model_type=deep_learning_model)
            except Exception as e:
                logger.warning(f"Failed to initialize deep learning classifier: {e}")
                self.analyzers['deep_learning'] = None
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image using all available analyzers and combine results.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Dictionary containing combined analysis results and final determination
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        logger.info(f"Analyzing image: {image_path}")
        
        # Run each analyzer and collect results
        results = {}
        for name, analyzer in self.analyzers.items():
            if analyzer is not None:
                try:
                    logger.info(f"Running {name} analysis")
                    results[name] = analyzer.analyze(image_path)
                except Exception as e:
                    logger.error(f"Error in {name} analysis: {e}")
                    results[name] = {"error": str(e)}
        
        # Combine results and make final determination
        combined_result = self._combine_results(results)
        
        # Add image metadata
        combined_result["metadata"] = self._extract_image_metadata(image_path)
        
        return combined_result
    
    def _extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image metadata
        """
        try:
            image = cv2.imread(image_path)
            height, width, channels = image.shape
            file_size = os.path.getsize(image_path)
            file_extension = os.path.splitext(image_path)[1].lower()
            
            return {
                "file_name": os.path.basename(image_path),
                "file_size": file_size,
                "file_extension": file_extension,
                "dimensions": (width, height),
                "channels": channels
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "file_name": os.path.basename(image_path),
                "error": str(e)
            }
    
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
            "deep_learning": 0.5,  # Deep learning classification has highest weight
            "frequency": 0.3,      # Frequency domain analysis has medium weight
            "texture": 0.2         # Texture analysis has lowest weight
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
            classification = "real-photo"
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
            classification: Final classification (ai-generated, real-photo, uncertain)
            score: Final suspicion score
            
        Returns:
            String with explanation
        """
        explanation_parts = []
        
        # Add classification-specific introduction
        if classification == "ai-generated":
            explanation_parts.append(
                f"This image is likely AI-generated (confidence: {score:.2f}). "
                f"The analysis detected several indicators of AI-generated content:"
            )
        elif classification == "real-photo":
            explanation_parts.append(
                f"This image is likely a real photograph (confidence: {1-score:.2f}). "
                f"The analysis found few or no indicators of AI-generated content:"
            )
        else:
            explanation_parts.append(
                f"The analysis is uncertain about whether this image is AI-generated or real. "
                f"Some indicators were detected, but they are not conclusive:"
            )
        
        # Add deep learning classifier results if available
        if "deep_learning" in results and "classification" in results["deep_learning"]:
            deep_learning_result = results["deep_learning"]
            explanation_parts.append(
                f"- Deep learning classifier: {deep_learning_result['classification']} "
                f"(confidence: {deep_learning_result['confidence']:.2f})"
            )
        
        # Add frequency domain results if available
        if "frequency" in results and "ai_indicators" in results["frequency"]:
            frequency_indicators = results["frequency"]["ai_indicators"]
            suspicious_indicators = [
                k for k, v in frequency_indicators.items() 
                if isinstance(v, dict) and v.get("is_suspicious", False)
            ]
            
            if suspicious_indicators and len(suspicious_indicators) > 0:
                explanation_parts.append(
                    f"- Frequency domain analysis: detected {len(suspicious_indicators)} suspicious patterns "
                    f"({', '.join(suspicious_indicators)})"
                )
            else:
                explanation_parts.append(
                    f"- Frequency domain analysis: no suspicious patterns detected"
                )
        
        # Add texture and noise results if available
        if "texture" in results and "ai_indicators" in results["texture"]:
            texture_indicators = results["texture"]["ai_indicators"]
            suspicious_indicators = [
                k for k, v in texture_indicators.items() 
                if isinstance(v, dict) and v.get("is_suspicious", False)
            ]
            
            if suspicious_indicators and len(suspicious_indicators) > 0:
                explanation_parts.append(
                    f"- Texture and noise analysis: detected {len(suspicious_indicators)} suspicious patterns "
                    f"({', '.join(suspicious_indicators)})"
                )
            else:
                explanation_parts.append(
                    f"- Texture and noise analysis: no suspicious patterns detected"
                )
        
        # Add conclusion
        if classification == "ai-generated":
            explanation_parts.append(
                "Based on these indicators, this image exhibits characteristics "
                "commonly found in AI-generated content."
            )
        elif classification == "real-photo":
            explanation_parts.append(
                "Based on these indicators, this image exhibits characteristics "
                "more typical of real photographs."
            )
        else:
            explanation_parts.append(
                "The mixed indicators make it difficult to conclusively determine "
                "whether this image is AI-generated or a real photograph."
            )
        
        return "\n".join(explanation_parts)


# Example usage
if __name__ == "__main__":
    analyzer = ImageAnalysisModule(
        use_frequency=True,
        use_texture=True,
        use_deep_learning=True
    )
    
    sample_image_path = "sample_image.jpg"
    if os.path.exists(sample_image_path):
        results = analyzer.analyze(sample_image_path)
        print(f"Classification: {results['classification']} (confidence: {results['confidence']:.2f})")
        print(f"Explanation: {results['explanation']}")
