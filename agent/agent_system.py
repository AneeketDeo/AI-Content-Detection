"""
Agent System for AI-Generated Content Detection

This module implements the agentic behavior for the AI content detection application.
It autonomously decides how to process inputs, routes them to appropriate analyzers,
and handles decision-making for uncertain classifications.
"""

import os
import mimetypes
import tempfile
import logging
import requests
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import base64
import re

# Import analysis modules
from text_analysis.text_analyzer import TextAnalysisModule
from image_analysis.image_analyzer import ImageAnalysisModule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentDetectionAgent:
    """
    Agent system for AI-generated content detection.
    
    This agent autonomously decides how to process inputs, routes them to
    appropriate analyzers, and handles decision-making for uncertain cases.
    """
    
    def __init__(self):
        """Initialize the agent system with analysis modules."""
        # Initialize analysis modules
        self.text_analyzer = TextAnalysisModule(
            use_stylometric=True,
            use_perplexity=True,
            use_transformer=True
        )
        
        self.image_analyzer = ImageAnalysisModule(
            use_frequency=True,
            use_texture=True,
            use_deep_learning=True
        )
        
        # Initialize external API keys (if needed)
        self.api_keys = {
            # Add API keys for external services if needed
        }
        
        # Set confidence thresholds
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.4
        
        logger.info("Content Detection Agent initialized")
    
    def process_input(self, input_data: Any, input_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process input data and route to appropriate analyzers.
        
        Args:
            input_data: The input data (text string, file path, or file-like object)
            input_type: Optional hint about input type ('text', 'image', or None for auto-detection)
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        logger.info(f"Processing input with type hint: {input_type}")
        
        # Determine input type if not provided
        if input_type is None:
            input_type = self._detect_input_type(input_data)
            logger.info(f"Auto-detected input type: {input_type}")
        
        # Process based on input type
        if input_type == "text":
            return self._process_text(input_data)
        elif input_type == "image":
            return self._process_image(input_data)
        else:
            error_msg = f"Unsupported input type: {input_type}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _detect_input_type(self, input_data: Any) -> str:
        """
        Automatically detect the type of input data.
        
        Args:
            input_data: The input data to analyze
            
        Returns:
            String indicating detected type ('text', 'image', or 'unknown')
        """
        # If input is a string, check if it's a file path or text content
        if isinstance(input_data, str):
            # Check if it's a file path
            if os.path.exists(input_data):
                mime_type, _ = mimetypes.guess_type(input_data)
                if mime_type:
                    if mime_type.startswith('image/'):
                        return "image"
                    elif mime_type.startswith('text/') or mime_type in ['application/pdf', 'application/msword']:
                        return "text"
            
            # If not a file or unrecognized mime type, analyze the content
            # Check if it looks like an image path or URL
            if re.search(r'\.(jpg|jpeg|png|gif|bmp|webp)$', input_data, re.IGNORECASE):
                return "image"
            
            # If it has significant text content, treat as text
            if len(input_data) > 50 or input_data.count(' ') > 10:
                return "text"
        
        # If input is a file-like object, check its content type
        elif hasattr(input_data, 'read') and callable(input_data.read):
            # Try to determine from filename if available
            if hasattr(input_data, 'name'):
                mime_type, _ = mimetypes.guess_type(input_data.name)
                if mime_type:
                    if mime_type.startswith('image/'):
                        return "image"
                    elif mime_type.startswith('text/') or mime_type in ['application/pdf', 'application/msword']:
                        return "text"
            
            # Read a sample of the content to guess the type
            try:
                # Remember current position
                pos = input_data.tell()
                
                # Read a sample
                sample = input_data.read(1024)
                
                # Reset to original position
                input_data.seek(pos)
                
                # Check if it looks like binary (image) data
                if isinstance(sample, bytes) and not all(c < 128 for c in sample):
                    return "image"
                
                # If it looks like text, treat as text
                return "text"
            except:
                pass
        
        # If input is bytes, check if it looks like an image
        elif isinstance(input_data, bytes):
            # Check for common image file signatures
            if input_data.startswith(b'\xff\xd8\xff'):  # JPEG
                return "image"
            elif input_data.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                return "image"
            elif input_data.startswith(b'GIF87a') or input_data.startswith(b'GIF89a'):  # GIF
                return "image"
            
            # If not recognized as an image, try to decode as text
            try:
                decoded = input_data.decode('utf-8')
                if len(decoded) > 50 or decoded.count(' ') > 10:
                    return "text"
            except:
                pass
        
        # Default to unknown if we can't determine
        return "unknown"
    
    def _process_text(self, text_input: Any) -> Dict[str, Any]:
        """
        Process text input through the text analysis pipeline.
        
        Args:
            text_input: Text content or file path/object
            
        Returns:
            Dictionary containing text analysis results
        """
        # Extract text content if input is a file path or file-like object
        text_content = self._extract_text_content(text_input)
        
        if not text_content:
            return {"error": "Failed to extract text content from input"}
        
        logger.info(f"Processing text content (length: {len(text_content)})")
        
        # Pre-process the text
        processed_text = self._preprocess_text(text_content)
        
        # Analyze the text
        results = self.text_analyzer.analyze(processed_text)
        
        # Check confidence level
        confidence = results.get("confidence", 0.0)
        
        # If confidence is low, try to improve results
        if confidence < self.low_confidence_threshold:
            logger.info(f"Low confidence ({confidence}), attempting to improve results")
            improved_results = self._handle_uncertain_text_classification(processed_text, results)
            
            # Merge improved results with original results
            results = self._merge_results(results, improved_results)
        
        # Add metadata
        results["metadata"] = self._extract_text_metadata(text_content)
        
        # Generate final report
        results["report"] = self._generate_text_report(results)
        
        return results
    
    def _extract_text_content(self, text_input: Any) -> Optional[str]:
        """
        Extract text content from various input types.
        
        Args:
            text_input: Text content, file path, or file-like object
            
        Returns:
            Extracted text content or None if extraction failed
        """
        # If input is already a string, return it directly
        if isinstance(text_input, str):
            # Check if it's a file path
            if os.path.exists(text_input):
                try:
                    with open(text_input, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading text file: {e}")
                    return None
            else:
                # Treat as direct text content
                return text_input
        
        # If input is a file-like object, read from it
        elif hasattr(text_input, 'read') and callable(text_input.read):
            try:
                # Check if it's a binary or text file
                if hasattr(text_input, 'mode') and 'b' in text_input.mode:
                    # Binary mode, decode as UTF-8
                    content = text_input.read()
                    return content.decode('utf-8')
                else:
                    # Text mode
                    return text_input.read()
            except Exception as e:
                logger.error(f"Error reading from file-like object: {e}")
                return None
        
        # If input is bytes, try to decode as UTF-8
        elif isinstance(text_input, bytes):
            try:
                return text_input.decode('utf-8')
            except Exception as e:
                logger.error(f"Error decoding bytes as UTF-8: {e}")
                return None
        
        # Unsupported input type
        logger.error(f"Unsupported text input type: {type(text_input)}")
        return None
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before analysis.
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any control characters
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        return text
    
    def _handle_uncertain_text_classification(self, text: str, initial_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cases where text classification confidence is low.
        
        Args:
            text: The text content
            initial_results: Initial analysis results
            
        Returns:
            Improved analysis results
        """
        improved_results = {}
        
        # Try different analysis settings
        logger.info("Trying alternative analysis settings for uncertain text")
        
        # Create a new analyzer with different settings
        alt_analyzer = TextAnalysisModule(
            use_stylometric=True,
            use_perplexity=True,
            use_transformer=True,
            transformer_model="roberta-large"  # Try a different model
        )
        
        # Get alternative analysis
        alt_results = alt_analyzer.analyze(text)
        
        # If alternative analysis has higher confidence, use it
        if alt_results.get("confidence", 0.0) > initial_results.get("confidence", 0.0):
            logger.info("Alternative analysis produced higher confidence results")
            improved_results = alt_results
        
        # Try to query external resources if needed
        if not improved_results or improved_results.get("confidence", 0.0) < self.high_confidence_threshold:
            logger.info("Querying external resources for additional information")
            external_results = self._query_external_resources_for_text(text)
            
            if external_results:
                # Merge external results
                if improved_results:
                    improved_results = self._merge_results(improved_results, external_results)
                else:
                    improved_results = external_results
        
        return improved_results
    
    def _query_external_resources_for_text(self, text: str) -> Dict[str, Any]:
        """
        Query external resources to improve text classification.
        
        Args:
            text: The text content
            
        Returns:
            Additional analysis results from external resources
        """
        # This is a placeholder for querying external APIs or services
        # In a real implementation, this would call external AI detection services
        
        # Simulate external API call
        logger.info("Simulating external API call for text analysis")
        
        # In a real implementation, this would be an actual API call
        # For example:
        # response = requests.post(
        #     "https://api.external-ai-detector.com/analyze",
        #     json={"text": text},
        #     headers={"Authorization": f"Bearer {self.api_keys['external_service']}"}
        # )
        # external_results = response.json()
        
        # For now, return a simulated result
        external_results = {
            "external_analysis": {
                "classification": "ai-generated" if len(text) % 2 == 0 else "human-written",
                "confidence": 0.75,
                "source": "simulated_external_api"
            }
        }
        
        return external_results
    
    def _extract_text_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text content.
        
        Args:
            text: The text content
            
        Returns:
            Dictionary containing text metadata
        """
        # Calculate basic text statistics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        
        # Detect language (simplified)
        language = self._detect_language(text)
        
        # Extract any URLs
        urls = re.findall(r'https?://\S+', text)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "language": language,
            "urls": urls,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the text (simplified implementation).
        
        Args:
            text: The text content
            
        Returns:
            Detected language code
        """
        # This is a very simplified language detection
        # In a real implementation, use a proper language detection library
        
        # Check for common English words
        english_words = ['the', 'and', 'is', 'in', 'to', 'it', 'that', 'was', 'for', 'on']
        english_count = sum(1 for word in english_words if f" {word} " in f" {text} ")
        
        # Check for common Spanish words
        spanish_words = ['el', 'la', 'en', 'y', 'es', 'que', 'de', 'un', 'una', 'por']
        spanish_count = sum(1 for word in spanish_words if f" {word} " in f" {text} ")
        
        # Check for common French words
        french_words = ['le', 'la', 'et', 'est', 'en', 'que', 'qui', 'dans', 'pour', 'pas']
        french_count = sum(1 for word in french_words if f" {word} " in f" {text} ")
        
        # Determine language based on word counts
        if english_count > spanish_count and english_count > french_count:
            return "en"
        elif spanish_count > english_count and spanish_count > french_count:
            return "es"
        elif french_count > english_count and french_count > spanish_count:
            return "fr"
        else:
            return "unknown"
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive text analysis report.
        
        Args:
            results: The analysis results
            
        Returns:
            Formatted report as a string
        """
        report = []
        report.append("=" * 50)
        report.append("AI-GENERATED CONTENT DETECTION REPORT - TEXT ANALYSIS")
        report.append("=" * 50)
        report.append("")
        
        # Add timestamp
        report.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add metadata
        if "metadata" in results:
            metadata = results["metadata"]
            report.append("TEXT METADATA:")
            report.append(f"Word Count: {metadata.get('word_count', 'Unknown')}")
            report.append(f"Sentence Count: {metadata.get('sentence_count', 'Unknown')}")
            report.append(f"Average Word Length: {metadata.get('avg_word_length', 0):.2f}")
            report.append(f"Language: {metadata.get('language', 'Unknown')}")
            if metadata.get('urls'):
                report.append(f"URLs Found: {', '.join(metadata.get('urls', []))}")
            report.append("")
        
        # Add main classification result
        classification = results.get("classification", "unknown")
        confidence = results.get("confidence", 0.0)
        report.append(f"CLASSIFICATION: {classification.upper()}")
        report.append(f"Confidence: {confidence:.2f}")
        report.append("")
        
        # Add explanation
        report.append("ANALYSIS EXPLANATION:")
        report.append(results.get("explanation", "No explanation available."))
        report.append("")
        
        # Add suspicion score
        suspicion_score = results.get("suspicion_score", 0.0)
        report.append(f"AI SUSPICION SCORE: {suspicion_score:.2f}")
        report.append("")
        
        # Add key indicators
        report.append("KEY INDICATORS:")
        for analyzer_name, analyzer_results in results.get("ai_indicators", {}).items():
            for indicator_name, indicator_data in analyzer_results.items():
                if isinstance(indicator_data, dict) and "is_suspicious" in indicator_data:
                    report.append(
                        f"- {indicator_name}: {indicator_data.get('value', 0):.2f} "
                        f"({'SUSPICIOUS' if indicator_data.get('is_suspicious', False) else 'NORMAL'})"
                    )
        report.append("")
        
        # Add detailed results
        report.append("DETAILED ANALYSIS RESULTS:")
        for analyzer_name, analyzer_results in results.get("detailed_results", {}).items():
            report.append(f"\n{analyzer_name.replace('_', ' ').upper()} ANALYSIS:")
            
            # Format the results
            for key, value in analyzer_results.items():
                if key != "ai_indicators":
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for subkey, subvalue in value.items():
                            report.append(f"    {subkey}: {subvalue}")
                    else:
                        report.append(f"  {key}: {value}")
        
        # Add external analysis results if available
        if "external_analysis" in results:
            report.append("\nEXTERNAL ANALYSIS RESULTS:")
            for key, value in results["external_analysis"].items():
                report.append(f"  {key}: {value}")
        
        return "\n".join(report)
    
    def _process_image(self, image_input: Any) -> Dict[str, Any]:
        """
        Process image input through the image analysis pipeline.
        
        Args:
            image_input: Image file path, file-like object, or bytes
            
        Returns:
            Dictionary containing image analysis results
        """
        # Save image to a temporary file if needed
        temp_image_path = None
        
        try:
            # Get image path or create temporary file
            image_path = self._get_image_path(image_input)
            
            if not image_path:
                return {"error": "Failed to process image input"}
            
            if image_path != image_input:
                temp_image_path = image_path
            
            logger.info(f"Processing image: {image_path}")
            
            # Analyze the image
            results = self.image_analyzer.analyze(image_path)
            
            # Check confidence level
            confidence = results.get("confidence", 0.0)
            
            # If confidence is low, try to improve results
            if confidence < self.low_confidence_threshold:
                logger.info(f"Low confidence ({confidence}), attempting to improve results")
                improved_results = self._handle_uncertain_image_classification(image_path, results)
                
                # Merge improved results with original results
                results = self._merge_results(results, improved_results)
            
            # Generate final report
            results["report"] = self._generate_image_report(results)
            
            return results
            
        finally:
            # Clean up temporary file if created
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except:
                    pass
    
    def _get_image_path(self, image_input: Any) -> Optional[str]:
        """
        Get image path or save to temporary file if needed.
        
        Args:
            image_input: Image file path, file-like object, or bytes
            
        Returns:
            Path to the image file or None if processing failed
        """
        # If input is a string, check if it's a file path
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                return image_input
            else:
                # Could be a URL or invalid path
                logger.error(f"Image path does not exist: {image_input}")
                return None
        
        # If input is a file-like object, save to temporary file
        elif hasattr(image_input, 'read') and callable(image_input.read):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    # Read from input and write to temporary file
                    tmp_file.write(image_input.read())
                    return tmp_file.name
            except Exception as e:
                logger.error(f"Error saving file-like object to temporary file: {e}")
                return None
        
        # If input is bytes, save to temporary file
        elif isinstance(image_input, bytes):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    # Write bytes to temporary file
                    tmp_file.write(image_input)
                    return tmp_file.name
            except Exception as e:
                logger.error(f"Error saving bytes to temporary file: {e}")
                return None
        
        # Unsupported input type
        logger.error(f"Unsupported image input type: {type(image_input)}")
        return None
    
    def _handle_uncertain_image_classification(self, image_path: str, initial_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cases where image classification confidence is low.
        
        Args:
            image_path: Path to the image file
            initial_results: Initial analysis results
            
        Returns:
            Improved analysis results
        """
        improved_results = {}
        
        # Try different analysis settings
        logger.info("Trying alternative analysis settings for uncertain image")
        
        # Create a new analyzer with different settings
        alt_analyzer = ImageAnalysisModule(
            use_frequency=True,
            use_texture=True,
            use_deep_learning=True,
            deep_learning_model="efficientnet"  # Try a different model
        )
        
        # Get alternative analysis
        alt_results = alt_analyzer.analyze(image_path)
        
        # If alternative analysis has higher confidence, use it
        if alt_results.get("confidence", 0.0) > initial_results.get("confidence", 0.0):
            logger.info("Alternative analysis produced higher confidence results")
            improved_results = alt_results
        
        # Try to query external resources if needed
        if not improved_results or improved_results.get("confidence", 0.0) < self.high_confidence_threshold:
            logger.info("Querying external resources for additional information")
            external_results = self._query_external_resources_for_image(image_path)
            
            if external_results:
                # Merge external results
                if improved_results:
                    improved_results = self._merge_results(improved_results, external_results)
                else:
                    improved_results = external_results
        
        return improved_results
    
    def _query_external_resources_for_image(self, image_path: str) -> Dict[str, Any]:
        """
        Query external resources to improve image classification.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Additional analysis results from external resources
        """
        # This is a placeholder for querying external APIs or services
        # In a real implementation, this would call external AI detection services
        
        # Simulate external API call
        logger.info("Simulating external API call for image analysis")
        
        # In a real implementation, this would be an actual API call
        # For example:
        # with open(image_path, 'rb') as img_file:
        #     response = requests.post(
        #         "https://api.external-ai-detector.com/analyze-image",
        #         files={"image": img_file},
        #         headers={"Authorization": f"Bearer {self.api_keys['external_service']}"}
        #     )
        # external_results = response.json()
        
        # For now, return a simulated result
        file_size = os.path.getsize(image_path)
        external_results = {
            "external_analysis": {
                "classification": "ai-generated" if file_size % 2 == 0 else "real-photo",
                "confidence": 0.75,
                "source": "simulated_external_api"
            }
        }
        
        return external_results
    
    def _generate_image_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive image analysis report.
        
        Args:
            results: The analysis results
            
        Returns:
            Formatted report as a string
        """
        report = []
        report.append("=" * 50)
        report.append("AI-GENERATED CONTENT DETECTION REPORT - IMAGE ANALYSIS")
        report.append("=" * 50)
        report.append("")
        
        # Add timestamp
        report.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add image metadata
        if "metadata" in results:
            metadata = results["metadata"]
            report.append("IMAGE METADATA:")
            report.append(f"Dimensions: {metadata.get('dimensions', 'Unknown')}")
            report.append(f"File Type: {metadata.get('file_extension', 'Unknown')}")
            report.append(f"File Size: {metadata.get('file_size', 0) / 1024:.1f} KB")
            report.append("")
        
        # Add main classification result
        classification = results.get("classification", "unknown")
        confidence = results.get("confidence", 0.0)
        report.append(f"CLASSIFICATION: {classification.upper()}")
        report.append(f"Confidence: {confidence:.2f}")
        report.append("")
        
        # Add explanation
        report.append("ANALYSIS EXPLANATION:")
        report.append(results.get("explanation", "No explanation available."))
        report.append("")
        
        # Add suspicion score
        suspicion_score = results.get("suspicion_score", 0.0)
        report.append(f"AI SUSPICION SCORE: {suspicion_score:.2f}")
        report.append("")
        
        # Add key indicators
        report.append("KEY INDICATORS:")
        for analyzer_name, analyzer_results in results.get("ai_indicators", {}).items():
            for indicator_name, indicator_data in analyzer_results.items():
                if isinstance(indicator_data, dict) and "is_suspicious" in indicator_data:
                    report.append(
                        f"- {indicator_name}: {indicator_data.get('value', 0):.2f} "
                        f"({'SUSPICIOUS' if indicator_data.get('is_suspicious', False) else 'NORMAL'})"
                    )
        report.append("")
        
        # Add detailed results
        report.append("DETAILED ANALYSIS RESULTS:")
        for analyzer_name, analyzer_results in results.get("detailed_results", {}).items():
            report.append(f"\n{analyzer_name.replace('_', ' ').upper()} ANALYSIS:")
            
            # Format the results
            for key, value in analyzer_results.items():
                if key != "ai_indicators":
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for subkey, subvalue in value.items():
                            report.append(f"    {subkey}: {subvalue}")
                    else:
                        report.append(f"  {key}: {value}")
        
        # Add external analysis results if available
        if "external_analysis" in results:
            report.append("\nEXTERNAL ANALYSIS RESULTS:")
            for key, value in results["external_analysis"].items():
                report.append(f"  {key}: {value}")
        
        return "\n".join(report)
    
    def _merge_results(self, primary_results: Dict[str, Any], secondary_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge results from multiple analyses, prioritizing primary results.
        
        Args:
            primary_results: Primary analysis results
            secondary_results: Secondary analysis results to merge
            
        Returns:
            Merged results dictionary
        """
        merged = primary_results.copy()
        
        # Merge top-level keys that don't exist in primary results
        for key, value in secondary_results.items():
            if key not in merged:
                merged[key] = value
        
        # Special handling for confidence and classification
        if "confidence" in secondary_results and secondary_results["confidence"] > merged.get("confidence", 0.0):
            merged["confidence"] = secondary_results["confidence"]
            if "classification" in secondary_results:
                merged["classification"] = secondary_results["classification"]
        
        # Merge detailed_results if present
        if "detailed_results" in merged and "detailed_results" in secondary_results:
            for analyzer, results in secondary_results["detailed_results"].items():
                if analyzer not in merged["detailed_results"]:
                    merged["detailed_results"][analyzer] = results
        
        # Merge ai_indicators if present
        if "ai_indicators" in merged and "ai_indicators" in secondary_results:
            for analyzer, indicators in secondary_results["ai_indicators"].items():
                if analyzer not in merged["ai_indicators"]:
                    merged["ai_indicators"][analyzer] = indicators
        
        # Add external_analysis if present in secondary results
        if "external_analysis" in secondary_results:
            merged["external_analysis"] = secondary_results["external_analysis"]
        
        return merged


# Example usage
if __name__ == "__main__":
    agent = ContentDetectionAgent()
    
    # Example text analysis
    text_results = agent.process_input("This is a sample text to analyze for AI-generated content detection.")
    print(f"Text Classification: {text_results.get('classification')} (confidence: {text_results.get('confidence'):.2f})")
    
    # Example image analysis (if an image file exists)
    sample_image_path = "sample_image.jpg"
    if os.path.exists(sample_image_path):
        image_results = agent.process_input(sample_image_path)
        print(f"Image Classification: {image_results.get('classification')} (confidence: {image_results.get('confidence'):.2f})")
