"""
Texture and Noise Analysis Module for AI-Generated Image Detection

This module implements texture and noise analysis techniques to detect AI-generated images.
It focuses on analyzing noise patterns, texture consistency, and other visual artifacts
that are characteristic of AI-synthesized images.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import os
from scipy import ndimage


class TextureNoiseAnalyzer:
    """
    Analyzes image texture and noise patterns to detect AI-generated content.
    
    AI-generated images often have distinctive noise patterns and texture
    inconsistencies that differ from natural photographs. This class implements
    methods to detect these patterns and inconsistencies.
    """
    
    def __init__(self):
        """Initialize the texture and noise analyzer."""
        pass
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform texture and noise analysis on the provided image.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Dictionary containing texture and noise analysis results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Failed to load image: {image_path}"}
            
            # Convert to RGB for analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform texture and noise analysis
            noise_features = self._analyze_noise_patterns(image_rgb)
            texture_features = self._analyze_texture_consistency(image_rgb)
            edge_features = self._analyze_edge_coherence(image_rgb)
            
            # Combine all features
            features = {
                "noise_features": noise_features,
                "texture_features": texture_features,
                "edge_features": edge_features
            }
            
            # Add AI detection indicators
            features["ai_indicators"] = self._extract_ai_indicators(features)
            
            return features
            
        except Exception as e:
            return {"error": f"Error analyzing image: {str(e)}"}
    
    def _analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze noise patterns in the image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary containing noise pattern features
        """
        # Extract noise using a high-pass filter
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blurred)
        
        # Calculate noise statistics
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        noise_entropy = self._calculate_entropy(noise)
        
        # Analyze noise in different channels
        channels = cv2.split(image)
        channel_noise = []
        
        for i, channel in enumerate(channels):
            blurred_channel = cv2.GaussianBlur(channel, (5, 5), 0)
            channel_noise.append(cv2.subtract(channel, blurred_channel))
        
        # Calculate cross-channel noise correlation
        corr_rg = np.corrcoef(channel_noise[0].flatten(), channel_noise[1].flatten())[0, 1]
        corr_rb = np.corrcoef(channel_noise[0].flatten(), channel_noise[2].flatten())[0, 1]
        corr_gb = np.corrcoef(channel_noise[1].flatten(), channel_noise[2].flatten())[0, 1]
        
        # Calculate noise periodicity
        noise_periodicity = self._calculate_noise_periodicity(noise)
        
        # Calculate noise uniformity across image regions
        noise_uniformity = self._calculate_noise_uniformity(noise)
        
        return {
            "noise_mean": float(noise_mean),
            "noise_std": float(noise_std),
            "noise_entropy": float(noise_entropy),
            "cross_correlation": {
                "rg": float(corr_rg),
                "rb": float(corr_rb),
                "gb": float(corr_gb)
            },
            "noise_periodicity": float(noise_periodicity),
            "noise_uniformity": float(noise_uniformity)
        }
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate the entropy of an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Entropy value
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _calculate_noise_periodicity(self, noise: np.ndarray) -> float:
        """
        Calculate the periodicity of noise patterns.
        
        Args:
            noise: Noise image as numpy array
            
        Returns:
            Periodicity score (higher = more periodic)
        """
        # Calculate autocorrelation
        noise_norm = noise - np.mean(noise)
        autocorr = ndimage.correlate(noise_norm, noise_norm, mode='constant')
        
        # Normalize autocorrelation
        autocorr = autocorr / np.max(autocorr)
        
        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr.flatten(), height=0.2)
        
        # Calculate periodicity score based on number and strength of peaks
        if len(peaks) == 0:
            return 0.0
        
        peak_heights = autocorr.flatten()[peaks]
        periodicity_score = np.mean(peak_heights) * min(1.0, len(peaks) / 10)
        
        return periodicity_score
    
    def _calculate_noise_uniformity(self, noise: np.ndarray) -> float:
        """
        Calculate the uniformity of noise across different image regions.
        
        Args:
            noise: Noise image as numpy array
            
        Returns:
            Uniformity score (higher = more uniform)
        """
        h, w = noise.shape
        block_h, block_w = max(1, h // 4), max(1, w // 4)
        
        # Calculate noise statistics for each block
        block_stats = []
        for i in range(0, h, block_h):
            for j in range(0, w, block_w):
                block = noise[i:min(i+block_h, h), j:min(j+block_w, w)]
                block_stats.append((np.mean(block), np.std(block)))
        
        # Calculate variance of statistics across blocks
        mean_variance = np.var([stat[0] for stat in block_stats])
        std_variance = np.var([stat[1] for stat in block_stats])
        
        # Calculate uniformity score (higher variance = lower uniformity)
        uniformity_score = 1.0 - min(1.0, (mean_variance + std_variance) / 100)
        
        return uniformity_score
    
    def _analyze_texture_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze texture consistency across the image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary containing texture consistency features
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture features using Haralick texture features
        texture_features = self._calculate_haralick_features(gray)
        
        # Calculate local binary patterns
        lbp_features = self._calculate_lbp_features(gray)
        
        # Calculate texture consistency across regions
        texture_consistency = self._calculate_texture_consistency(gray)
        
        return {
            "haralick_features": texture_features,
            "lbp_features": lbp_features,
            "texture_consistency": float(texture_consistency)
        }
    
    def _calculate_haralick_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Calculate Haralick texture features.
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            Dictionary containing Haralick texture features
        """
        # Calculate GLCM (Gray-Level Co-occurrence Matrix)
        glcm = self._calculate_glcm(gray)
        
        # Calculate Haralick features from GLCM
        contrast = np.sum(np.abs(np.subtract.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])))**2 * glcm)
        dissimilarity = np.sum(np.abs(np.subtract.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1]))) * glcm)
        homogeneity = np.sum(glcm / (1 + np.abs(np.subtract.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])))))
        energy = np.sum(glcm**2)
        correlation = np.sum((np.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])) - np.outer(np.mean(np.arange(glcm.shape[0])), np.mean(np.arange(glcm.shape[1])))) * glcm)
        
        return {
            "contrast": float(contrast),
            "dissimilarity": float(dissimilarity),
            "homogeneity": float(homogeneity),
            "energy": float(energy),
            "correlation": float(correlation)
        }
    
    def _calculate_glcm(self, gray: np.ndarray) -> np.ndarray:
        """
        Calculate Gray-Level Co-occurrence Matrix.
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            GLCM matrix
        """
        # Reduce gray levels for computational efficiency
        gray_reduced = (gray / 16).astype(np.uint8)
        
        # Calculate GLCM
        h, w = gray_reduced.shape
        glcm = np.zeros((16, 16))
        
        for i in range(h-1):
            for j in range(w-1):
                glcm[gray_reduced[i, j], gray_reduced[i, j+1]] += 1
                glcm[gray_reduced[i, j], gray_reduced[i+1, j]] += 1
        
        # Normalize GLCM
        if np.sum(glcm) > 0:
            glcm = glcm / np.sum(glcm)
        
        return glcm
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Calculate Local Binary Pattern features.
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            Dictionary containing LBP features
        """
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        
        # Calculate LBP statistics
        lbp_mean = np.mean(lbp)
        lbp_std = np.std(lbp)
        lbp_entropy = self._calculate_entropy(lbp)
        
        # Calculate uniformity of LBP codes
        # Uniform patterns have at most 2 bitwise transitions
        uniform_patterns = 0
        for i in range(256):
            binary = format(i, '08b')
            transitions = sum(binary[j] != binary[j+1] for j in range(7))
            if transitions <= 2:
                uniform_patterns += hist[i]
        
        return {
            "lbp_mean": float(lbp_mean),
            "lbp_std": float(lbp_std),
            "lbp_entropy": float(lbp_entropy),
            "uniform_pattern_ratio": float(uniform_patterns)
        }
    
    def _calculate_texture_consistency(self, gray: np.ndarray) -> float:
        """
        Calculate texture consistency across different image regions.
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            Texture consistency score (higher = more consistent)
        """
        h, w = gray.shape
        block_h, block_w = max(1, h // 4), max(1, w // 4)
        
        # Calculate LBP histograms for each block
        block_hists = []
        for i in range(0, h, block_h):
            for j in range(0, w, block_w):
                block = gray[i:min(i+block_h, h), j:min(j+block_w, w)]
                
                # Calculate simple LBP for this block
                block_lbp = np.zeros_like(block)
                for bi in range(1, block.shape[0]-1):
                    for bj in range(1, block.shape[1]-1):
                        center = block[bi, bj]
                        code = 0
                        code |= (block[bi-1, bj-1] >= center) << 7
                        code |= (block[bi-1, bj] >= center) << 6
                        code |= (block[bi-1, bj+1] >= center) << 5
                        code |= (block[bi, bj+1] >= center) << 4
                        code |= (block[bi+1, bj+1] >= center) << 3
                        code |= (block[bi+1, bj] >= center) << 2
                        code |= (block[bi+1, bj-1] >= center) << 1
                        code |= (block[bi, bj-1] >= center) << 0
                        block_lbp[bi, bj] = code
                
                # Calculate histogram
                hist, _ = np.histogram(block_lbp, bins=256, range=(0, 256))
                hist = hist / (np.sum(hist) + 1e-10)
                block_hists.append(hist)
        
        # Calculate pairwise histogram distances
        distances = []
        for i in range(len(block_hists)):
            for j in range(i+1, len(block_hists)):
                # Chi-square distance between histograms
                chi_square = np.sum((block_hists[i] - block_hists[j])**2 / (block_hists[i] + block_hists[j] + 1e-10))
                distances.append(chi_square)
        
        # Calculate consistency score (lower distance = higher consistency)
        if len(distances) > 0:
            consistency_score = 1.0 - min(1.0, np.mean(distances) / 2.0)
        else:
            consistency_score = 0.5  # Default for very small images
        
        return consistency_score
    
    def _analyze_edge_coherence(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze edge coherence and artifacts.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary containing edge coherence features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Canny edge detector
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge statistics
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate edge continuity
        edge_continuity = self._calculate_edge_continuity(edges)
        
        # Calculate edge direction coherence
        edge_direction_coherence = self._calculate_edge_direction_coherence(gray)
        
        # Detect unnatural edge artifacts
        edge_artifacts = self._detect_edge_artifacts(gray, edges)
        
        return {
            "edge_density": float(edge_density),
            "edge_continuity": float(edge_continuity),
            "edge_direction_coherence": float(edge_direction_coherence),
            "edge_artifacts": float(edge_artifacts)
        }
    
    def _calculate_edge_continuity(self, edges: np.ndarray) -> float:
        """
        Calculate the continuity of edges.
        
        Args:
            edges: Binary edge image
            
        Returns:
            Edge continuity score (higher = more continuous)
        """
        # Count isolated edge pixels
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(edges, kernel, iterations=1)
        
        # Edge pixels that disappear after erosion are potentially isolated
        isolated = cv2.subtract(edges, eroded)
        isolated_count = np.sum(isolated > 0)
        
        # Calculate continuity score
        if np.sum(edges > 0) > 0:
            continuity_score = 1.0 - (isolated_count / np.sum(edges > 0))
        else:
            continuity_score = 0.5  # Default for images with no edges
        
        return continuity_score
    
    def _calculate_edge_direction_coherence(self, gray: np.ndarray) -> float:
        """
        Calculate the coherence of edge directions.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Edge direction coherence score (higher = more coherent)
        """
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Threshold to consider only strong edges
        strong_edges = magnitude > np.mean(magnitude) + np.std(magnitude)
        
        # Calculate histogram of directions for strong edges
        hist, _ = np.histogram(direction[strong_edges], bins=36, range=(-np.pi, np.pi))
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Calculate entropy of direction distribution
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate coherence score (lower entropy = higher coherence)
        coherence_score = 1.0 - min(1.0, entropy / 5.0)
        
        return coherence_score
    
    def _detect_edge_artifacts(self, gray: np.ndarray, edges: np.ndarray) -> float:
        """
        Detect unnatural edge artifacts.
        
        Args:
            gray: Grayscale image
            edges: Binary edge image
            
        Returns:
            Edge artifact score (higher = more artifacts)
        """
        # Look for perfectly straight lines (common in AI-generated images)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
        
        # Calculate straight line score
        if lines is not None:
            straight_line_count = len(lines)
            straight_line_score = min(1.0, straight_line_count / 50)
        else:
            straight_line_score = 0.0
        
        # Look for regular patterns in edge orientations
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        direction = np.arctan2(grad_y, grad_x)
        
        # Calculate histogram of directions
        hist, _ = np.histogram(direction, bins=36, range=(-np.pi, np.pi))
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Look for peaks in the histogram (preferred directions)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=0.05)
        
        # Calculate regularity score based on number of peaks
        regularity_score = min(1.0, len(peaks) / 5)
        
        # Combine scores
        artifact_score = 0.7 * straight_line_score + 0.3 * regularity_score
        
        return artifact_score
    
    def _extract_ai_indicators(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract indicators that might suggest AI-generated content based on texture and noise analysis.
        
        Args:
            features: Dictionary with texture and noise analysis features
            
        Returns:
            Dictionary with AI indicators
        """
        indicators = {}
        
        # Noise pattern indicators
        noise_features = features["noise_features"]
        
        # Noise uniformity (AI-generated images often have too uniform noise)
        noise_uniformity = noise_features["noise_uniformity"]
        indicators["noise_uniformity"] = {
            "value": noise_uniformity,
            "is_suspicious": noise_uniformity > 0.8,
            "explanation": "AI-generated images often have unnaturally uniform noise patterns"
        }
        
        # Noise periodicity (AI-generated images may have periodic noise)
        noise_periodicity = noise_features["noise_periodicity"]
        indicators["noise_periodicity"] = {
            "value": noise_periodicity,
            "is_suspicious": noise_periodicity > 0.6,
            "explanation": "Periodic noise patterns are common in some AI-generated images"
        }
        
        # Cross-channel correlation (AI-generated images often have high correlation)
        cross_corr = noise_features["cross_correlation"]
        avg_corr = (abs(cross_corr["rg"]) + abs(cross_corr["rb"]) + abs(cross_corr["gb"])) / 3
        indicators["noise_cross_correlation"] = {
            "value": avg_corr,
            "is_suspicious": avg_corr > 0.7,
            "explanation": "High correlation between color channels is common in AI-generated images"
        }
        
        # Texture consistency indicators
        texture_consistency = features["texture_features"]["texture_consistency"]
        indicators["texture_consistency"] = {
            "value": texture_consistency,
            "is_suspicious": texture_consistency > 0.85,
            "explanation": "AI-generated images often have unnaturally consistent texture across regions"
        }
        
        # Edge coherence indicators
        edge_features = features["edge_features"]
        
        # Edge artifacts
        edge_artifacts = edge_features["edge_artifacts"]
        indicators["edge_artifacts"] = {
            "value": edge_artifacts,
            "is_suspicious": edge_artifacts > 0.7,
            "explanation": "AI-generated images often contain unnatural edge artifacts"
        }
        
        # Edge direction coherence
        edge_direction_coherence = edge_features["edge_direction_coherence"]
        indicators["edge_direction_coherence"] = {
            "value": edge_direction_coherence,
            "is_suspicious": edge_direction_coherence > 0.8,
            "explanation": "Unnaturally coherent edge directions can indicate AI generation"
        }
        
        # Overall suspicion score (weighted average of binary indicators)
        weights = {
            "noise_uniformity": 0.2,
            "noise_periodicity": 0.15,
            "noise_cross_correlation": 0.15,
            "texture_consistency": 0.2,
            "edge_artifacts": 0.15,
            "edge_direction_coherence": 0.15
        }
        
        suspicion_score = sum(
            weights[key] * indicators[key]["is_suspicious"]
            for key in weights.keys()
        )
        
        indicators["overall_suspicion_score"] = suspicion_score
        
        return indicators


# Example usage
if __name__ == "__main__":
    analyzer = TextureNoiseAnalyzer()
    sample_image_path = "sample_image.jpg"
    if os.path.exists(sample_image_path):
        results = analyzer.analyze(sample_image_path)
        print(f"AI indicators: {results['ai_indicators']}")
