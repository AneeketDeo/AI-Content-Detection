"""
Frequency Domain Analysis Module for AI-Generated Image Detection

This module implements frequency domain analysis techniques to detect AI-generated images.
It focuses on analyzing artifacts and patterns in the frequency domain that are
characteristic of GAN-generated and other AI-synthesized images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import os


class FrequencyDomainAnalyzer:
    """
    Analyzes images in the frequency domain to detect AI-generated content.
    
    Many AI image generators leave characteristic artifacts in the frequency domain
    that can be detected through Fourier transform analysis. This class implements
    methods to detect these artifacts and patterns.
    """
    
    def __init__(self):
        """Initialize the frequency domain analyzer."""
        pass
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform frequency domain analysis on the provided image.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Dictionary containing frequency domain analysis results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Failed to load image: {image_path}"}
            
            # Convert to grayscale for frequency analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Perform frequency domain analysis
            spectral_features = self._extract_spectral_features(gray)
            frequency_artifacts = self._detect_frequency_artifacts(gray)
            dct_features = self._extract_dct_features(gray)
            
            # Combine all features
            features = {
                "spectral_features": spectral_features,
                "frequency_artifacts": frequency_artifacts,
                "dct_features": dct_features
            }
            
            # Add AI detection indicators
            features["ai_indicators"] = self._extract_ai_indicators(features)
            
            return features
            
        except Exception as e:
            return {"error": f"Error analyzing image: {str(e)}"}
    
    def _extract_spectral_features(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the frequency spectrum using Fourier Transform.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            Dictionary containing spectral features
        """
        # Apply 2D Fourier Transform
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Calculate statistics of the magnitude spectrum
        mean_magnitude = np.mean(magnitude_spectrum)
        std_magnitude = np.std(magnitude_spectrum)
        max_magnitude = np.max(magnitude_spectrum)
        
        # Analyze spectral distribution in different regions
        h, w = magnitude_spectrum.shape
        center_region = magnitude_spectrum[h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8]
        mid_region = magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]
        high_freq_region = magnitude_spectrum.copy()
        high_freq_region[h//4:3*h//4, w//4:3*w//4] = 0
        
        # Calculate region statistics
        center_mean = np.mean(center_region)
        mid_mean = np.mean(mid_region)
        high_freq_mean = np.mean(high_freq_region)
        
        # Calculate ratios between regions
        center_to_high_ratio = center_mean / high_freq_mean if high_freq_mean != 0 else 0
        mid_to_high_ratio = mid_mean / high_freq_mean if high_freq_mean != 0 else 0
        
        return {
            "mean_magnitude": float(mean_magnitude),
            "std_magnitude": float(std_magnitude),
            "max_magnitude": float(max_magnitude),
            "center_mean": float(center_mean),
            "mid_mean": float(mid_mean),
            "high_freq_mean": float(high_freq_mean),
            "center_to_high_ratio": float(center_to_high_ratio),
            "mid_to_high_ratio": float(mid_to_high_ratio)
        }
    
    def _detect_frequency_artifacts(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect artifacts in the frequency domain that are characteristic of AI-generated images.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            Dictionary containing detected artifacts
        """
        # Apply 2D Fourier Transform
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Detect grid-like patterns (common in some GANs)
        h, w = magnitude_spectrum.shape
        grid_score = self._detect_grid_pattern(magnitude_spectrum)
        
        # Detect radial patterns (common in some AI generators)
        radial_score = self._detect_radial_pattern(magnitude_spectrum)
        
        # Detect frequency dropouts (missing frequency components)
        dropout_score = self._detect_frequency_dropouts(magnitude_spectrum)
        
        # Detect abnormal frequency distribution
        distribution_score = self._analyze_frequency_distribution(magnitude_spectrum)
        
        return {
            "grid_pattern_score": float(grid_score),
            "radial_pattern_score": float(radial_score),
            "frequency_dropout_score": float(dropout_score),
            "abnormal_distribution_score": float(distribution_score)
        }
    
    def _detect_grid_pattern(self, magnitude_spectrum: np.ndarray) -> float:
        """
        Detect grid-like patterns in the frequency spectrum.
        
        Args:
            magnitude_spectrum: Magnitude spectrum of the image
            
        Returns:
            Score indicating the presence of grid patterns (higher = more likely)
        """
        # Apply edge detection to find strong lines in the spectrum
        edges = cv2.Canny(magnitude_spectrum.astype(np.uint8), 100, 200)
        
        # Use Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # If no lines detected, return low score
        if lines is None:
            return 0.0
        
        # Count horizontal and vertical lines (near 0 and 90 degrees)
        h_count = 0
        v_count = 0
        for line in lines:
            rho, theta = line[0]
            if abs(theta) < 0.1 or abs(theta - np.pi) < 0.1:
                h_count += 1
            elif abs(theta - np.pi/2) < 0.1:
                v_count += 1
        
        # Calculate grid score based on number of horizontal and vertical lines
        grid_score = (h_count + v_count) / (len(lines) + 1)
        
        return grid_score
    
    def _detect_radial_pattern(self, magnitude_spectrum: np.ndarray) -> float:
        """
        Detect radial patterns in the frequency spectrum.
        
        Args:
            magnitude_spectrum: Magnitude spectrum of the image
            
        Returns:
            Score indicating the presence of radial patterns (higher = more likely)
        """
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create a radial distance map from center
        y, x = np.ogrid[:h, :w]
        distance_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distance map
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance_map / max_distance
        
        # Create radial bins
        num_bins = 20
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_means = []
        
        # Calculate mean magnitude for each radial bin
        for i in range(num_bins):
            bin_mask = (normalized_distance >= bin_edges[i]) & (normalized_distance < bin_edges[i+1])
            bin_mean = np.mean(magnitude_spectrum[bin_mask])
            bin_means.append(bin_mean)
        
        # Calculate variance of bin means
        bin_variance = np.var(bin_means)
        
        # Calculate radial score (higher variance indicates stronger radial patterns)
        radial_score = min(1.0, bin_variance / 100)
        
        return radial_score
    
    def _detect_frequency_dropouts(self, magnitude_spectrum: np.ndarray) -> float:
        """
        Detect frequency dropouts (missing frequency components).
        
        Args:
            magnitude_spectrum: Magnitude spectrum of the image
            
        Returns:
            Score indicating the presence of frequency dropouts (higher = more likely)
        """
        # Threshold the magnitude spectrum to identify significant components
        threshold = np.mean(magnitude_spectrum) + 0.5 * np.std(magnitude_spectrum)
        binary_spectrum = (magnitude_spectrum > threshold).astype(np.uint8)
        
        # Count the number of connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_spectrum)
        
        # Calculate the average size of connected components
        avg_size = np.mean(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0
        
        # Calculate the coverage ratio (percentage of spectrum covered by significant components)
        coverage_ratio = np.sum(binary_spectrum) / binary_spectrum.size
        
        # Calculate dropout score (higher coverage and more components = lower dropout)
        dropout_score = 1.0 - (coverage_ratio * min(1.0, num_labels / 100))
        
        return dropout_score
    
    def _analyze_frequency_distribution(self, magnitude_spectrum: np.ndarray) -> float:
        """
        Analyze the distribution of frequency components.
        
        Args:
            magnitude_spectrum: Magnitude spectrum of the image
            
        Returns:
            Score indicating abnormality in frequency distribution (higher = more abnormal)
        """
        # Calculate histogram of magnitude values
        hist, bins = np.histogram(magnitude_spectrum, bins=50)
        hist = hist / np.sum(hist)  # Normalize
        
        # Calculate entropy of the distribution
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate skewness
        mean = np.mean(magnitude_spectrum)
        std = np.std(magnitude_spectrum)
        skewness = np.mean(((magnitude_spectrum - mean) / std) ** 3) if std > 0 else 0
        
        # Calculate kurtosis
        kurtosis = np.mean(((magnitude_spectrum - mean) / std) ** 4) if std > 0 else 0
        
        # Calculate abnormality score based on entropy, skewness, and kurtosis
        # Natural images typically have high entropy and moderate skewness/kurtosis
        entropy_score = 1.0 - min(1.0, entropy / 5.0)  # Lower entropy is more suspicious
        skewness_score = min(1.0, abs(skewness) / 2.0)  # Higher absolute skewness is more suspicious
        kurtosis_score = min(1.0, abs(kurtosis - 3.0) / 5.0)  # Deviation from normal kurtosis (3) is suspicious
        
        # Combine scores
        abnormality_score = 0.4 * entropy_score + 0.3 * skewness_score + 0.3 * kurtosis_score
        
        return abnormality_score
    
    def _extract_dct_features(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features using Discrete Cosine Transform (DCT).
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            Dictionary containing DCT features
        """
        # Resize image to ensure consistent DCT block size
        h, w = gray_image.shape
        h_new, w_new = (h // 8) * 8, (w // 8) * 8
        if h_new == 0 or w_new == 0:
            h_new, w_new = 8, 8
        resized = cv2.resize(gray_image, (w_new, h_new))
        
        # Apply DCT to 8x8 blocks (similar to JPEG compression)
        dct_coeffs = np.zeros_like(resized, dtype=np.float32)
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                block = resized[i:i+8, j:j+8].astype(np.float32)
                dct_coeffs[i:i+8, j:j+8] = cv2.dct(block)
        
        # Analyze DCT coefficient statistics
        dct_mean = np.mean(np.abs(dct_coeffs))
        dct_std = np.std(dct_coeffs)
        dct_max = np.max(np.abs(dct_coeffs))
        
        # Analyze AC coefficient distribution (excluding DC component)
        ac_coeffs = dct_coeffs.copy()
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                ac_coeffs[i, j] = 0  # Zero out DC components
        
        ac_mean = np.mean(np.abs(ac_coeffs))
        ac_std = np.std(ac_coeffs)
        
        # Calculate ratio of high to low frequency components
        low_freq = np.zeros_like(dct_coeffs, dtype=bool)
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                low_freq[i:i+4, j:j+4] = True
        
        high_freq = ~low_freq
        low_mean = np.mean(np.abs(dct_coeffs[low_freq]))
        high_mean = np.mean(np.abs(dct_coeffs[high_freq]))
        high_to_low_ratio = high_mean / low_mean if low_mean > 0 else 0
        
        return {
            "dct_mean": float(dct_mean),
            "dct_std": float(dct_std),
            "dct_max": float(dct_max),
            "ac_mean": float(ac_mean),
            "ac_std": float(ac_std),
            "high_to_low_ratio": float(high_to_low_ratio)
        }
    
    def _extract_ai_indicators(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract indicators that might suggest AI-generated content based on frequency analysis.
        
        Args:
            features: Dictionary with frequency analysis features
            
        Returns:
            Dictionary with AI indicators
        """
        indicators = {}
        
        # Grid pattern indicator (common in some GANs)
        grid_score = features["frequency_artifacts"]["grid_pattern_score"]
        indicators["grid_pattern"] = {
            "value": grid_score,
            "is_suspicious": grid_score > 0.5,
            "explanation": "Grid-like patterns in frequency domain often indicate GAN-generated images"
        }
        
        # Radial pattern indicator
        radial_score = features["frequency_artifacts"]["radial_pattern_score"]
        indicators["radial_pattern"] = {
            "value": radial_score,
            "is_suspicious": radial_score > 0.6,
            "explanation": "Strong radial patterns can indicate certain types of AI generation"
        }
        
        # Frequency dropout indicator
        dropout_score = features["frequency_artifacts"]["frequency_dropout_score"]
        indicators["frequency_dropout"] = {
            "value": dropout_score,
            "is_suspicious": dropout_score > 0.7,
            "explanation": "Missing frequency components are common in AI-generated images"
        }
        
        # Abnormal frequency distribution
        distribution_score = features["frequency_artifacts"]["abnormal_distribution_score"]
        indicators["frequency_distribution"] = {
            "value": distribution_score,
            "is_suspicious": distribution_score > 0.6,
            "explanation": "Abnormal frequency distribution is characteristic of synthetic images"
        }
        
        # High-to-low frequency ratio
        high_to_low_ratio = features["dct_features"]["high_to_low_ratio"]
        indicators["high_to_low_ratio"] = {
            "value": high_to_low_ratio,
            "is_suspicious": high_to_low_ratio < 0.1,
            "explanation": "AI-generated images often have unusually low high-frequency components"
        }
        
        # Overall suspicion score (weighted average of binary indicators)
        weights = {
            "grid_pattern": 0.25,
            "radial_pattern": 0.15,
            "frequency_dropout": 0.25,
            "frequency_distribution": 0.2,
            "high_to_low_ratio": 0.15
        }
        
        suspicion_score = sum(
            weights[key] * indicators[key]["is_suspicious"]
            for key in weights.keys()
        )
        
        indicators["overall_suspicion_score"] = suspicion_score
        
        return indicators


# Example usage
if __name__ == "__main__":
    analyzer = FrequencyDomainAnalyzer()
    sample_image_path = "sample_image.jpg"
    if os.path.exists(sample_image_path):
        results = analyzer.analyze(sample_image_path)
        print(f"AI indicators: {results['ai_indicators']}")
