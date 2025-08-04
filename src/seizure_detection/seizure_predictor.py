"""
Seizure Predictor - Real-time seizure probability analysis
Handles temporal analysis and prediction logic for seizure detection
"""

import numpy as np
from typing import List, Dict, Optional
import logging
from collections import deque
import time

class SeizurePredictor:
    """
    Real-time seizure prediction and analysis
    Handles temporal smoothing, probability analysis, and alert generation
    """
    
    def __init__(self, 
                 temporal_window: int = 30,
                 smoothing_factor: float = 0.3,      # Tăng để responsive hơn
                 alert_threshold: float = 0.65,     # Giảm để dễ detect hơn  
                 warning_threshold: float = 0.45):  # Giảm để sensitive hơn
        """
        Initialize seizure predictor
        
        Args:
            temporal_window: Number of frames for temporal analysis
            smoothing_factor: Exponential smoothing factor for predictions
            alert_threshold: Threshold for seizure alerts
            warning_threshold: Threshold for warnings
        """
        self.logger = logging.getLogger(__name__)
        
        self.temporal_window = temporal_window
        self.smoothing_factor = smoothing_factor
        self.alert_threshold = alert_threshold
        self.warning_threshold = warning_threshold
        
        # Temporal buffers
        self.confidence_history = deque(maxlen=temporal_window)
        self.smoothed_confidence = 0.0
        self.prediction_history = deque(maxlen=100)  # Store last 100 predictions
        
        # Alert state management
        self.current_alert_level = 'normal'
        self.last_alert_time = None
        self.seizure_start_time = None
        self.seizure_duration = 0.0
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'seizures_detected': 0,
            'false_positives': 0,
            'average_confidence': 0.0,
            'max_confidence': 0.0,
            'alert_history': []
        }
        
        self.logger.info("SeizurePredictor initialized")
    
    def update_prediction(self, confidence: float, timestamp: Optional[float] = None) -> Dict:
        """
        Update seizure prediction with new confidence score
        
        Args:
            confidence: Raw confidence from VSViG model (0.0-1.0)
            timestamp: Optional timestamp, uses current time if None
            
        Returns:
            dict: Prediction result with alerts and analysis
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add to confidence history
        self.confidence_history.append(confidence)
        
        # Exponential smoothing
        if len(self.confidence_history) == 1:
            self.smoothed_confidence = confidence
        else:
            self.smoothed_confidence = (
                self.smoothing_factor * confidence + 
                (1 - self.smoothing_factor) * self.smoothed_confidence
            )
        
        # Temporal analysis
        temporal_analysis = self._analyze_temporal_pattern()
        
        # Determine alert level
        alert_result = self._determine_alert_level(
            confidence, self.smoothed_confidence, temporal_analysis
        )
        
        # Create prediction result
        result = {
            'raw_confidence': confidence,
            'smoothed_confidence': self.smoothed_confidence,
            'alert_level': alert_result['level'],
            'seizure_detected': alert_result['seizure_detected'],
            'alert_message': alert_result['message'],
            'temporal_analysis': temporal_analysis,
            'timestamp': timestamp,
            'seizure_duration': self.seizure_duration
        }
        
        # Update statistics
        self._update_statistics(result)
        
        # Store prediction
        self.prediction_history.append(result)
        
        return result
    
    def _analyze_temporal_pattern(self) -> Dict:
        """
        Analyze temporal patterns in confidence history
        
        Returns:
            dict: Temporal analysis results
        """
        if len(self.confidence_history) < 5:
            return {
                'trend': 'insufficient_data',
                'volatility': 0.0,
                'peak_confidence': 0.0,
                'sustained_high': False
            }
        
        history = np.array(list(self.confidence_history))
        
        # Calculate trend (simple linear regression slope)
        x = np.arange(len(history))
        trend_slope = np.polyfit(x, history, 1)[0]
        
        if trend_slope > 0.01:
            trend = 'increasing'
        elif trend_slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate volatility (standard deviation)
        volatility = float(np.std(history))
        
        # Peak confidence in recent window
        peak_confidence = float(np.max(history))
        
        # Check for sustained high confidence
        recent_window = history[-10:] if len(history) >= 10 else history
        sustained_high = bool(np.mean(recent_window) > self.warning_threshold)
        
        return {
            'trend': trend,
            'volatility': volatility,
            'peak_confidence': peak_confidence,
            'sustained_high': sustained_high,
            'trend_slope': float(trend_slope)
        }
    
    def _determine_alert_level(self, raw_conf: float, smooth_conf: float, temporal: Dict) -> Dict:
        """
        Determine alert level based on confidence and temporal analysis
        
        Args:
            raw_conf: Raw confidence score
            smooth_conf: Smoothed confidence score
            temporal: Temporal analysis results
            
        Returns:
            dict: Alert determination result
        """
        current_time = time.time()
        
        # Primary alert logic
        if smooth_conf >= self.alert_threshold or raw_conf >= self.alert_threshold + 0.1:
            # High confidence seizure detection
            alert_level = 'critical'
            seizure_detected = True
            message = f"🚨 SEIZURE DETECTED! Confidence: {smooth_conf:.2f}"
            
            # Track seizure duration
            if self.seizure_start_time is None:
                self.seizure_start_time = current_time
            self.seizure_duration = current_time - self.seizure_start_time
            
        elif smooth_conf >= self.warning_threshold:
            # Warning level
            alert_level = 'warning'
            seizure_detected = False
            message = f"⚠️ Seizure Warning: {smooth_conf:.2f}"
            
            # Reset seizure tracking if we were in seizure state
            if self.current_alert_level == 'critical':
                self.seizure_start_time = None
                self.seizure_duration = 0.0
                
        else:
            # Normal state
            alert_level = 'normal'
            seizure_detected = False
            message = f"✅ Normal: {smooth_conf:.2f}"
            
            # Reset seizure tracking
            self.seizure_start_time = None
            self.seizure_duration = 0.0
        
        # Enhanced logic based on temporal patterns
        if temporal['sustained_high'] and temporal['trend'] == 'increasing':
            if alert_level == 'normal':
                alert_level = 'warning'
                message = f"⚠️ Rising Pattern Detected: {smooth_conf:.2f}"
        
        # Update alert state
        if alert_level != self.current_alert_level:
            self.last_alert_time = current_time
            self.current_alert_level = alert_level
            
            # Log alert changes
            self.stats['alert_history'].append({
                'timestamp': current_time,
                'level': alert_level,
                'confidence': smooth_conf,
                'message': message
            })
        
        return {
            'level': alert_level,
            'seizure_detected': seizure_detected,
            'message': message
        }
    
    def _update_statistics(self, result: Dict):
        """Update prediction statistics"""
        self.stats['total_predictions'] += 1
        
        confidence = result['smoothed_confidence']
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (self.stats['total_predictions'] - 1) + 
             confidence) / self.stats['total_predictions']
        )
        
        if confidence > self.stats['max_confidence']:
            self.stats['max_confidence'] = confidence
        
        if result['seizure_detected']:
            self.stats['seizures_detected'] += 1
    
    def get_current_status(self) -> Dict:
        """
        Get current seizure prediction status
        
        Returns:
            dict: Current status information
        """
        return {
            'alert_level': self.current_alert_level,
            'smoothed_confidence': self.smoothed_confidence,
            'seizure_duration': self.seizure_duration,
            'buffer_filled': len(self.confidence_history) >= self.temporal_window,
            'last_alert_time': self.last_alert_time,
            'recent_trend': self._get_recent_trend()
        }
    
    def _get_recent_trend(self) -> str:
        """Get recent confidence trend"""
        if len(self.confidence_history) < 5:
            return 'insufficient_data'
        
        recent = list(self.confidence_history)[-5:]
        if recent[-1] > recent[0] + 0.1:
            return 'increasing'
        elif recent[-1] < recent[0] - 0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        return {
            **self.stats,
            'current_status': self.get_current_status(),
            'confidence_history_length': len(self.confidence_history),
            'prediction_history_length': len(self.prediction_history),
            'thresholds': {
                'alert': self.alert_threshold,
                'warning': self.warning_threshold
            }
        }
    
    def reset(self):
        """Reset predictor state"""
        self.confidence_history.clear()
        self.smoothed_confidence = 0.0
        self.current_alert_level = 'normal'
        self.seizure_start_time = None
        self.seizure_duration = 0.0
        self.logger.info("SeizurePredictor reset")
    
    def export_history(self) -> List[Dict]:
        """
        Export prediction history for analysis
        
        Returns:
            list: Historical predictions
        """
        return list(self.prediction_history)
