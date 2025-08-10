#!/usr/bin/env python3
"""
Logging Utilities
Function-based logging system
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Setup comprehensive logging system
    
    Args:
        log_dir: Directory for log files (optional)
        
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path("healthcare_monitor_functional/logs")
    
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('HealthcareMonitorFunctional')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    log_file = log_dir / f"healthcare_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler  
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Function-based logging system initialized")
    return logger


def log_detection_event(logger: logging.Logger, event_type: str, 
                       confidence: float, details: str = "") -> None:
    """
    Log detection events
    
    Args:
        logger: Logger instance
        event_type: Type of detection (fall, seizure, etc)
        confidence: Detection confidence score
        details: Additional event details
    """
    if event_type in ['fall_detected', 'seizure_detected']:
        logger.critical(f"{event_type.upper()}: confidence={confidence:.3f} {details}")
    elif 'warning' in event_type:
        logger.warning(f"{event_type}: confidence={confidence:.3f} {details}")
    else:
        logger.info(f"{event_type}: confidence={confidence:.3f} {details}")


def log_system_event(logger: logging.Logger, event: str, details: str = "") -> None:
    """
    Log system events
    
    Args:
        logger: Logger instance  
        event: Event description
        details: Additional details
    """
    logger.info(f"SYSTEM: {event} {details}")


def log_performance_metrics(logger: logging.Logger, metrics: dict) -> None:
    """
    Log performance metrics
    
    Args:
        logger: Logger instance
        metrics: Performance metrics dictionary
    """
    fps = metrics.get('fps', 0)
    processing_time = metrics.get('processing_time', 0)
    efficiency = metrics.get('efficiency', 0)
    
    logger.info(f"PERFORMANCE: FPS={fps:.1f}, ProcessingTime={processing_time:.3f}s, Efficiency={efficiency:.1f}%")


def log_error(logger: logging.Logger, component: str, error: Exception) -> None:
    """
    Log error events
    
    Args:
        logger: Logger instance
        component: Component where error occurred
        error: Exception object
    """
    logger.error(f"{component} ERROR: {str(error)}")
    

def log_statistics_summary(logger: logging.Logger, stats: dict) -> None:
    """
    Log final statistics summary
    
    Args:
        logger: Logger instance
        stats: Statistics dictionary
    """
    runtime = stats.get('runtime', 0)
    total_frames = stats.get('total_frames', 0)
    fall_detections = stats.get('fall_detections', 0)
    seizure_detections = stats.get('seizure_detections', 0)
    
    logger.info(f"FINAL STATS: Runtime={runtime:.1f}min, Frames={total_frames}, Falls={fall_detections}, Seizures={seizure_detections}")
