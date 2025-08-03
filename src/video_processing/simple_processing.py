"""
Simple Video Processing Components - Integrated Keyframe Detection
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import peakutils

# Import fall detection
try:
    from fall_detection import SimpleFallDetector
    FALL_DETECTION_AVAILABLE = True
except ImportError:
    try:
        from src.fall_detection import SimpleFallDetector
        FALL_DETECTION_AVAILABLE = True
    except ImportError:
        print("⚠️ Fall detection not available - continuing without it")
        FALL_DETECTION_AVAILABLE = False


class SimpleMotionDetector:
    """Simple Motion Detector không dùng loguru"""
    
    def __init__(self, threshold=150, start_frames=2, resolution=(256, 144)):
        """Initialize motion detector"""
        self.threshold = threshold
        self.start_frames = start_frames
        self.resolution = resolution
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=32, detectShadows=True
        )
        
        # Frame tracking
        self.frame_count = 0
        self.motion_detected = False
        
    def detect_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect motion in frame"""
        try:
            # Resize frame for performance
            resized = cv2.resize(frame, self.resolution)
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(resized)
            
            # Remove shadows
            fg_mask[fg_mask == 127] = 0
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Count white pixels
            motion_pixels = cv2.countNonZero(fg_mask)
            
            self.frame_count += 1
            
            # Determine motion
            if self.frame_count > self.start_frames:
                self.motion_detected = motion_pixels > self.threshold
            
            return {
                'motion_detected': self.motion_detected,
                'motion_pixels': motion_pixels,
                'threshold': self.threshold,
                'frame_count': self.frame_count
            }
            
        except Exception as e:
            print(f"❌ Motion detection error: {e}")
            return {
                'motion_detected': False,
                'motion_pixels': 0,
                'threshold': self.threshold,
                'frame_count': self.frame_count
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get motion detector statistics"""
        return {
            'frame_count': self.frame_count,
            'motion_detected': self.motion_detected,
            'threshold': self.threshold,
            'resolution': self.resolution
        }


class SimpleKeyframeDetector:
    """Simple Keyframe Detector adapted from video-keyframe-detector"""
    
    def __init__(self, threshold=0.3, max_keyframes=5, min_diff_threshold=0.01):
        """Initialize keyframe detector
        
        Args:
            threshold: Peak detection threshold (0.1-0.9)
            max_keyframes: Maximum keyframes to track
            min_diff_threshold: Minimum difference to consider as keyframe
        """
        self.threshold = threshold
        self.max_keyframes = max_keyframes
        self.min_diff_threshold = min_diff_threshold
        
        # Frame tracking
        self.last_frame = None
        self.diff_history = []
        self.frame_count = 0
        
        print(f"🎬 Keyframe detector initialized (threshold: {threshold})")
        
    def is_keyframe(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Real-time keyframe detection based on frame difference
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            (is_keyframe, confidence_score)
        """
        try:
            # Convert to grayscale and blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
            
            self.frame_count += 1
            
            # First frame is always a keyframe
            if self.last_frame is None:
                self.last_frame = blur_gray
                self.diff_history.append(0)
                return True, 1.0
            
            # Calculate frame difference
            diff = cv2.subtract(blur_gray, self.last_frame)
            diff_magnitude = cv2.countNonZero(diff)
            
            # Normalize by frame size
            normalized_diff = diff_magnitude / (frame.shape[0] * frame.shape[1])
            
            self.diff_history.append(normalized_diff)
            self.last_frame = blur_gray.copy()
            
            # Keep history manageable
            if len(self.diff_history) > 50:
                self.diff_history = self.diff_history[-30:]
            
            # Simple threshold-based detection for real-time
            is_keyframe = normalized_diff > self.min_diff_threshold
            
            # Enhanced detection using recent history
            if len(self.diff_history) >= 5:
                recent_avg = np.mean(self.diff_history[-5:])
                is_keyframe = is_keyframe and (normalized_diff > recent_avg * 1.5)
            
            return is_keyframe, normalized_diff
            
        except Exception as e:
            print(f"❌ Keyframe detection error: {e}")
            return False, 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get keyframe detector statistics"""
        return {
            'frame_count': self.frame_count,
            'diff_history_length': len(self.diff_history),
            'avg_diff': np.mean(self.diff_history) if self.diff_history else 0,
            'threshold': self.threshold,
            'min_diff_threshold': self.min_diff_threshold
        }


class SimpleFrameSaver:
    """Simple Frame Saver for important frames"""
    
    def __init__(self, base_path="data/saved_frames", max_files_per_folder=1000):
        """Initialize frame saver
        
        Args:
            base_path: Base directory for saving frames
            max_files_per_folder: Maximum files per category folder
        """
        self.base_path = base_path
        self.max_files_per_folder = max_files_per_folder
        
        # Create directories
        self.keyframes_path = os.path.join(base_path, "keyframes")
        self.detections_path = os.path.join(base_path, "detections") 
        self.alerts_path = os.path.join(base_path, "alerts")
        
        self._create_directories()
        
        print(f"💾 Frame saver initialized: {base_path}")
    
    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.keyframes_path, self.detections_path, self.alerts_path]:
            os.makedirs(path, exist_ok=True)
    
    def _get_timestamp_filename(self, prefix: str, suffix: str = "") -> str:
        """Generate timestamp-based filename"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Remove last 3 digits of microseconds
        if suffix:
            return f"{prefix}_{timestamp}_{suffix}.jpg"
        return f"{prefix}_{timestamp}.jpg"
    
    def save_keyframe(self, frame: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save keyframe with metadata
        
        Args:
            frame: Frame to save
            metadata: Frame metadata (confidence, timestamp, etc.)
            
        Returns:
            True if saved successfully
        """
        try:
            confidence = metadata.get('confidence', 0.0)
            filename = self._get_timestamp_filename("keyframe", f"conf_{confidence:.3f}")
            filepath = os.path.join(self.keyframes_path, filename)
            
            # Save image
            success = cv2.imwrite(filepath, frame)
            
            if success:
                # Save metadata
                self._save_metadata(filepath, metadata)
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error saving keyframe: {e}")
            return False
    
    def save_detection(self, frame: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save detection frame with metadata"""
        try:
            person_count = len(metadata.get('persons', []))
            confidence = metadata.get('max_confidence', 0.0)
            
            filename = self._get_timestamp_filename("detection", f"persons_{person_count}_conf_{confidence:.3f}")
            filepath = os.path.join(self.detections_path, filename)
            
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self._save_metadata(filepath, metadata)
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error saving detection: {e}")
            return False
    
    def save_alert(self, frame: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save alert frame with metadata"""
        try:
            alert_type = metadata.get('alert_type', 'unknown')
            confidence = metadata.get('confidence', 0.0)
            
            filename = self._get_timestamp_filename("alert", f"{alert_type}_conf_{confidence:.3f}")
            filepath = os.path.join(self.alerts_path, filename)
            
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self._save_metadata(filepath, metadata)
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error saving alert: {e}")
            return False
    
    def _save_metadata(self, image_path: str, metadata: Dict[str, Any]):
        """Save metadata as JSON file"""
        try:
            metadata_path = image_path.replace('.jpg', '_metadata.json')
            
            # Add timestamp if not present
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")
    
    def cleanup_old_files(self, days_old=7):
        """Clean up files older than specified days"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            
            for folder in [self.keyframes_path, self.detections_path, self.alerts_path]:
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.getctime(filepath) < cutoff_time:
                        os.remove(filepath)
                        
            print(f"🧹 Cleaned up files older than {days_old} days")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")


class SimpleYOLODetector:
    """Simple YOLO Detector"""
    
    def __init__(self, model_name='yolov8s', confidence=0.5, healthcare_mode=True, device='auto'):
        """Initialize YOLO detector"""
        self.model_name = model_name
        self.confidence = confidence
        self.healthcare_mode = healthcare_mode
        self.device = device
        
        self.model = None
        self.class_names = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            print(f"📦 Loading YOLO model: {self.model_name}")
            
            self.model = YOLO(f"{self.model_name}.pt")
            self.class_names = self.model.names
            
            print(f"✅ YOLO model loaded: {len(self.class_names)} classes")
            
        except Exception as e:
            print(f"❌ YOLO model loading error: {e}")
            
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect objects in frame"""
        try:
            if self.model is None:
                return {'detections': [], 'annotated_frame': frame}
            
            # Run inference
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            detections = []
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names.get(class_id, 'unknown') if self.class_names else 'unknown'
                        
                        # Healthcare mode: focus on person
                        if self.healthcare_mode and class_name != 'person':
                            continue
                            
                        # Add detection
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                        
                        # Draw on frame
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", 
                                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return {
                'detections': detections,
                'annotated_frame': annotated_frame
            }
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            return {'detections': [], 'annotated_frame': frame}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get YOLO detector statistics"""
        return {
            'model_name': self.model_name,
            'confidence': self.confidence,
            'healthcare_mode': self.healthcare_mode,
            'device': self.device,
            'model_loaded': self.model is not None,
            'class_count': len(self.class_names) if self.class_names else 0
        }


class SimpleVideoProcessor:
    """Simple Video Processor"""
    
    def __init__(self, motion_detector=None, yolo_detector=None, camera_object=None):
        """Initialize video processor"""
        self.motion_detector = motion_detector or SimpleMotionDetector()
        self.yolo_detector = yolo_detector or SimpleYOLODetector()
        self.camera_object = camera_object
        
        # Processing stats
        self.total_frames = 0
        self.motion_frames = 0
        self.detection_frames = 0
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame"""
        try:
            self.total_frames += 1
            
            # Step 1: Motion detection
            motion_result = self.motion_detector.detect_motion(frame)
            
            # Step 2: YOLO detection (if motion detected)
            if motion_result.get('motion_detected', False):
                self.motion_frames += 1
                yolo_result = self.yolo_detector.detect(frame)
                
                if yolo_result.get('detections'):
                    self.detection_frames += 1
                
                return {
                    'motion_result': motion_result,
                    'detections': yolo_result.get('detections', []),
                    'annotated_frame': yolo_result.get('annotated_frame', frame),
                    'processing_stats': self.get_stats()
                }
            else:
                return {
                    'motion_result': motion_result,
                    'detections': [],
                    'annotated_frame': frame,
                    'processing_stats': self.get_stats()
                }
                
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return {
                'motion_result': {'motion_detected': False},
                'detections': [],
                'annotated_frame': frame,
                'processing_stats': self.get_stats()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'total_frames': self.total_frames,
            'motion_frames': self.motion_frames,
            'detection_frames': self.detection_frames,
            'motion_rate': self.motion_frames / max(self.total_frames, 1),
            'detection_rate': self.detection_frames / max(self.total_frames, 1)
        }


class SimpleHealthcareAnalyzer:
    """Simple Healthcare Analyzer"""
    
    def __init__(self, fall_detection=True, position_tracking=True, 
                 medical_object_analysis=True, alert_threshold=0.7):
        """Initialize healthcare analyzer"""
        self.fall_detection = fall_detection
        self.position_tracking = position_tracking
        self.medical_object_analysis = medical_object_analysis
        self.alert_threshold = alert_threshold
        
        # Tracking data
        self.person_history = []
        self.fall_alerts = 0
        
    def analyze_frame(self, frame: np.ndarray, detections: list) -> Dict[str, Any]:
        """Analyze frame for healthcare events"""
        try:
            alerts = []
            
            # Focus on person detections
            persons = [d for d in detections if d.get('class_name') == 'person']
            
            # TODO: Fall detection sẽ được thay thế bằng model chuyên về té ngã
            # Removed simple aspect ratio fall detection
            
            # Track person positions
            if self.position_tracking:
                import time
                self.person_history.append({
                    'timestamp': time.time(),
                    'persons': len(persons),
                    'positions': [p['bbox'] for p in persons]
                })
                
                # Keep only recent history
                if len(self.person_history) > 100:
                    self.person_history.pop(0)
            
            return {
                'alerts': alerts,
                'person_count': len(persons),
                'fall_alerts_total': self.fall_alerts,
                'tracking_history_length': len(self.person_history)
            }
            
        except Exception as e:
            print(f"❌ Healthcare analysis error: {e}")
            return {
                'alerts': [],
                'person_count': 0,
                'fall_alerts_total': self.fall_alerts,
                'tracking_history_length': len(self.person_history)
            }


class IntegratedVideoProcessor:
    """Integrated Video Processor with Keyframe Detection Pipeline"""
    
    def __init__(self, 
                 motion_threshold=150,
                 keyframe_threshold=0.3,
                 yolo_confidence=0.5,
                 save_frames=True,
                 base_save_path="data/saved_frames"):
        """Initialize integrated processor
        
        Args:
            motion_threshold: Motion detection threshold
            keyframe_threshold: Keyframe detection threshold
            yolo_confidence: YOLO confidence threshold
            save_frames: Whether to save important frames
            base_save_path: Base path for saving frames
        """
        
        # Initialize components
        self.motion_detector = SimpleMotionDetector(threshold=motion_threshold)
        self.keyframe_detector = SimpleKeyframeDetector(threshold=keyframe_threshold)
        self.yolo_detector = SimpleYOLODetector(confidence=yolo_confidence)
        self.healthcare_analyzer = SimpleHealthcareAnalyzer()
        
        # Fall detection (optional)
        if FALL_DETECTION_AVAILABLE:
            self.fall_detector = SimpleFallDetector(confidence_threshold=0.7)
            print("🩺 Fall detection initialized")
        else:
            self.fall_detector = None
            print("⚠️ Fall detection not available")
        
        # Frame saver (optional)
        self.frame_saver = SimpleFrameSaver(base_save_path) if save_frames else None
        self.save_frames = save_frames
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'motion_frames': 0,
            'keyframes': 0,
            'yolo_processed': 0,
            'persons_detected': 0,
            'fall_detections': 0,
            'alerts_generated': 0
        }
        
        print("🚀 Integrated Video Processor initialized with Keyframe Detection!")
        print(f"   📹 Motion threshold: {motion_threshold}")
        print(f"   🎬 Keyframe threshold: {keyframe_threshold}")
        print(f"   🤖 YOLO confidence: {yolo_confidence}")
        print(f"   💾 Frame saving: {'Enabled' if save_frames else 'Disabled'}")
    
    def process_frame(self, frame: np.ndarray, save_keyframes=True) -> Dict[str, Any]:
        """Process frame through the integrated pipeline
        
        Pipeline: Motion Detection → Keyframe Detection → YOLO → Healthcare Analysis
        
        Args:
            frame: Input frame
            save_keyframes: Whether to save detected keyframes
            
        Returns:
            Processing results with all analysis data
        """
        
        self.stats['total_frames'] += 1
        
        # Stage 1: Motion Detection (quick filter)
        motion_result = self.motion_detector.detect_motion(frame)
        
        if motion_result['motion_detected']:
            self.stats['motion_frames'] += 1
            
            # Stage 2: Keyframe Detection (important frame filter)
            is_keyframe, keyframe_confidence = self.keyframe_detector.is_keyframe(frame)
            
            if is_keyframe:
                self.stats['keyframes'] += 1
                
                # Stage 3: YOLO Detection (only on keyframes)
                yolo_result = self.yolo_detector.detect(frame)
                detections = yolo_result.get('detections', [])
                self.stats['yolo_processed'] += 1
                
                # Stage 3.5: Fall Detection (if persons detected and available)
                fall_detected = False
                fall_confidence = 0.0
                fall_analysis = {}
                
                if FALL_DETECTION_AVAILABLE and self.fall_detector and detections:
                    # Check if any person detections exist
                    person_detections = [d for d in detections if d.get('class_name') == 'person']
                    if person_detections:
                        try:
                            # Use the person with highest confidence for fall detection
                            best_person = max(person_detections, key=lambda x: x.get('confidence', 0))
                            
                            # Extract bounding box
                            bbox = best_person.get('bbox', [])
                            if len(bbox) >= 4:
                                # Run fall detection
                                fall_result = self.fall_detector.detect_fall(frame, bbox)
                                fall_detected = fall_result.get('fall_detected', False)
                                fall_confidence = fall_result.get('confidence', 0.0)
                                fall_analysis = fall_result
                                
                                if fall_detected:
                                    self.stats['fall_detections'] += 1
                                    
                        except Exception as e:
                            print(f"⚠️ Fall detection error: {e}")
                            fall_analysis = {'error': str(e)}
                
                # Stage 4: Healthcare Analysis
                health_analysis = self.healthcare_analyzer.analyze_frame(frame, detections)
                
                # Add fall detection to health analysis if detected
                if fall_detected:
                    if 'alerts' not in health_analysis:
                        health_analysis['alerts'] = []
                    health_analysis['alerts'].append({
                        'type': 'fall_detected',
                        'confidence': fall_confidence,
                        'severity': 'high',
                        'message': f'Fall detected with {fall_confidence:.1%} confidence'
                    })
                
                # Count persons and alerts
                person_count = health_analysis.get('person_count', 0)
                alerts = health_analysis.get('alerts', [])
                
                if person_count > 0:
                    self.stats['persons_detected'] += 1
                
                if alerts:
                    self.stats['alerts_generated'] += len(alerts)
                
                # Save frames if enabled
                if self.save_frames and self.frame_saver:
                    
                    # Save keyframe
                    if save_keyframes and keyframe_confidence > 0.5:
                        self.frame_saver.save_keyframe(frame, {
                            'confidence': keyframe_confidence,
                            'motion_pixels': motion_result['motion_pixels'],
                            'detections': len(detections),
                            'person_count': person_count
                        })
                    
                    # Save detection frame if persons detected
                    if person_count > 0:
                        max_conf = max([d.get('confidence', 0) for d in detections], default=0)
                        self.frame_saver.save_detection(frame, {
                            'persons': [d for d in detections if d.get('class_name') == 'person'],
                            'max_confidence': max_conf,
                            'keyframe_confidence': keyframe_confidence
                        })
                    
                    # Save alert frame if alerts generated
                    if alerts:
                        for alert in alerts:
                            self.frame_saver.save_alert(frame, {
                                'alert_type': alert.get('type', 'unknown'),
                                'confidence': alert.get('confidence', 0),
                                'keyframe_confidence': keyframe_confidence
                            })
                
                return {
                    'processed': True,
                    'motion_detected': True,
                    'is_keyframe': True,
                    'keyframe_confidence': keyframe_confidence,
                    'detections': detections,
                    'health_analysis': health_analysis,
                    'person_count': person_count,
                    'alerts': alerts,
                    'fall_detected': fall_detected,
                    'fall_confidence': fall_confidence,
                    'fall_analysis': fall_analysis,
                    'processing_stats': self.get_processing_stats()
                }
            
            else:
                # Motion detected but not a keyframe
                return {
                    'processed': False,
                    'motion_detected': True,
                    'is_keyframe': False,
                    'keyframe_confidence': keyframe_confidence,
                    'reason': 'Not a keyframe',
                    'processing_stats': self.get_processing_stats()
                }
        
        else:
            # No motion detected
            return {
                'processed': False,
                'motion_detected': False,
                'is_keyframe': False,
                'reason': 'No motion detected',
                'processing_stats': self.get_processing_stats()
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = max(self.stats['total_frames'], 1)
        
        stats = {
            'total_frames': self.stats['total_frames'],
            'motion_frames': self.stats['motion_frames'],
            'keyframes': self.stats['keyframes'],
            'yolo_processed': self.stats['yolo_processed'],
            'persons_detected': self.stats['persons_detected'],
            'alerts_generated': self.stats['alerts_generated'],
            
            'motion_rate': self.stats['motion_frames'] / total,
            'keyframe_rate': self.stats['keyframes'] / total,
            'yolo_rate': self.stats['yolo_processed'] / total,
            'person_detection_rate': self.stats['persons_detected'] / total,
            'processing_efficiency': f"{(1 - self.stats['yolo_processed'] / total) * 100:.1f}% frames skipped"
        }
        
        # Add fall detection stats if available
        if FALL_DETECTION_AVAILABLE:
            stats['fall_detections'] = self.stats['fall_detections']
            stats['fall_detection_rate'] = self.stats['fall_detections'] / total
            
        return stats
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get detailed component statistics"""
        return {
            'motion_detector': self.motion_detector.get_stats() if hasattr(self.motion_detector, 'get_stats') else {},
            'keyframe_detector': self.keyframe_detector.get_stats(),
            'yolo_detector': self.yolo_detector.get_stats() if hasattr(self.yolo_detector, 'get_stats') else {},
            'healthcare_analyzer': {},
            'processing_stats': self.get_processing_stats()
        }


# Aliases for compatibility
MotionDetector = SimpleMotionDetector
YOLODetector = SimpleYOLODetector  
VideoProcessor = SimpleVideoProcessor
HealthcareAnalyzer = SimpleHealthcareAnalyzer
KeyframeDetector = SimpleKeyframeDetector
FrameSaver = SimpleFrameSaver
