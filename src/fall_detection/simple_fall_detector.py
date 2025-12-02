"""
Simple Fall Detection for Healthcare Monitoring System
Simplified version without complex dependencies
"""
import logging
import time
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

class SimpleFallDetector:
    """
    Simplified fall detection component for healthcare monitoring.
    Uses lightweight approach without AI models.
    """
    
    def __init__(self, confidence_threshold=0.28):  # CÂN BẰNG 0.20→0.35→0.28: vừa detect được vừa tránh false positive
        """
        Initialize simplified fall detector.
        
        Args:
            confidence_threshold: Minimum confidence for fall detection (0.28 = balanced sweet spot)
        """
        self.confidence_threshold = confidence_threshold
        self.previous_frame = None
        self.previous_timestamp = None
        self.min_time_interval = 0.15  # Giảm từ 0.2→0.15s: yêu cầu rơi NHANH trong thời gian ngắn
        self.frame_buffer = []
        self.max_buffer_size = 5  # Giảm từ 7→5: chỉ giữ 5 frames (0.15s) để detect rapid movement
        
        # 🔥 NEW: Fall velocity tracking for stroke detection
        self.fall_start_time = None  # Thời điểm bắt đầu té
        self.fall_start_position = None  # Vị trí Y khi bắt đầu té
        self.fall_velocity_history = []  # Lịch sử vận tốc rơi
        
        # 🚨 DANGER EVENT COOLDOWN: Tránh spam fall events liên tiếp
        self.last_danger_fall_time = 0  # Thời điểm fall event DANGER cuối cùng
        self.danger_cooldown = 30  # 30 giây cooldown cho DANGER events (conf >= 0.60)
        
        log.info(f"🩺 Simplified fall detector initialized (confidence: {confidence_threshold})")
    
    def detect_fall(self, current_frame, timestamp=None, person_bbox=None, motion_level=None):
        """
        Detect fall in current frame using simplified approach.
        
        Args:
            current_frame: Current video frame (numpy array)
            timestamp: Frame timestamp (optional)
            person_bbox: Person bounding box from YOLO (optional)
            motion_level: Global motion level (0.0-1.0) to filter bbox jitter (optional)
            
        Returns:
            dict: Fall detection result
        """
        start_time = time.time()
        
        # Default result
        result = {
            'fall_detected': False,
            'confidence': 0.0,
            'angle': 0.0,
            'category': 'no-fall',
            'processing_time': 0.0,
            'method': 'simplified'
        }
        
        try:
            # Handle timestamp
            if timestamp is not None:
                if hasattr(timestamp, '__iter__') and not isinstance(timestamp, (str, bytes)):
                    try:
                        current_time = float(list(timestamp)[0])
                    except (ValueError, IndexError, TypeError):
                        current_time = time.time()
                else:
                    try:
                        current_time = float(timestamp)
                    except (ValueError, TypeError):
                        current_time = time.time()
            else:
                current_time = time.time()
            
            # Add frame to buffer
            safe_bbox = self._safe_bbox_conversion(person_bbox)
            
            frame_data = {
                'frame': current_frame,
                'timestamp': current_time,
                'bbox': safe_bbox
            }
            
            self.frame_buffer.append(frame_data)
            
            # Keep buffer size manageable
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
            
            # Check if we have enough frames and time interval
            if len(self.frame_buffer) >= 2:
                try:
                    first_timestamp = self.frame_buffer[0]['timestamp']
                    if hasattr(first_timestamp, '__iter__') and not isinstance(first_timestamp, (str, bytes)):
                        first_timestamp = float(list(first_timestamp)[0])
                    else:
                        first_timestamp = float(first_timestamp)
                    
                    # Check time interval
                    if (current_time - first_timestamp) >= self.min_time_interval:
                        # Process simplified fall detection with motion_level
                        fall_result = self._analyze_movement_pattern(motion_level=motion_level)
                        
                        if fall_result:
                            result.update(fall_result)
                            # REMOVED: log.warning(f"🚨 Fall detected! Confidence: {result['confidence']:.2f}")
                            # Only log once per detection event, not every frame
                            log.debug(f"Fall analysis result: {result['confidence']:.2f}")
                except (ValueError, TypeError, KeyError) as time_error:
                    log.debug(f"Timestamp processing error: {time_error}")
                
        except Exception as e:
            log.error(f"Fall detection error: {e}")
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _safe_bbox_conversion(self, bbox):
        """
        Safely convert bbox to standard format [x1, y1, x2, y2].
        
        Args:
            bbox: Input bbox in various formats
            
        Returns:
            list or None: Safe bbox format or None if invalid
        """
        if bbox is None:
            return None
            
        try:
            # Handle different input types
            if hasattr(bbox, '__iter__') and not isinstance(bbox, str):
                bbox_list = list(bbox)
                
                # Ensure we have at least 4 elements
                if len(bbox_list) >= 4:
                    # Convert to floats and ensure they are valid numbers
                    safe_bbox = []
                    for i in range(4):
                        try:
                            val = float(bbox_list[i])
                            if not np.isnan(val) and not np.isinf(val):
                                safe_bbox.append(val)
                            else:
                                return None
                        except (ValueError, TypeError):
                            return None
                    
                    # Validate bbox coordinates make sense
                    x1, y1, x2, y2 = safe_bbox
                    if x2 > x1 and y2 > y1 and all(coord >= 0 for coord in safe_bbox):
                        return safe_bbox
                        
        except Exception as e:
            log.debug(f"Bbox conversion error: {e}")
            
        return None
    
    def _analyze_movement_pattern(self, motion_level=None):
        """
        Analyze movement pattern for fall detection using simplified heuristics.
        
        Args:
            motion_level: Global motion level (0.0-1.0) to filter bbox jitter
        
        Returns:
            dict or None: Fall detection result
        """
        if len(self.frame_buffer) < 2:
            return None
            
        try:
            # Get first and last frames
            first_frame = self.frame_buffer[0]
            last_frame = self.frame_buffer[-1]
            
            # Simplified fall detection based on bbox changes
            if (first_frame['bbox'] is not None and 
                last_frame['bbox'] is not None):
                
                return self._analyze_bbox_changes(first_frame['bbox'], last_frame['bbox'], motion_level)
            
            # Fallback to frame difference analysis
            return self._analyze_frame_difference(first_frame['frame'], last_frame['frame'])
            
        except Exception as e:
            log.error(f"Movement analysis error: {e}")
            return None
    
    def _analyze_bbox_changes(self, bbox1, bbox2, motion_level=None):
        """
        Analyze bounding box changes to detect falls.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            motion_level: Global motion level (0.0-1.0) to filter bbox jitter
            
        Returns:
            dict or None: Fall result
        """
        try:
            # Use safe conversion for both bboxes
            safe_bbox1 = self._safe_bbox_conversion(bbox1)
            safe_bbox2 = self._safe_bbox_conversion(bbox2)
            
            if safe_bbox1 is None or safe_bbox2 is None:
                log.debug(f"Bbox conversion failed: bbox1={safe_bbox1}, bbox2={safe_bbox2}")
                return None
            
            # Convert to numpy arrays for safe arithmetic operations
            bbox1_arr = np.array(safe_bbox1, dtype=np.float64)
            bbox2_arr = np.array(safe_bbox2, dtype=np.float64)
            
            # Calculate dimensions safely
            w1 = bbox1_arr[2] - bbox1_arr[0]
            h1 = bbox1_arr[3] - bbox1_arr[1]
            w2 = bbox2_arr[2] - bbox2_arr[0]
            h2 = bbox2_arr[3] - bbox2_arr[1]
            
            # Validate dimensions
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                return None
            
            # Calculate aspect ratios
            aspect_ratio1 = w1 / h1
            aspect_ratio2 = w2 / h2
            
            # Calculate center positions
            center1_x = (bbox1_arr[0] + bbox1_arr[2]) / 2
            center2_x = (bbox2_arr[0] + bbox2_arr[2]) / 2
            center1_y = (bbox1_arr[1] + bbox1_arr[3]) / 2
            center2_y = (bbox2_arr[1] + bbox2_arr[3]) / 2
            
            # Fall detection criteria
            aspect_change = aspect_ratio2 / aspect_ratio1 if aspect_ratio1 > 0 else 1.0
            vertical_movement = abs(center2_y - center1_y)
            horizontal_movement = abs(center2_x - center1_x)
            
            # 🔍 AGGRESSIVE DEBUG LOGGING - Log mọi frame có movement để debug
            if vertical_movement > 30 or aspect_change > 1.2:  # Log khi có thay đổi đáng kể
                log.info(f"📊 FALL CHECK: aspect_change={aspect_change:.2f}x (need >1.25 moderate / >1.6 dynamic), vertical={vertical_movement:.1f}px (need >65 moderate / >60 dynamic)")
                log.info(f"   🔄 Movement: horizontal={horizontal_movement:.1f}px, vertical={vertical_movement:.1f}px, ratio={vertical_movement/(horizontal_movement+1):.2f}")
                log.info(f"   Aspect: {aspect_ratio1:.2f} → {aspect_ratio2:.2f}")
                log.info(f"   Position: center_x {center1_x:.1f}→{center2_x:.1f}, center_y {center1_y:.1f}→{center2_y:.1f}")
                log.info(f"   Bbox1: w={bbox1_arr[2]-bbox1_arr[0]:.1f} h={bbox1_arr[3]-bbox1_arr[1]:.1f}")
                log.info(f"   Bbox2: w={bbox2_arr[2]-bbox2_arr[0]:.1f} h={bbox2_arr[3]-bbox2_arr[1]:.1f}")
                
                # Log kết quả check
                if aspect_change > 1.25 and vertical_movement > 65 and center2_y > center1_y:
                    log.info(f"   ✅ PASS moderate thresholds - calculating confidence...")
                elif aspect_change > 1.6 and vertical_movement > 60 and center2_y > center1_y:
                    log.info(f"   ✅ PASS dynamic thresholds - calculating confidence...")
                else:
                    reasons = []
                    if aspect_change <= 1.25:
                        reasons.append(f"aspect_change {aspect_change:.2f} <= 1.25 (need >1.25 moderate / >1.6 dynamic)")
                    if vertical_movement <= 60:
                        reasons.append(f"vertical {vertical_movement:.1f} <= 60-65px")
                    if center2_y <= center1_y:
                        reasons.append(f"not moving down")
                    log.info(f"   ❌ FAIL: {', '.join(reasons)}")
            
            # STRATEGY 0: RAPID DOWNWARD MOVEMENT (person falling/dropping)
            # Detect large vertical movement downward - HIGHEST PRIORITY!
            # 🎯 CÂN BẰNG: 100px để phân biệt di chuyển thường vs TÉ NGÃ
            
            # 🔍 Log STRATEGY 0 check
            if vertical_movement > 70:  # Log khi gần threshold
                log.info(f"🔍 STRATEGY 0 CHECK: vertical={vertical_movement:.1f}px (need >80), horizontal={horizontal_movement:.1f}px, downward={center2_y > center1_y}")
            
            if vertical_movement > 80 and center2_y > center1_y:  # GIẢM 100→80px: NHẠY HƠN để detect fall thật
                # 🚫 HORIZONTAL MOVEMENT FILTER: Reject WALKING/MOVING ACROSS
                # Nếu horizontal > vertical = người đi ngang, KHÔNG PHẢI TÉ NGÃ!
                # Té ngã thật: vertical >> horizontal (rơi xuống dưới)
                # Đi ngang: horizontal >> vertical (di chuyển ngang qua camera)
                movement_ratio = vertical_movement / (horizontal_movement + 1)  # +1 tránh chia 0
                
                if horizontal_movement > vertical_movement * 0.8:  # Horizontal > 80% vertical = đi ngang
                    log.info(f"🚶 Rejected WALKING: horizontal={horizontal_movement:.1f}px > vertical={vertical_movement:.1f}px * 0.8 (ratio={movement_ratio:.2f}) - Person walking across, not falling")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'walking-across',
                        'method': 'horizontal_movement_filtered'
                    }
                
                # 🕐 GET CURRENT TIME for cooldown check
                current_time = time.time()
                
                # 🚫 POSTURE FILTER: Reject if person is ALREADY LYING DOWN (not falling)
                # Initial aspect > 1.3 = person lying horizontally (width > height)
                # Only detect fall from STANDING/SITTING → LYING, not LYING → LYING movement
                is_initially_lying = aspect_ratio1 > 1.3
                
                if is_initially_lying:
                    log.info(f"⚠️ Rejected ALREADY LYING: initial_aspect={aspect_ratio1:.2f} > 1.3 (person already on ground, not falling)")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'already-lying',
                        'method': 'already_lying_filtered'
                    }
                
                # 🚨 DANGER COOLDOWN CHECK: Tránh spam DANGER fall events
                # Chỉ cho phép 1 DANGER event mỗi 30 giây
                time_since_last_danger = current_time - self.last_danger_fall_time
                if time_since_last_danger < self.danger_cooldown:
                    log.info(f"⏰ Rejected DANGER COOLDOWN: last_danger={time_since_last_danger:.1f}s ago (need >{self.danger_cooldown}s)")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'danger-cooldown',
                        'method': 'danger_cooldown_filtered'
                    }
                
                # 🪑 SITTING FILTER: Reject NGỒI NHANH (sitting down quickly)
                # Nếu vị trí cuối (center2_y) ở giữa frame = NGỒI, không phải TÉ NGÃ
                # Frame height = 480px → NGỒI thường ở y=200-350px (40-70% chiều cao)
                # TÉ NGÃ thật → người nằm sàn → y > 350px (>70% chiều cao)
                frame_height = 480  # Assumed frame height
                final_position_ratio = center2_y / frame_height
                
                if final_position_ratio < 0.70:  # Vị trí cuối < 70% = NGỒI hoặc ĐỨNG
                    log.info(f"🪑 Rejected SITTING: final_y={center2_y:.1f}px ({final_position_ratio:.1%} < 70% frame) - Person sitting, not falling")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'sitting-down',
                        'method': 'sitting_filtered'
                    }
                
                # 🧎 BENDING DETECTION: CÚI NGƯỜI (bending down) → WARNING
                # Người CÚI: aspect < 1.0 (vẫn đứng, height > width) → confidence thấp
                # Người TÉ: aspect >= 1.0 (nằm ngang, width >= height) → confidence cao
                if aspect_ratio2 < 1.0:  # Tư thế cuối vẫn ĐỨNG/CÚI
                    bending_confidence = 0.45  # WARNING level (0.40-0.59)
                    log.warning(f"🧎 BENDING DETECTED: final_aspect={aspect_ratio2:.2f} < 1.0 (bending, not lying) - WARNING level, conf={bending_confidence}")
                    
                    # Update last danger time để tránh spam
                    self.last_danger_fall_time = current_time
                    
                    return {
                        'fall_detected': True,
                        'confidence': bending_confidence,
                        'angle': 0.0,
                        'category': 'bending-posture',
                        'method': 'bending_warning',
                        'fall_type': 'bending_down',
                        'alert_level': 'warning'  # Không phải CRITICAL
                    }
                
                # 🔥 FILTER BBOX JITTER: Reject if motion_level is very low (person motionless)
                # When person is laying still, bbox can jitter 100-200px due to detection variance
                # Only allow rapid fall detection if there's actual motion in the scene
                motion_str = f"{motion_level:.3f}" if motion_level is not None else "None"
                if motion_level is not None and motion_level < 0.05:  # Giảm 0.15→0.05: chỉ reject jitter thực sự
                    log.debug(f"⚠️ Rejected bbox jitter: vertical_movement={vertical_movement:.1f}px but motion_level={motion_str} too low")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'no-fall',
                        'method': 'bbox_jitter_filtered'
                    }
                
                # 🔥 NEW: FALL VELOCITY ANALYSIS (Phân tích tốc độ té)
                # Note: current_time already defined above for cooldown check
                
                # Bắt đầu tracking nếu chưa có
                if self.fall_start_time is None:
                    self.fall_start_time = current_time
                    self.fall_start_position = center1_y
                    log.debug(f"⏱️ Fall tracking started: position={center1_y:.1f}px")
                
                # Tính fall duration và fall velocity
                fall_duration = current_time - self.fall_start_time
                total_fall_distance = center2_y - self.fall_start_position
                
                # Tính vận tốc rơi (pixels/second)
                if fall_duration > 0:
                    fall_velocity = total_fall_distance / fall_duration
                else:
                    fall_velocity = 0
                
                # Phân loại loại té dựa trên duration
                fall_type = "unknown"
                severity_multiplier = 1.0
                
                if fall_duration < 0.5:
                    fall_type = "fast_fall"  # TÉ NHANH - Fall bình thường
                    severity_multiplier = 1.0
                    log.info(f"⚡ FAST FALL DETECTED: duration={fall_duration:.2f}s, velocity={fall_velocity:.1f}px/s")
                elif fall_duration >= 1.0:
                    fall_type = "slow_collapse"  # TÉ CHẬM - Đột quỵ/yếu sức
                    severity_multiplier = 1.3  # Tăng severity vì nguy hiểm hơn!
                    log.warning(f"🏥 SLOW COLLAPSE (Possible Stroke): duration={fall_duration:.2f}s, velocity={fall_velocity:.1f}px/s")
                else:
                    fall_type = "moderate_fall"
                    severity_multiplier = 1.1
                    log.info(f"⚠️ MODERATE FALL: duration={fall_duration:.2f}s, velocity={fall_velocity:.1f}px/s")
                
                downward_confidence = min(0.9, 0.50 + (vertical_movement / 180))  # GIẢM base 0.55→0.50
                downward_confidence *= severity_multiplier  # Điều chỉnh theo loại té
                downward_confidence = min(0.95, downward_confidence)  # Cap ở 0.95
                
                if downward_confidence >= 0.50:  # GIẢM threshold 0.60→0.50 để dễ detect
                    log.warning(f"🚨 RAPID FALL DETECTED: type={fall_type}, vertical={vertical_movement:.1f}px, duration={fall_duration:.2f}s, motion={motion_str}, conf={downward_confidence:.3f}")
                    
                    # 🚨 UPDATE LAST DANGER TIME: Bắt đầu cooldown 30s
                    self.last_danger_fall_time = current_time
                    log.info(f"⏰ DANGER cooldown started: next event allowed after {self.danger_cooldown}s")
                    
                    # Reset tracking sau khi detect
                    self.fall_start_time = None
                    self.fall_start_position = None
                    
                    return {
                        'fall_detected': True,
                        'confidence': downward_confidence,
                        'angle': 60.0,
                        'category': 'fall',
                        'method': 'rapid_downward',
                        'fall_type': fall_type,  # NEW: Loại té
                        'fall_duration': fall_duration,  # NEW: Thời gian té
                        'fall_velocity': fall_velocity  # NEW: Vận tốc té
                    }
                else:
                    log.info(f"⚠️ RAPID FALL LOW CONFIDENCE: vertical={vertical_movement:.1f}px, conf={downward_confidence:.3f} < 0.50")
            else:
                # Reset tracking nếu không có vertical movement
                if self.fall_start_time is not None:
                    elapsed = time.time() - self.fall_start_time
                    if elapsed > 2.0:  # Timeout sau 2s
                        log.debug(f"⏹️ Fall tracking reset (timeout after {elapsed:.1f}s)")
                        self.fall_start_time = None
                        self.fall_start_position = None
            
            # STRATEGY 0.5: MODERATE FALL - Cân bằng giữa nhạy và chính xác
            # 🎯 BALANCED: Detect fall thật nhưng TRÁNH NGỐI XUỐNG
            # Kiểm tra: aspect tăng nhiều (>1.25) + vertical đủ lớn (>65px)
            if (vertical_movement > 65 and  # TĂNG 50→65px: tránh ngồi xuống
                aspect_change > 1.25 and  # TĂNG 1.1→1.25: người PHẢI nằm ngang thật sự
                center2_y > center1_y and  # Moving downward
                horizontal_movement < vertical_movement * 1.2):  # Vertical phải lớn hơn horizontal
                
                # 🚫 FILTER NGỐI XUỐNG: Nếu aspect không tăng nhiều lắm = chỉ ngồi
                if aspect_change < 1.35:  # Ngồi xuống thường aspect ~1.2-1.3x
                    log.info(f"⚠️ Rejected SITTING: vertical={vertical_movement:.1f}px, aspect={aspect_change:.2f}x < 1.35 (likely sitting down)")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'sitting',
                        'method': 'sitting_filtered'
                    }
                
                confidence = min(0.85, 0.55 + (vertical_movement / 100) * 0.2 + (aspect_change - 1.25) * 0.15)
                
                if confidence >= 0.55:  # TĂNG threshold 0.50→0.55
                    log.warning(f"🚨 MODERATE FALL DETECTED: vertical={vertical_movement:.1f}px, aspect={aspect_change:.2f}x, conf={confidence:.3f}")
                    return {
                        'fall_detected': True,
                        'confidence': confidence,
                        'angle': 50.0,
                        'category': 'fall',
                        'method': 'moderate_fall',
                        'fall_type': 'moderate_fall'
                    }
            
            # STRATEGY 1: Dynamic Fall Detection - Person transitioning from standing to lying
            # 🎯 STRICT THRESHOLDS: Chỉ detect fall rõ ràng (aspect tăng nhiều)
            # - Aspect ratio 1.6x: Cân bằng giữa nhạy và chính xác
            # - Vertical 60px: Rơi xuống đáng kể
            if (aspect_change > 1.6 and  # TĂNG 1.5→1.6: chặt hơn để tránh false positive
                vertical_movement > 60 and  # TĂNG 55→60px: đảm bảo rơi thật sự
                center2_y > center1_y and  # Moving downward
                horizontal_movement < vertical_movement * 1.5):  # Vertical phải dominant
                
                confidence = min(0.9, 0.60 + (aspect_change - 1.7) * 0.35 + min(vertical_movement / 140, 0.28))  # CÂN BẰNG base
                
                if confidence >= self.confidence_threshold:
                    log.warning(f"🚨 DYNAMIC FALL: aspect_change={aspect_change:.2f}, vertical_movement={vertical_movement:.1f}px, conf={confidence:.3f}")
                    return {
                        'fall_detected': True,
                        'confidence': confidence,
                        'angle': 90.0 - (45.0 / aspect_change),  # Estimated angle
                        'category': 'fall',
                        'method': 'bbox_analysis_dynamic'
                    }
                else:
                    log.debug(f"❌ DYNAMIC FALL REJECTED: confidence {confidence:.3f} < threshold {self.confidence_threshold:.2f}")
            
            # STRATEGY 2: Static Lying Position Detection - Person already on floor
            # 🎯 RE-ENABLED với điều kiện CỰC KỲ CHẶT CHẼ để tránh false positive
            # Phân biệt: ĐANG TÉ (strategy 1) vs ĐÃ NẰM (strategy 2)
            
            bbox_width = bbox2_arr[2] - bbox2_arr[0]
            bbox_height = bbox2_arr[3] - bbox2_arr[1]
            frame_bottom = bbox2_arr[3]
            
            # 🔥 ĐIỀU KIỆN CỰC KỲ NGHIÊM NGẶT để tránh false positive:
            # 1. Aspect ratio > 2.5 (nằm RẤT NGANG, không phải người đứng)
            # 2. Frame bottom > 480 (ở SÁT ĐÁY màn hình, không phải giữa/trên)
            # 3. Bbox height < 120px (RẤT THẤP, chắc chắn nằm)
            # 4. Confidence > 0.75 (CHẮC CHẮN mới báo)
            
            is_very_horizontal = aspect_ratio2 > 2.5  # TĂNG từ 2.0 → 2.5
            is_at_bottom = frame_bottom > 480  # TĂNG từ 450 → 480
            is_very_short = bbox_height < 120  # GIẢM từ 150 → 120
            
            log.debug(f"🔍 STATIC CHECK: aspect={aspect_ratio2:.2f} (need >2.5), bottom={frame_bottom:.1f} (need >480), height={bbox_height:.1f} (need <120)")
            
            if is_very_horizontal and is_at_bottom and is_very_short:
                # Calculate confidence based on multiple factors
                horizontal_factor = min(0.5, (aspect_ratio2 - 2.5) * 0.4)  # Bắt đầu từ 2.5
                position_factor = min(0.3, (frame_bottom - 480) / 80)  # Bắt đầu từ 480
                height_factor = min(0.2, (120 - bbox_height) / 120)  # Càng thấp càng chắc
                
                horizontal_confidence = horizontal_factor + position_factor + height_factor
                
                # 🎯 THRESHOLD CỰC KỲ CAO: 0.75 (chỉ báo khi CHẮC CHẮN)
                if horizontal_confidence >= 0.75:
                    log.warning(f"🚨 STATIC LYING DETECTED: aspect={aspect_ratio2:.2f}, bottom={frame_bottom:.1f}, height={bbox_height:.1f}, conf={horizontal_confidence:.3f}")
                    return {
                        'fall_detected': True,
                        'confidence': min(0.95, horizontal_confidence),  # Cap at 0.95
                        'angle': 90.0 - (45.0 / max(aspect_ratio2, 0.5)),
                        'category': 'fall',
                        'method': 'bbox_analysis_static',
                        'fall_type': 'static_lying',  # 🎯 PHÂN LOẠI: người đã nằm sẵn
                        'fall_duration': 0.0,  # Không có thời gian rơi
                        'fall_velocity': 0.0  # Không có vận tốc
                    }
                else:
                    log.debug(f"❌ STATIC LYING REJECTED: confidence too low ({horizontal_confidence:.3f} < 0.75)")
            else:
                log.debug(f"❌ STATIC LYING REJECTED: conditions not met (horizontal={is_very_horizontal}, bottom={is_at_bottom}, short={is_very_short})")
                    
        except Exception as e:
            log.error(f"Bbox analysis error: {e}")
            
        return None
    
    def _analyze_frame_difference(self, frame1, frame2):
        """
        Analyze frame differences for fall detection.
        
        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)
            
        Returns:
            dict or None: Fall result
        """
        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = np.mean(frame1, axis=2).astype(np.uint8)
            else:
                gray1 = frame1
                
            if len(frame2.shape) == 3:
                gray2 = np.mean(frame2, axis=2).astype(np.uint8)
            else:
                gray2 = frame2
                
            # Calculate frame difference
            diff = np.abs(gray2.astype(np.float32) - gray1.astype(np.float32))
            
            # Analyze movement patterns
            movement_intensity = np.mean(diff)
            horizontal_movement = np.mean(np.abs(np.diff(diff, axis=1)))
            vertical_movement = np.mean(np.abs(np.diff(diff, axis=0)))
            
            # Simple fall detection heuristic
            movement_ratio = horizontal_movement / (vertical_movement + 1e-6)
            
            if (movement_intensity > 15 and  # Significant movement
                movement_ratio > 1.3):  # More horizontal than vertical movement
                
                confidence = min(0.85, movement_intensity / 50 + movement_ratio / 5)
                
                if confidence >= self.confidence_threshold:
                    return {
                        'fall_detected': True,
                        'confidence': confidence,
                        'angle': 60.0 + movement_ratio * 10,  # Estimated angle
                        'category': 'fall',
                        'method': 'frame_difference'
                    }
                    
        except Exception as e:
            log.error(f"Frame difference analysis error: {e}")
            
        return None
    
    def reset(self):
        """Reset detector state."""
        self.frame_buffer.clear()
        self.previous_frame = None
        self.previous_timestamp = None
        log.debug("Fall detector state reset")
    
    def get_stats(self):
        """Get detector statistics."""
        return {
            'confidence_threshold': self.confidence_threshold,
            'min_time_interval': self.min_time_interval,
            'buffer_size': len(self.frame_buffer),
            'max_buffer_size': self.max_buffer_size
        }
