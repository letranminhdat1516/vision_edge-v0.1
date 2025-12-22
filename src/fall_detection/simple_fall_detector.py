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
    
    def __init__(self, confidence_threshold=0.40):  # TĂNG 0.28→0.40: giảm false positive khi ngồi xuống
        """
        Initialize simplified fall detector.
        
        Args:
            confidence_threshold: Minimum confidence for fall detection (0.40 = balanced sweet spot)
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
        self.danger_cooldown = 15  # 🔧 FIX v3: 40s→15s cooldown - nếu có ngã thứ 2 ngay sau thì vẫn detect được
        
        # 🧍 STANDING UP COOLDOWN: Sau khi đứng dậy, block fall detection 3s
        # Vì sau đứng dậy, người có thể cúi/nghiêng nhẹ bị detect nhầm là fall
        self.last_standing_up_time = 0
        self.standing_up_cooldown = 3  # 3 giây cooldown sau khi đứng dậy
        
        # 🪑 REPEATED SITTING PATTERN: Phát hiện ngồi-đứng liên tục (squat exercise)
        self.sitting_events = []  # [(timestamp, position_y), ...]
        self.sitting_pattern_window = 10  # 10 giây window
        self.sitting_pattern_threshold = 3  # 3 lần ngồi-đứng trong 10s = exercise
        
        # 🎯 STATE MACHINE: Track person posture state
        self.person_state = "UNKNOWN"  # STANDING, SITTING, LYING, UNKNOWN
        self.lying_start_time = None  # Thời điểm bắt đầu nằm
        self.state_change_time = None  # Thời điểm thay đổi state cuối cùng
        
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
                
                frame_h = first_frame.get('frame', {}).shape[0] if first_frame.get('frame') is not None else 480
                return self._analyze_bbox_changes(first_frame['bbox'], last_frame['bbox'], motion_level, frame_h)
            
            # Fallback to frame difference analysis
            return self._analyze_frame_difference(first_frame['frame'], last_frame['frame'])
            
        except Exception as e:
            log.error(f"Movement analysis error: {e}")
            return None
    
    def _analyze_bbox_changes(self, bbox1, bbox2, motion_level=None, frame_height=480):
        """
        Analyze bounding box changes to detect falls.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            motion_level: Global motion level (0.0-1.0) to filter bbox jitter
            frame_height: Frame height in pixels (default 480)
            
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
            
            # 🔍 DEBUG - Chỉ log khi có thay đổi đáng kể
            if vertical_movement > 30 or aspect_change > 1.2:
                log.info(f"📊 FALL CHECK: aspect_change={aspect_change:.2f}x (need >1.15 moderate / >1.6 dynamic), vertical={vertical_movement:.1f}px (need >50 moderate / >45 dynamic)")
                log.info(f"   🔄 Movement: horizontal={horizontal_movement:.1f}px, vertical={vertical_movement:.1f}px, ratio={vertical_movement/(horizontal_movement+1):.2f}")
                log.info(f"   Aspect: {aspect_ratio1:.2f} → {aspect_ratio2:.2f}")
                log.info(f"   Position: center_x {center1_x:.1f}→{center2_x:.1f}, center_y {center1_y:.1f}→{center2_y:.1f}")
                log.info(f"   Bbox1: w={bbox1_arr[2]-bbox1_arr[0]:.1f} h={bbox1_arr[3]-bbox1_arr[1]:.1f}")
                log.info(f"   Bbox2: w={bbox2_arr[2]-bbox2_arr[0]:.1f} h={bbox2_arr[3]-bbox2_arr[1]:.1f}")
                
                # Log kết quả check
                if aspect_change > 1.15 and vertical_movement > 50 and center2_y > center1_y:
                    log.info(f"   ✅ PASS moderate thresholds - calculating confidence...")
                elif aspect_change > 1.6 and vertical_movement > 45 and center2_y > center1_y:
                    log.info(f"   ✅ PASS dynamic thresholds - calculating confidence...")
                else:
                    reasons = []
                    if aspect_change <= 1.15:
                        reasons.append(f"aspect_change {aspect_change:.2f} <= 1.15 (need >1.15 moderate / >1.6 dynamic)")
                    if vertical_movement <= 45:
                        reasons.append(f"vertical {vertical_movement:.1f} <= 45-50px")
                    if center2_y <= center1_y:
                        reasons.append(f"not moving down")
                    log.info(f"   ❌ FAIL: {', '.join(reasons)}")
            
            # 🚶 PRIORITY CHECK 1: STANDING UP DETECTION (người đứng dậy từ sàn)
            # Phải check TRƯỚC rapid fall để tránh nhầm "đứng dậy" thành "rơi xuống"
            # Standing up: vertical lớn (>300px) NHƯNG đi LÊN (center2_y < center1_y)
            is_moving_upward = center2_y < center1_y
            has_large_upward_movement = vertical_movement > 300 and is_moving_upward
            
            if has_large_upward_movement:
                log.info(f"🧍 Rejected STANDING UP: vertical={vertical_movement:.1f}px upward (center_y: {center1_y:.1f}→{center2_y:.1f})")
                log.info(f"   Person standing up from floor/chair, not falling")
                
                # Reset fall tracking khi đứng dậy
                self.fall_start_time = None
                self.fall_start_position = None
                
                # 🧍 UPDATE STANDING UP COOLDOWN
                self.last_standing_up_time = time.time()
                log.info(f"⏰ STANDING UP cooldown started: blocking fall detection for {self.standing_up_cooldown}s")
                
                return {
                    'fall_detected': False,
                    'confidence': 0.0,
                    'angle': 0.0,
                    'category': 'standing-up',
                    'method': 'standing_up_filtered'
                }
            
            # 🧘 PRIORITY CHECK 2: SMALL POSTURE ADJUSTMENT (điều chỉnh tư thế nhỏ)
            # Movement nhỏ (<60px) thường là cúi người, xê dịch tư thế, KHÔNG phải té ngã thật
            # 🔧 TĂNG 40px → 60px: Giảm false positive khi ngồi xuống
            # 
            # ⚠️ BYPASS: TÉ NGANG (sideways fall) có vertical nhỏ nhưng horizontal + aspect change lớn
            # Điều kiện bypass: horizontal > 40px + aspect_change > 1.2 + final_aspect > 1.4
            is_moving_downward = center2_y > center1_y
            has_small_downward_movement = vertical_movement < 60 and is_moving_downward
            
            # 🔥 CHECK FOR SIDEWAYS FALL PATTERN - BYPASS posture adjustment filter
            is_sideways_fall_pattern = (
                horizontal_movement > 40 and  # Horizontal movement đáng kể
                aspect_change > 1.2 and  # Aspect tăng đáng kể (đổi tư thế)
                aspect_ratio2 > 1.4  # Kết thúc ở tư thế nằm ngang
            )
            
            if has_small_downward_movement and not is_sideways_fall_pattern:
                log.info(f"🧘 Rejected POSTURE ADJUSTMENT: vertical={vertical_movement:.1f}px downward (<40px threshold)")
                log.info(f"   Small movement: likely bending/adjusting posture, not falling")
                
                return {
                    'fall_detected': False,
                    'confidence': 0.0,
                    'angle': 0.0,
                    'category': 'posture-adjustment',
                    'method': 'small_movement_filtered'
                }
            
            # 🔥 EARLY CHECK: SIDEWAYS FALL DETECTION (TÉ NGANG)
            # Phải check SỚM trước các filter khác vì vertical nhỏ sẽ bị reject
            if is_sideways_fall_pattern:
                log.info(f"🔍 SIDEWAYS FALL CANDIDATE: horiz={horizontal_movement:.1f}px, vert={vertical_movement:.1f}px")
                log.info(f"   aspect: {aspect_ratio1:.2f} → {aspect_ratio2:.2f} (change={aspect_change:.2f}x)")
                
                # 🧍 STANDING UP COOLDOWN CHECK cho sideways fall
                current_time = time.time()
                time_since_standing_up = current_time - self.last_standing_up_time
                if time_since_standing_up < self.standing_up_cooldown:
                    log.info(f"⏰ Rejected SIDEWAYS FALL (STANDING UP COOLDOWN): {time_since_standing_up:.1f}s < {self.standing_up_cooldown}s")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'standing-up-cooldown',
                        'method': 'standing_up_cooldown_filtered'
                    }
                
                # 🚫 REJECT: Walking sideways - aspect không thay đổi nhiều + vẫn đứng
                is_walking_sideways = (aspect_change < 1.3 and aspect_ratio2 < 1.5 and vertical_movement < 15)
                
                if is_walking_sideways:
                    log.info(f"🚶 Rejected WALKING SIDEWAYS: aspect_change={aspect_change:.2f}x < 1.3, final_aspect={aspect_ratio2:.2f} < 1.5")
                else:
                    # 🚨 DANGER COOLDOWN CHECK cho sideways fall
                    current_time = time.time()
                    time_since_last_danger = current_time - self.last_danger_fall_time
                    
                    if time_since_last_danger < self.danger_cooldown:
                        log.info(f"⏰ Rejected SIDEWAYS FALL COOLDOWN: last_danger={time_since_last_danger:.1f}s ago (need >{self.danger_cooldown}s)")
                    else:
                        # Calculate confidence for sideways fall
                        sideways_conf = 0.55  # Base confidence
                        sideways_conf += min((aspect_change - 1.2) * 0.25, 0.15)  # Aspect change bonus
                        sideways_conf += min(horizontal_movement / 150, 0.15)  # Horizontal movement bonus
                        sideways_conf += min(vertical_movement / 80, 0.10)  # Some vertical bonus
                        sideways_conf = min(0.90, sideways_conf)  # Cap at 0.90
                        
                        if sideways_conf >= 0.50:
                            log.warning(f"🚨 SIDEWAYS FALL DETECTED (TÉ NGANG)!")
                            log.warning(f"   horizontal={horizontal_movement:.1f}px, vertical={vertical_movement:.1f}px")
                            log.warning(f"   aspect: {aspect_ratio1:.2f} → {aspect_ratio2:.2f} (change={aspect_change:.2f}x)")
                            log.warning(f"   confidence={sideways_conf:.3f}")
                            
                            # Update cooldown
                            self.last_danger_fall_time = current_time
                            log.info(f"⏰ DANGER cooldown started: next event allowed after {self.danger_cooldown}s")
                            
                            return {
                                'fall_detected': True,
                                'confidence': sideways_conf,
                                'angle': 90.0,  # Horizontal = 90 degrees
                                'category': 'fall',
                                'method': 'sideways_fall',
                                'fall_type': 'sideways_fall',
                                'fall_duration': 0.0,
                                'fall_velocity': horizontal_movement,
                                'alert_level': 'DANGER'
                            }
            
            # 🛌 PRIORITY CHECK 3: SLOW LYING DOWN (nằm từ từ xuống)
            # Detect controlled descent (ngồi xuống/nằm xuống có kiểm soát) vs rapid fall (té ngã)
            # 
            # KEY DIFFERENCE:
            # - TÉ THẬT: aspect GIẢM hoặc tăng nhẹ (0.8-1.1 = aspect_change 0.8-1.1)
            # - NẰM TỪ TỪ: aspect gần như KHÔNG ĐỔI (aspect_change ~1.0, range 0.95-1.05)
            # 
            # LOGIC: Chỉ reject nếu TẤT CẢ điều kiện sau:
            # 1. Final position gần sàn (>98% - rất sát sàn)
            # 2. Final aspect nằm ngang (>1.3 - nằm hoàn toàn)
            # 3. Aspect change GẦN 1.0 (0.95-1.05 = không đổi tư thế)  ← TIGHTENED!
            # 4. Vertical < 400px (di chuyển chậm)  ← GIẢM từ 600→400px
            has_large_downward_movement = vertical_movement >= 100 and is_moving_downward
            
            if has_large_downward_movement:
                # Tính vị trí cuối cùng so với frame height
                final_position_ratio = center2_y / frame_height
                final_aspect_ratio = aspect_ratio2
                
                # 🔥 TÍNH VELOCITY: Tốc độ rơi để phân biệt "té" vs "nằm từ từ"
                time_diff = 0.3  # Giả sử ~3 frames với 10 FPS = 0.3s
                vertical_velocity = abs(vertical_movement) / time_diff if time_diff > 0 else 0  # px/s
                
                # 🔥 LOGIC MỚI: Nới lỏng ngưỡng + thêm velocity check
                # Aspect change có thể dao động nhiều hơn khi người từ từ nằm xuống
                is_aspect_stable = 0.85 <= aspect_change <= 1.15  # Nới lỏng: 0.95-1.05 → 0.85-1.15
                
                # Check nếu người đang nằm xuống sàn CHẬM với tư thế ít thay đổi
                is_lying_down_pattern = (final_position_ratio > 0.90 and  # Giảm 0.98→0.90: Bắt sớm hơn
                                        final_aspect_ratio > 1.2 and      # Giảm 1.3→1.2: Cho phép chưa nằm hoàn toàn
                                        is_aspect_stable and              # Aspect thay đổi ít (0.85-1.15)
                                        vertical_movement < 600 and       # Tăng 400→600px: Cho phép di chuyển chậm hơn
                                        vertical_velocity < 1500)         # ⭐ THÊM: Tốc độ < 1500px/s = chậm, không phải té
                
                if is_lying_down_pattern:
                    log.info(f"🛌 Rejected LYING DOWN: vertical={vertical_movement:.1f}px, velocity={vertical_velocity:.0f}px/s, final_y={center2_y:.1f} ({final_position_ratio:.1%}), aspect={final_aspect_ratio:.2f}, aspect_change={aspect_change:.2f}x")
                    log.info(f"   Controlled descent to floor (lying down), not falling")
                    
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'lying-down',
                        'method': 'controlled_descent_filtered'
                    }
            
            # STRATEGY 0: RAPID DOWNWARD MOVEMENT (person falling/dropping)
            # Detect large vertical movement downward - HIGHEST PRIORITY!
            # 🎯 TĂNG 50→70px: Giảm false positive khi ngồi xuống
            
            # 🧍 CHECK STANDING UP COOLDOWN: Block fall detection ngay sau khi đứng dậy
            current_time = time.time()
            time_since_standing_up = current_time - self.last_standing_up_time
            if time_since_standing_up < self.standing_up_cooldown:
                log.info(f"⏰ Blocked by STANDING UP COOLDOWN: {time_since_standing_up:.1f}s < {self.standing_up_cooldown}s")
                log.info(f"   Person just stood up, ignoring small downward movements")
                return {
                    'fall_detected': False,
                    'confidence': 0.0,
                    'angle': 0.0,
                    'category': 'standing-up-cooldown',
                    'method': 'standing_up_cooldown_filtered'
                }
            
            # 🔍 Log STRATEGY 0 check
            if vertical_movement > 50:  # Log khi gần threshold
                log.info(f"🔍 STRATEGY 0 CHECK: vertical={vertical_movement:.1f}px (need >70), horizontal={horizontal_movement:.1f}px, downward={center2_y > center1_y}")
            
            if vertical_movement > 70 and center2_y > center1_y:  # TĂNG 50→70px: GIẢM FALSE POSITIVE
                # 🚫 HORIZONTAL MOVEMENT FILTER: Reject WALKING/MOVING ACROSS
                # Nếu horizontal > vertical = người đi ngang, KHÔNG PHẢI TÉ NGÃ!
                # Té ngã thật: vertical >> horizontal (rơi xuống dưới)
                # Đi ngang: horizontal >> vertical (di chuyển ngang qua camera)
                movement_ratio = vertical_movement / (horizontal_movement + 1)  # +1 tránh chia 0
                
                # 🚨 BYPASS WALKING filter nếu có vertical lớn (>150px)
                # Vertical > 150px = té ngã thật, cho dù có horizontal movement lớn
                has_significant_vertical = vertical_movement > 150 and center2_y > center1_y
                
                if horizontal_movement > vertical_movement * 0.8 and not has_significant_vertical:  # Horizontal > 80% vertical = đi ngang
                    log.info(f"🚶 Rejected WALKING: horizontal={horizontal_movement:.1f}px > vertical={vertical_movement:.1f}px * 0.8 (ratio={movement_ratio:.2f}) - Person walking across, not falling")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'walking-across',
                        'method': 'horizontal_movement_filtered'
                    }
                
                # 🔍 NEW FILTER: DEPTH MOVEMENT (di chuyển ra xa/gần camera)
                # Khi người di chuyển theo chiều sâu:
                # - Lùi xa camera: bbox nhỏ đi, center_y TĂNG (xuống dưới màn hình)
                # - Tiến lại gần: bbox lớn ra, center_y GIẢM (lên trên màn hình)
                # → vertical_movement lớn NHƯNG là di chuyển depth, KHÔNG PHẢI TÉ NGÃ!
                
                # Tính thay đổi size của bbox
                bbox_size1 = w1 * h1
                bbox_size2 = w2 * h2
                size_change_ratio = abs(bbox_size2 - bbox_size1) / (bbox_size1 + 1)  # % thay đổi size
                
                # 🔧 FIX v2: TĂNG threshold 80%→150% vì log cho thấy té thật có 97-122% size change bị reject!
                # Chỉ reject khi size thay đổi RẤT LỚN (>150%) = chắc chắn là depth movement
                # BYPASS: vertical > 400px = té từ rất cao, KHÔNG thể là depth movement
                is_depth_movement = size_change_ratio > 1.50 and vertical_movement > 150 and vertical_movement < 400
                
                if is_depth_movement:
                    log.info(f"🚶 Rejected DEPTH MOVEMENT: bbox_size change={size_change_ratio:.2%}, vertical={vertical_movement:.1f}px")
                    log.info(f"   Bbox1: {w1:.0f}x{h1:.0f} ({bbox_size1:.0f}px²) → Bbox2: {w2:.0f}x{h2:.0f} ({bbox_size2:.0f}px²)")
                    log.info(f"   Person moving toward/away from camera, not falling")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'depth-movement',
                        'method': 'depth_movement_filtered'
                    }
                
                # 🕐 GET CURRENT TIME for cooldown check
                current_time = time.time()
                
                # 🚨 CHECK FOR VERY LARGE VERTICAL MOVEMENT (>250px)
                # Vertical > 250px = người từ ĐỨng CAO té xuống, không thể là nằm sẵn!
                has_very_large_vertical = vertical_movement > 250 and center2_y > center1_y
                
                # 🚫 POSTURE FILTER: Reject if person is ALREADY LYING DOWN (not falling)
                # Initial aspect > 1.5 = person lying horizontally (was 1.3, quá thấp reject sai người vừa té)
                # Only detect fall from STANDING/SITTING → LYING, not LYING → LYING movement
                is_initially_lying = aspect_ratio1 > 1.5
                
                # 🚨 BYPASS: Nếu có vertical movement cực lớn (>250px) → KHÔNG reject
                # Vertical=656px nghĩa là người ĐỨng CAO rồi TÉ XUỐNG, không phải nằm sẵn
                if is_initially_lying and not has_very_large_vertical:
                    log.info(f"⚠️ Rejected ALREADY LYING: initial_aspect={aspect_ratio1:.2f} > 1.5 (person already on ground, not falling)")
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
                # Frame height = dynamic → NGỒI thường ở 40-70% chiều cao
                # NGỒI XỔM gần sàn → 70-95% chiều cao (chưa nằm hoàn toàn)
                # TÉ NGÃ thật → người nằm sàn → y >= 90% + aspect >= 1.4 (nằm ngang)
                final_position_ratio = center2_y / frame_height
                
                # 🔥 ENHANCED: Check aspect ratio for squatting detection
                # Squatting: aspect < 1.2 (still vertical, height > width)
                # Falling: aspect >= 1.4 (horizontal, width >> height)
                final_aspect_ratio = aspect_ratio2
                
                # 🔧 FIX v4: ĐẢO NGƯỢC LOGIC - Chỉ CHO PHÉP fall khi CHẮC CHẮN là té
                # TÉ THẬT: position >= 90% VÀ aspect >= 1.4 (nằm sát sàn + nằm ngang hoàn toàn)
                # CÒN LẠI: có thể là NGỒI/XỔM → REJECT
                # 
                # Logic cũ (SAI): position < 85% AND aspect < 1.3 = sitting → Bypass khi 1 điều kiện đúng
                # Logic mới (ĐÚNG): Chỉ fall khi position >= 90% VÀ aspect >= 1.4 → Chặt hơn
                is_definitely_falling = (final_position_ratio >= 0.90) and (final_aspect_ratio >= 1.4)
                
                if not is_definitely_falling:  # Không chắc chắn là té → REJECT
                    # 🔄 CHECK REPEATED SITTING PATTERN (ngồi-đứng-ngồi-đứng)
                    # Nếu phát hiện ngồi xuống nhiều lần trong 10s = đang tập squat
                    current_time_check = time.time()
                    
                    # Thêm event ngồi mới
                    self.sitting_events.append((current_time_check, center2_y))
                    
                    # Xóa events cũ ngoài window 10s
                    self.sitting_events = [(t, y) for t, y in self.sitting_events 
                                          if current_time_check - t <= self.sitting_pattern_window]
                    
                    # Nếu có ≥3 lần ngồi trong 10s = REPEATED PATTERN
                    if len(self.sitting_events) >= self.sitting_pattern_threshold:
                        log.info(f"🏋️ Rejected REPEATED SITTING: {len(self.sitting_events)} times in {self.sitting_pattern_window}s (likely squat exercise)")
                        return {
                            'fall_detected': False,
                            'confidence': 0.0,
                            'angle': 0.0,
                            'category': 'exercise-squat',
                            'method': 'repeated_sitting_filtered'
                        }
                    
                    log.info(f"🪑 Rejected SITTING (NOT DEFINITELY FALL): position={final_position_ratio:.1%} (need >=90%), aspect={final_aspect_ratio:.2f} (need >=1.4)")
                    log.info(f"   Person is likely sitting/squatting, not falling to ground")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'sitting-down',
                        'method': 'sitting_filtered'
                    }
                
                # 🧎 BENDING DETECTION: CÚI NGƯỜI (bending down) → KHÔNG PHẢI FALL!
                # 🔧 FIX v2: Chỉ reject khi aspect < 0.6 (cúi rất sâu, chắc chắn là cúi)
                # aspect 0.6-1.0 có thể là TÉ VỀ PHÍA TRƯỚC → cho phép detect
                # aspect < 0.6: chắc chắn là cúi người bình thường → reject
                if aspect_ratio2 < 0.6:  # 🔧 FIX: 1.0→0.6 - chỉ reject cúi rất sâu
                    log.info(f"🧎 Rejected DEEP BENDING: final_aspect={aspect_ratio2:.2f} < 0.6 (person deeply bent, NOT falling)")
                    log.info(f"   Deep bending is NORMAL activity, not a fall")
                    
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'bending-normal',
                        'method': 'bending_filtered'
                    }
                
                # 🔥 FILTER BBOX JITTER: DISABLED - motion_level không đáng tin cậy
                # 🔧 FIX v2: Bỏ hẳn filter này vì gây quá nhiều false negatives
                # motion_level tính trên toàn frame, không phản ánh chuyển động của người
                motion_str = f"{motion_level:.3f}" if motion_level is not None else "None"
                # DISABLED: filter này reject quá nhiều fall thật
                # if motion_level is not None and motion_level < 0.001:
                #     log.debug(f"⚠️ Rejected bbox jitter: vertical_movement={vertical_movement:.1f}px but motion_level={motion_str} too low")
                if False:  # 🔧 FIX: DISABLED - gây false negatives
                    log.debug(f"⚠️ [DISABLED] bbox jitter filter")
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
                    # 🔥 FIX: Return early on first frame - need at least 2 frames to calculate velocity
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'tracking-started',
                        'method': 'fall_tracking_init'
                    }
                
                # Tính fall duration và fall velocity
                fall_duration = current_time - self.fall_start_time
                total_fall_distance = center2_y - self.fall_start_position
                
                # 🔥 FIX: Cần ít nhất 0.1s để tính velocity hợp lệ
                if fall_duration < 0.1:
                    log.debug(f"⏱️ Fall tracking: waiting for duration >= 0.1s (current: {fall_duration:.2f}s)")
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'tracking-in-progress',
                        'method': 'fall_tracking_wait'
                    }
                
                # Tính vận tốc rơi (pixels/second)
                fall_velocity = total_fall_distance / fall_duration
                
                # 🔥 NEW: CONTROLLED DESCENT DETECTION (Nằm xuống chủ động)
                # 🔧 FIX v2: GIẢM 300→150px/s vì log cho thấy té thật có velocity=163px/s bị reject!
                # - Té ngã thật: velocity > 150 px/s
                # - Nằm xuống chủ động: velocity < 150 px/s (rất chậm, có kiểm soát)
                MIN_FALL_VELOCITY = 150  # 🔧 FIX: 300→150 px/s - tối thiểu để coi là té ngã
                
                if fall_velocity < MIN_FALL_VELOCITY and fall_duration > 0.5:  # 🔧 FIX: duration 0.3→0.5s
                    log.info(f"🧘 Rejected CONTROLLED DESCENT: velocity={fall_velocity:.1f}px/s < {MIN_FALL_VELOCITY}px/s, duration={fall_duration:.2f}s")
                    log.info(f"   Person is LYING DOWN INTENTIONALLY, not falling")
                    # Reset tracking
                    self.fall_start_time = None
                    self.fall_start_position = None
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'controlled-descent',
                        'method': 'controlled_descent_filtered'
                    }
                
                # Phân loại loại té dựa trên velocity và duration
                fall_type = "unknown"
                severity_multiplier = 1.0
                
                # 🔧 FIX v3: GIẢM velocity thresholds vì log cho thấy té thật có velocity=128-197px/s bị reject!
                # Velocity >= 150px/s với duration hợp lý = ngã thật
                if fall_velocity > 400:  # 🔧 FIX: 500→400px/s - Rất nhanh
                    fall_type = "fast_fall"  # TÉ NHANH - Fall bình thường
                    severity_multiplier = 1.0
                    log.info(f"⚡ FAST FALL DETECTED: duration={fall_duration:.2f}s, velocity={fall_velocity:.1f}px/s")
                elif fall_velocity >= 150:  # 🔧 FIX: 300→150px/s - Velocity trung bình cũng là ngã
                    fall_type = "moderate_fall"
                    severity_multiplier = 1.1
                    log.info(f"⚠️ MODERATE FALL: duration={fall_duration:.2f}s, velocity={fall_velocity:.1f}px/s")
                elif fall_duration >= 1.5:
                    fall_type = "slow_collapse"  # TÉ CHẬM - Đột quỵ/yếu sức
                    severity_multiplier = 1.3  # Tăng severity vì nguy hiểm hơn!
                    log.warning(f"🏥 SLOW COLLAPSE (Possible Stroke): duration={fall_duration:.2f}s, velocity={fall_velocity:.1f}px/s")
                else:
                    # Velocity quá thấp (< 150px/s) + duration ngắn = có thể là nằm xuống chủ động
                    log.info(f"🧘 Rejected AMBIGUOUS DESCENT: velocity={fall_velocity:.1f}px/s < 150, duration={fall_duration:.2f}s - too slow for fall")
                    self.fall_start_time = None
                    self.fall_start_position = None
                    return {
                        'fall_detected': False,
                        'confidence': 0.0,
                        'angle': 0.0,
                        'category': 'ambiguous-descent',
                        'method': 'ambiguous_descent_filtered'
                    }
                
                downward_confidence = min(0.9, 0.50 + (vertical_movement / 180))  # GIẢM base 0.55→0.50
                downward_confidence *= severity_multiplier  # Điều chỉnh theo loại té
                downward_confidence = min(0.95, downward_confidence)  # Cap ở 0.95
                
                if downward_confidence >= 0.50:  # GIẢM threshold 0.60→0.50 để dễ detect
                    # 🪑 FINAL CHECK: Chỉ cho phép RAPID FALL khi CHẮC CHẮN là té ngã
                    # TÉ THẬT: position >= 90% VÀ aspect >= 1.4 (nằm sát sàn + nằm ngang)
                    # HOẶC velocity >= 600px/s (quá nhanh để là ngồi)
                    final_position_ratio_check = center2_y / frame_height
                    final_aspect_check = aspect_ratio2
                    
                    # 🔧 FIX v4: Logic nhất quán với STRATEGY 0
                    # Chỉ CHO PHÉP fall khi: velocity >= 600 HOẶC (position >= 90% VÀ aspect >= 1.4)
                    is_definitely_rapid_fall = (
                        fall_velocity >= 600 or  # Velocity quá cao = chắc chắn té
                        (final_position_ratio_check >= 0.90 and final_aspect_check >= 1.4)  # Nằm sát sàn + nằm ngang
                    )
                    
                    if not is_definitely_rapid_fall:
                        log.info(f"🪑 Rejected RAPID SITTING: velocity={fall_velocity:.1f}px/s, position={final_position_ratio_check:.1%} (need >=90%), aspect={final_aspect_check:.2f} (need >=1.4)")
                        log.info(f"   Not fast enough OR not on ground - likely sitting/squatting")
                        # Reset tracking
                        self.fall_start_time = None
                        self.fall_start_position = None
                        return {
                            'fall_detected': False,
                            'confidence': 0.0,
                            'angle': 0.0,
                            'category': 'rapid-sitting',
                            'method': 'rapid_sitting_filtered'
                        }
                    
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
            
            # NOTE: SIDEWAYS FALL detection đã được xử lý ở PRIORITY CHECK 2 (trước POSTURE ADJUSTMENT)
            # Code duplicate đã được xóa để tránh confusion
            
            # STRATEGY 0.5: MODERATE FALL - Cân bằng giữa nhạy và chính xác
            # 🎯 BALANCED: Detect fall thật nhưng TRÁNH NGỐI XUỐNG
            # Kiểm tra: aspect tăng nhiều (>1.25) + vertical đủ lớn (>65px)
            if (vertical_movement > 65 and  # TĂNG 50→65px: tránh ngồi xuống
                aspect_change > 1.25 and  # TĂNG 1.1→1.25: người PHẢI nằm ngang thật sự
                center2_y > center1_y and  # Moving downward
                horizontal_movement < vertical_movement * 1.2 and  # Vertical phải lớn hơn horizontal
                aspect_ratio1 < 1.3):  # 🚫 REJECT REVERSE: initial pose must be UPRIGHT (not lying)
                
                # 🪑 SITTING FILTER CHO STRATEGY 0.5: Check position + aspect
                final_position_for_moderate = center2_y / frame_height
                final_aspect_for_moderate = aspect_ratio2
                
                # 🚫 FILTER NGỐI XUỐNG: 
                # 1. Aspect change không đủ cao (< 1.35)
                # 2. HOẶC position còn cao (< 85%) VÀ aspect cuối nhỏ (< 1.3) = chưa nằm xuống sàn
                is_moderate_sitting = (
                    aspect_change < 1.35 or  # Aspect thay đổi ít = ngồi
                    (final_position_for_moderate < 0.85 and final_aspect_for_moderate < 1.3)  # Position cao + aspect nhỏ = ngồi
                )
                
                if is_moderate_sitting:
                    log.info(f"⚠️ Rejected SITTING (STRATEGY 0.5): vertical={vertical_movement:.1f}px, aspect={aspect_change:.2f}x, position={final_position_for_moderate:.1%}, final_aspect={final_aspect_for_moderate:.2f}")
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
                horizontal_movement < vertical_movement * 1.5 and  # Vertical phải dominant
                aspect_ratio1 < 1.3):  # 🚫 REJECT REVERSE: person must start UPRIGHT, not lying
                
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
        
        # 🎯 UPDATE STATE MACHINE: Track person posture based on aspect ratio
        try:
            current_time = time.time()
            aspect_ratio = aspect_ratio2
            
            # Determine state based on aspect ratio
            new_state = self.person_state
            if aspect_ratio > 1.3:
                new_state = "LYING"
            elif aspect_ratio > 1.0:
                new_state = "SITTING"
            else:
                new_state = "STANDING"
            
            # Update state and track lying time
            if new_state != self.person_state:
                self.person_state = new_state
                self.state_change_time = current_time
                
                if new_state == "LYING":
                    self.lying_start_time = current_time
                    log.debug(f"🛏️ State changed to LYING (aspect={aspect_ratio:.2f})")
                else:
                    self.lying_start_time = None
                    log.debug(f"🚶 State changed to {new_state} (aspect={aspect_ratio:.2f})")
        except:
            pass
            
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
    
    def get_person_state(self):
        """Get current person state and lying duration."""
        lying_duration = 0
        if self.person_state == "LYING" and self.lying_start_time:
            lying_duration = time.time() - self.lying_start_time
        
        return {
            'state': self.person_state,
            'lying_duration': lying_duration,
            'state_change_time': self.state_change_time
        }
