import cv2
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path

# Import Supabase integration
from service.emergency_notification_dispatcher import HealthcareEventPublisher

# Import snapshot service for image storage
from infrastructure.services.snapshot_service import get_snapshot_service

class AdvancedHealthcarePipeline:
    def __init__(self, camera, video_processor, fall_detector, seizure_detector, seizure_predictor, alerts_folder, camera_id=None, user_id=None):
        self.camera = camera
        self.video_processor = video_processor
        self.fall_detector = fall_detector
        self.seizure_detector = seizure_detector
        self.seizure_predictor = seizure_predictor
        self.alert_save_path = alerts_folder
        self.camera_id = camera_id
        self.user_id = user_id
        
        # Initialize Supabase event publisher with real camera_id
        self.event_publisher = HealthcareEventPublisher(
            default_user_id=user_id,
            default_camera_id=camera_id
        )
        
        # Initialize snapshot service for image storage
        try:
            self.snapshot_service = get_snapshot_service()
            print(f"📸 Snapshot service initialized - MinIO storage ready")
        except Exception as e:
            print(f"⚠️ Snapshot service failed to initialize: {e}")
            print(f"📸 Will use local file storage fallback")
            self.snapshot_service = None
        
        # Create alert save directory
        Path(self.alert_save_path).mkdir(parents=True, exist_ok=True)
        
        # Simplified statistics - only essential metrics
        self.stats = {
            'start_time': time.time(),
            'total_frames': 0,
            'frames_processed': 0,
            'keyframes_detected': 0,
            'persons_detected': 0,
            'fps': 0.0,
            'frame_times': [],
            # Fall detection metrics
            'fall_detections': 0,
            'last_fall_time': None,
            # Seizure detection metrics
            'seizure_detections': 0,
            'last_seizure_time': None,
            'seizure_warnings': 0,
            'pose_extraction_failures': 0,
            # Alert metrics
            'critical_alerts': 0,
            'total_alerts': 0,
            'last_alert_time': None,
            'alert_type': 'normal'
        }
        
        # Enhanced detection history
        self.detection_history = {
            'fall_confidences': [],
            'seizure_confidences': [],
            'motion_levels': [],
            'max_history': 10,
            'fall_confirmation_frames': 0,
            'seizure_confirmation_frames': 0,
            'last_significant_motion': time.time()
        }
        
        # Performance tracking - merged with stats
        self.performance = {
            'fall_detection_time': 0.0,
            'seizure_detection_time': 0.0,
            'total_detection_time': 0.0
        }
        
        # FIX: Initialize frame tracking for motion calculation
        self._prev_frame = None
        self._current_frame = None
        
        # 🎬 FRAME BUFFER: Lưu 5 frames gần nhất để chụp lại hành động trước khi sự kiện xảy ra
        # Khi phát hiện fall/seizure, sẽ chụp: [frame-2, frame-1, current_frame]
        from collections import deque
        self._frame_buffer = deque(maxlen=5)  # Lưu tối đa 5 frames gần nhất
        self._frame_timestamps = deque(maxlen=5)  # Timestamp tương ứng

    def process_frame(self, frame, other_cameras=None):
        """
        Process frame với skip frame logic và keyframe detection như file mẫu
        
        Args:
            frame: Current frame to process
            other_cameras: List of other camera dicts in same room (for multi-angle capture)
        """
        # Cập nhật total frames
        self.stats['total_frames'] += 1
        
        # FIX: Lưu frame trước và hiện tại ĐÚNG để tính motion
        # _current_frame là frame hiện tại
        self._current_frame = frame.copy()
        # Không set _prev_frame ở đây! Nó được set ở cuối hàm

        # SKIP FRAME LOGIC - chỉ xử lý keyframe quan trọng
        processing_result = self.video_processor.process_frame(frame)
        
        # Nếu không phải keyframe, trả về kết quả đơn giản
        if not processing_result['processed']:
            normal_window = self.create_normal_camera_window(frame, [])
            ai_window = frame.copy()
            
            # FIX: Vẫn cần update _prev_frame ngay cả khi skip
            self._prev_frame = self._current_frame
            
            return {
                "normal_window": normal_window, 
                "ai_window": ai_window,
                "detection_result": {
                    'fall_detected': False, 'fall_confidence': 0.0,
                    'seizure_detected': False, 'seizure_confidence': 0.0,
                    'seizure_ready': False, 'keypoints': None,
                    'alert_level': 'normal', 'emergency_type': None
                },
                "person_detections": []
            }

        # KEYFRAME DETECTED - xử lý AI detection
        self.stats['keyframes_detected'] += 1
        persons = processing_result.get('person_detections', processing_result.get('detections', []))
        
        # Process dual detection như file mẫu
        detection_result = self.process_dual_detection(frame, persons)
        
        # Update statistics
        self.update_statistics(detection_result, len(persons))

        # Vẽ overlay
        normal_window = self.create_normal_camera_window(frame, persons)
        ai_window = frame.copy()

        # FIX: Update _prev_frame SAU KHI xử lý xong frame hiện tại
        self._prev_frame = self._current_frame
        
        # 🎬 LƯU FRAME VÀO BUFFER để có thể lấy lại frames trước đó
        # Buffer sẽ tự động loại bỏ frame cũ nhất khi đầy (maxlen=5)
        if self._current_frame is not None:
            self._frame_buffer.append(self._current_frame.copy())
            self._frame_timestamps.append(time.time())

        return {
            "normal_window": normal_window,
            "ai_window": ai_window,
            "detection_result": detection_result,
            "person_detections": persons
        }
        
    def process_dual_detection(self, frame, person_detections):
        """Process dual detection như file mẫu với enhanced accuracy"""
        start_time = time.time()
        
        result = {
            'fall_detected': False, 'fall_confidence': 0.0,
            'seizure_detected': False, 'seizure_confidence': 0.0,
            'seizure_ready': False, 'alert_level': 'normal',
            'keypoints': None, 'emergency_type': None
        }
        
        # DEBUG: Log person detection count
        if self.stats['total_frames'] % 60 == 0:  # Every 2 seconds
            print(f"👥 Person Detections: {len(person_detections)} detected")
        
        if not person_detections:
            # Reset confirmation frames when no person detected
            self.detection_history['fall_confirmation_frames'] = 0
            self.detection_history['seizure_confirmation_frames'] = 0
            if self.stats['total_frames'] % 60 == 0:
                print(f"⚠️ No person detected - skipping fall/seizure detection")
            return result
            
        # Calculate motion level for enhanced detection
        motion_level = self.calculate_motion_level_person(person_detections)
        self.detection_history['motion_levels'].append(motion_level)
        if len(self.detection_history['motion_levels']) > self.detection_history['max_history']:
            self.detection_history['motion_levels'].pop(0)
            
        # Update significant motion tracker
        if motion_level > 0.3:
            self.detection_history['last_significant_motion'] = time.time()
            
        # Get primary person (largest detection)
        primary_person = max(person_detections, key=lambda x: x.get('bbox', [0,0,0,0])[2] * x.get('bbox', [0,0,0,0])[3])
        person_bbox = [
            int(primary_person['bbox'][0]), int(primary_person['bbox'][1]),
            int(primary_person['bbox'][0] + primary_person['bbox'][2]),
            int(primary_person['bbox'][1] + primary_person['bbox'][3])
        ]
        
        # Fall detection with improvements và COOLDOWN LOGIC
        fall_start = time.time()
        
        # COOLDOWN: Prevent fall detection spam - 10 GIÂY
        current_time = time.time()
        FALL_COOLDOWN = 10.0  # 10 giây giữa các fall detection - CHỈ 1 EVENT tại 1 thời điểm!
        if (self.stats['last_fall_time'] and 
            current_time - self.stats['last_fall_time'] < FALL_COOLDOWN):
            if self.stats['total_frames'] % 30 == 0:
                time_left = FALL_COOLDOWN - (current_time - self.stats['last_fall_time'])
                print(f"⏱️ Fall detection in COOLDOWN: {time_left:.1f}s remaining")
            result['fall_confidence'] = 0.0  # Force reset để tránh spam
            return result  # 🔥 RETURN IMMEDIATELY - skip all fall detection logic during cooldown!
        
        # ❌ REMOVED: Don't update last_fall_time here - only update when ACTUALLY detecting a fall!
        # The old code was causing every frame to trigger cooldown even without fall detection
        
        # Now proceed with fall detection
        # DEBUG: Check if fall detector exists (disabled)
        # if self.stats['total_frames'] % 60 == 0:
        #     print(f"🔍 Fall Detector Status: {'Available' if self.fall_detector else 'NULL'}")
        
        try:
            # FIXED: Pass correct parameters - frame, timestamp, and person_bbox
            fall_result = self.fall_detector.detect_fall(
                frame=frame,
                timestamp=current_time,
                person_bbox=person_bbox
            )
            base_fall_confidence = fall_result['confidence']
            fall_method = fall_result.get('method', 'unknown')
            
            # 🔥 NEW: Extract fall velocity information
            fall_type = fall_result.get('fall_type', 'unknown')  # fast_fall, slow_collapse, moderate_fall
            fall_duration = fall_result.get('fall_duration', 0.0)  # Thời gian té (seconds)
            fall_velocity = fall_result.get('fall_velocity', 0.0)  # Vận tốc té (px/s)
            
            # Debug: Log fall detection attempt - ENABLED để debug
            if self.stats['total_frames'] % 30 == 0:  # Every 1 second
                print(f"🔍 Fall Detection Debug: Confidence={base_fall_confidence:.3f}, Motion={motion_level:.3f}, Threshold=0.28, Method={fall_method}")
            
            # FALL DETECTION THRESHOLD: TĂNG NHẠY để detect fall tốt hơn
            # Giảm từ 0.35 → 0.28 (28%) để tăng sensitivity, vẫn có motion check
            # Yêu cầu thêm: motion_level > 0.015 để đảm bảo có chuyển động thực sự
            has_real_motion = motion_level > 0.015  # Motion thật sự, giảm từ 0.02→0.015 để nhạy hơn
            if base_fall_confidence >= 0.28 and has_real_motion:  # NHẠY HƠN: 28% + motion check nhẹ hơn
                result['fall_detected'] = True
                result['fall_confidence'] = base_fall_confidence
                self.stats['fall_detections'] += 1
                # 🔥 UPDATE last_fall_time NOW when fall is actually detected!
                self.stats['last_fall_time'] = current_time
                
                # 🔥 Log fall type information
                if fall_type == "slow_collapse":
                    print(f"🚨🏥 SLOW COLLAPSE DETECTED (POSSIBLE STROKE)! 🚨🏥")
                    print(f"⏱️ Duration: {fall_duration:.2f}s | Velocity: {fall_velocity:.1f}px/s")
                    print(f"🩺 HIGH PRIORITY - Medical attention needed!")
                elif fall_type == "fast_fall":
                    print(f"🚨⚡ FAST FALL DETECTED! 🚨⚡")
                    print(f"⏱️ Duration: {fall_duration:.2f}s | Velocity: {fall_velocity:.1f}px/s")
                else:
                    print(f"🚨🚨🚨 TÉ NGÃ PHÁT HIỆN! 🚨🚨🚨")
                    print(f"⏱️ Duration: {fall_duration:.2f}s | Velocity: {fall_velocity:.1f}px/s")
                
                print(f"📊 Confidence: {base_fall_confidence:.2f} | Motion: {motion_level:.2f} | Type: {fall_type}")
                print(f"📊 Method: {fall_method} | Alert Level: CRITICAL")
                print(f"🏥 Emergency Type: FALL (TÉ NGÃ)")
                
                # 🔥 CREATE EVENT FIRST (to get event_id for snapshots)
                import uuid
                
                event_id = None
                snapshot_id = None
                try:
                        # 🔥 FIX: Save alert image IMMEDIATELY to get correct image_path
                        # This prevents using OLD images from alerts folder for caption generation
                        alert_image_path = None
                        if frame is not None:
                            try:
                                import os
                                
                                # Create alerts directory if needed
                                alerts_dir = "examples/data/saved_frames/alerts"
                                
                                # Save current frame as alert image with timestamp
                                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                alert_image_path = os.path.join(
                                    alerts_dir,
                                    f"fall_detected_{timestamp_str}_conf_{base_fall_confidence:.2f}.jpg"
                                )
                                
                                import cv2
                                cv2.imwrite(alert_image_path, frame)
                                print(f"💾 Saved alert image BEFORE caption: {alert_image_path}")
                            except Exception as save_err:
                                print(f"⚠️ Failed to save alert image: {save_err}")
                                alert_image_path = None
                        
                        bounding_boxes = [{"bbox": person_bbox, "confidence": 1.0}] if person_bbox else []
                        
                        # 🔥 Determine severity based on fall type
                        severity = 'critical' if fall_type == 'slow_collapse' else 'high'
                        description = f'{fall_type.replace("_", " ").title()} detected with {base_fall_confidence:.1%} confidence'
                        if fall_type == 'slow_collapse':
                            description += ' - POSSIBLE STROKE OR WEAKNESS'
                        
                        event_data = {
                            'event_type': 'fall',
                            'description': description,
                            'detection_data': {
                                'algorithm': 'yolo_fall_detection',
                                'model_version': 'v1.0',
                                'detection_timestamp': datetime.now().isoformat(),
                                'severity': severity,
                                'method': fall_method,
                                'motion_level': motion_level,
                                'fall_type': fall_type,  # 🔥 NEW
                                'fall_duration': fall_duration,  # 🔥 NEW
                                'fall_velocity': fall_velocity  # 🔥 NEW
                            },
                            'confidence': base_fall_confidence,
                            'bounding_boxes': bounding_boxes,
                            'context': {
                                'motion_level': motion_level,
                                'detection_type': 'direct',
                                'person_bbox': person_bbox,
                                'method': fall_method,
                                'fall_type': fall_type,  # 🔥 NEW
                                'fall_duration': fall_duration,  # 🔥 NEW
                                'fall_velocity': fall_velocity  # 🔥 NEW
                            },
                            'camera_id': self.camera_id,
                            'user_id': self.user_id,
                            'image_path': alert_image_path,  # 🔥 FIX: Pass current image path
                            # NO frame - snapshot will be created separately
                        }
                        
                        # Publish event to get event_id (without snapshot yet)
                        event_result = self.event_publisher.postgresql_service.publish_event_detection(event_data)
                        event_id = event_result.get('event_id') if isinstance(event_result, dict) else str(event_result)
                        print(f"✅ Fall event created: {event_id}")
                        
                        # 🔥 CAPTURE NGAY TẠI THỜI ĐIỂM PHÁT HIỆN!
                        # Image 1: Frame hiện tại (đang phát hiện event)
                        # Image 2-5: Multi-angle (if same room) hoặc Temporal (if different rooms)
                        snapshot_ids = self.capture_5_snapshots_immediate(
                            current_frame=frame,  # Frame ĐANG phát hiện event
                            event_type='fall',
                            confidence=base_fall_confidence,
                            event_id=event_id,
                            metadata={
                                'motion_level': motion_level,
                                'detection_type': 'direct',
                                'person_bbox': person_bbox,
                                'method': fall_method,
                                'fall_type': fall_type,  # 🔥 NEW
                                'fall_duration': fall_duration,  # 🔥 NEW
                                'fall_velocity': fall_velocity  # 🔥 NEW
                            }
                        )
                        
                        # Update event with snapshot_id
                        if snapshot_ids and len(snapshot_ids) > 0:
                            snapshot_id = snapshot_ids[0]
                            self.event_publisher.postgresql_service.update_event_snapshot(event_id, snapshot_id)
                            print(f"📸 {len(snapshot_ids)} Fall images saved and linked to event!")
                        
                except Exception as e:
                    print(f"❌ Failed to create fall event: {e}")
                    event_id = str(uuid.uuid4())
                
                # ❌ REMOVED: Duplicate event creation via dispatcher
                # Event already created above (lines 240-295) with:
                # - Cooldown check (45s)
                # - 5 snapshots (temporal mode)
                # - Vietnamese caption
                # - Database event_detections record
                # Dispatcher (publish_fall_detection) is disabled and no longer creates events!
                
                print(f"✅ Fall event created: {event_id[:8]}... (already saved to DB with cooldown + 5 snapshots)")
                
                # try:
                #     context_data = {
                #         'motion_level': motion_level,
                #         'detection_type': 'direct',
                #         'processing_time': time.time() - fall_start,
                #         'frame_number': self.stats['total_frames'],
                #         'snapshot_id': snapshot_id,
                #         'event_id': event_id,
                #         'image_path': alert_image_path,
                #         'description': f'Fall activity detected with {base_fall_confidence:.1%} confidence'
                #     }
                #     
                #     response = self.event_publisher.publish_fall_detection(
                #         confidence=base_fall_confidence,
                #         bounding_boxes=bounding_boxes,
                #         context=context_data
                #     )
                #     
                #     if response.get('alert_created'):
                #         print(f"📡 Fall alert created: Priority {response.get('priority_level')}")
                #     else:
                #         print(f"📵 Fall alert skipped: Lower priority than existing alerts")
                #         
                # except Exception as e:
                #     print(f"Error publishing fall detection: {e}")
                    
            else:
                # Apply motion enhancement and smoothing for lower confidence cases
                enhanced_fall_confidence = self.enhance_detection_with_motion(base_fall_confidence, motion_level, 'fall')
                smoothed_fall_confidence = self.smooth_detection_confidence(enhanced_fall_confidence, 'fall')
                
                # BALANCED THRESHOLD: Reduce false positives while keeping sensitivity
                fall_threshold = 0.4  # Tăng từ 0.1 lên 0.4 để giảm false positive
                if smoothed_fall_confidence > fall_threshold:
                    self.detection_history['fall_confirmation_frames'] += 1
                else:
                    self.detection_history['fall_confirmation_frames'] = max(0, self.detection_history['fall_confirmation_frames'] - 1)
                
                # REQUIRE MORE FRAMES: More confirmation frames to reduce spam
                min_confirmation_frames = 3  # Giảm xuống 3 để nhanh hơn
                if self.detection_history['fall_confirmation_frames'] >= min_confirmation_frames:
                    result['fall_detected'] = True
                    result['fall_confidence'] = smoothed_fall_confidence
                    self.stats['fall_detections'] += 1
                    # last_fall_time already updated above at cooldown check!
                    print(f"🚨 FALL DETECTED! Confidence: {smoothed_fall_confidence:.2f} | Motion: {motion_level:.2f} | Frames: {self.detection_history['fall_confirmation_frames']}")
                    print(f"📊 Alert Level: HIGH | Emergency Type: Fall | Enhanced Detection")
                    
                    # Save detection snapshot to MinIO
                    snapshot_id = self.save_detection_snapshot(
                        frame=frame,
                        event_type='fall',
                        confidence=smoothed_fall_confidence,
                        metadata={
                            'motion_level': motion_level,
                            'detection_type': 'confirmation',
                            'confirmation_frames': self.detection_history['fall_confirmation_frames'],
                            'person_bbox': person_bbox
                        }
                    )
                    if snapshot_id:
                        print(f"📸 Fall confirmation image saved: {snapshot_id[:8]}...")
                    
                    # ❌ DISABLED: Confirmation notification removed to prevent duplicate events
                    # Event was already created with cooldown + 5 snapshots in direct detection path (lines 240-295)
                    # Supabase Realtime will handle notifications automatically when event is inserted
                    print(f"✅ Fall confirmation complete - event already published with cooldown")
                        
                else:
                    result['fall_confidence'] = smoothed_fall_confidence
                    
        except Exception as e:
            print(f"Fall detection error: {str(e)}")
            
        self.performance['fall_detection_time'] = time.time() - fall_start
        
        # Seizure detection with improvements và DEBUG LOGGING
        seizure_time_start = time.time()
        if self.stats['total_frames'] % 60 == 0:  # Every 2 seconds
            print(f"🧠 Seizure Detector Check: detector={'Available' if self.seizure_detector else 'NULL'}")
        
        if self.seizure_detector is not None:
            try:
                seizure_result = self.seizure_detector.detect_seizure(frame, person_bbox)
                result['seizure_ready'] = seizure_result.get('temporal_ready', False)
                result['keypoints'] = seizure_result.get('keypoints')
                
                # Debug: Always show seizure detector status with instructions
                if self.stats['total_frames'] % 60 == 0:  # Every 2 seconds
                    temporal_status = "READY" if seizure_result.get('temporal_ready') else "NOT_READY"
                    print(f"🧠 Seizure Detector Status: {temporal_status} | Raw Confidence: {seizure_result.get('confidence', 0):.3f}")
                    if not seizure_result.get('temporal_ready'):
                        print("💡 Hướng dẫn test co giật (threshold=0.35 - SIÊU NHẠY):")
                        print("   ⚠️ Phải NGỒI hoặc NẰM (không đứng - bị chặn bởi bộ lọc)")
                        print("   ✅ Vẫy tay NHANH qua lại liên tục (5-7 lần)")
                        print("   ✅ Lắc đầu + vai đột ngột và mạnh")
                        print("   ✅ Động tác GIẬT GIẬT - không mượt, có dừng đột ngột")
                
                if seizure_result.get('temporal_ready'):
                    # Update seizure predictor nếu có
                    if self.seizure_predictor:
                        pred_result = self.seizure_predictor.update_prediction(seizure_result['confidence'])
                        base_seizure_confidence = pred_result['smoothed_confidence']
                    else:
                        base_seizure_confidence = seizure_result['confidence']
                    
                    # Apply motion enhancement and additional smoothing
                    enhanced_seizure_confidence = self.enhance_detection_with_motion(base_seizure_confidence, motion_level, 'seizure')
                    final_seizure_confidence = self.smooth_detection_confidence(enhanced_seizure_confidence, 'seizure')
                    
                    # Debug: Log seizure detection attempt MORE FREQUENTLY
                    if self.stats['total_frames'] % 30 == 0:  # Every 1 second instead of 2 seconds
                        print(f"🔍 Seizure Debug: Base={base_seizure_confidence:.3f}, Final={final_seizure_confidence:.3f}, Motion={motion_level:.3f}, Temporal={seizure_result.get('temporal_ready', False)}")
                    
                    # BALANCED: Thresholds cân bằng để giảm false positive
                    seizure_threshold = 0.60   # CRITICAL: Tăng từ 0.5 lên 0.60 - chắc chắn hơn
                    warning_threshold = 0.40   # WARNING: Giảm từ 0.5 xuống 0.40 - tạo vùng warning
                    
                    if final_seizure_confidence > seizure_threshold:
                        self.detection_history['seizure_confirmation_frames'] += 1
                        if self.stats['total_frames'] % 30 == 0:
                            print(f"🎯 Seizure Above Threshold: {final_seizure_confidence:.3f} > {seizure_threshold}, Frames: {self.detection_history['seizure_confirmation_frames']}")
                    elif final_seizure_confidence > warning_threshold:
                        # Keep frames for warning level
                        if self.stats['total_frames'] % 30 == 0:
                            print(f"⚠️ Seizure Warning Level: {final_seizure_confidence:.3f} > {warning_threshold}, Frames: {self.detection_history['seizure_confirmation_frames']}")
                    else:
                        # Reset to 0 when below warning threshold
                        self.detection_history['seizure_confirmation_frames'] = 0
                    
                    # BALANCED: Cần nhiều frames hơn để confirm
                    min_seizure_confirmation = 3  # Tăng từ 1 lên 3 frames - chắc chắn hơn
                    if self.detection_history['seizure_confirmation_frames'] >= min_seizure_confirmation:
                        # COOLDOWN CHECK: 10 GIÂY giữa các seizure detection
                        current_time = time.time()
                        SEIZURE_COOLDOWN = 30.0  # 30 giây cooldown - TĂNG từ 10s để tránh spam alarm!
                        if (self.stats['last_seizure_time'] is None or 
                            current_time - self.stats['last_seizure_time'] > SEIZURE_COOLDOWN):
                            if self.stats['total_frames'] % 30 == 0 and self.stats['last_seizure_time']:
                                time_left = SEIZURE_COOLDOWN - (current_time - self.stats['last_seizure_time'])
                                if time_left > 0:
                                    print(f"⏱️ Seizure detection in COOLDOWN: {time_left:.1f}s remaining")
                            result['seizure_detected'] = True
                            result['seizure_confidence'] = final_seizure_confidence
                            self.stats['seizure_detections'] += 1
                            self.stats['last_seizure_time'] = time.time()
                            print(f"🚨 SEIZURE DETECTED! Confidence: {final_seizure_confidence:.2f} | Motion: {motion_level:.2f} | Frames: {self.detection_history['seizure_confirmation_frames']}")
                            print(f"📊 Alert Level: CRITICAL | Emergency Type: Seizure")
                            
                            # 🔥 CREATE EVENT FIRST (to get event_id for snapshots)
                            import uuid
                            
                            event_id = None
                            alert_image_path = None
                            try:
                                # 🔥 FIX: Save alert image IMMEDIATELY
                                if frame is not None:
                                    try:
                                        import os
                                        
                                        # Create alerts directory if needed
                                        alerts_dir = "examples/data/saved_frames/alerts"
                                        os.makedirs(alerts_dir, exist_ok=True)
                                        
                                        # Save current frame as alert image
                                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                        alert_image_path = os.path.join(
                                            alerts_dir,
                                            f"seizure_detected_{timestamp_str}_conf_{final_seizure_confidence:.2f}.jpg"
                                        )
                                        
                                        import cv2
                                        cv2.imwrite(alert_image_path, frame)
                                        print(f"💾 Saved seizure alert image: {alert_image_path}")
                                    except Exception as save_err:
                                        print(f"⚠️ Failed to save seizure alert image: {save_err}")
                                
                                bounding_boxes = [{"bbox": person_bbox, "confidence": 1.0}] if person_bbox else []
                                event_data = {
                                    'event_type': 'abnormal_behavior',
                                    'description': f'Seizure activity detected with {final_seizure_confidence:.1%} confidence',
                                    'detection_data': {
                                        'algorithm': 'seizure_detection',
                                        'behavior_type': 'seizure',
                                        'model_version': 'v1.0',
                                        'detection_timestamp': datetime.now().isoformat(),
                                        'severity': 'high',
                                        'motion_level': motion_level,
                                        'confirmation_frames': self.detection_history['seizure_confirmation_frames']
                                    },
                                    'confidence': final_seizure_confidence,
                                    'bounding_boxes': bounding_boxes,
                                    'context': {
                                        'motion_level': motion_level,
                                        'detection_type': 'confirmation',
                                        'confirmation_frames': self.detection_history['seizure_confirmation_frames'],
                                        'temporal_ready': seizure_result.get('temporal_ready', False),
                                        'person_bbox': person_bbox
                                    },
                                    'camera_id': self.camera_id,
                                    'user_id': self.user_id,
                                    'image_path': alert_image_path  # 🔥 FIX: Pass current image path
                                }
                                
                                # Publish event to get event_id
                                event_result = self.event_publisher.postgresql_service.publish_event_detection(event_data)
                                event_id = event_result.get('event_id') if isinstance(event_result, dict) else str(event_result)
                                print(f"✅ Seizure event created: {event_id}")
                            except Exception as e:
                                print(f"❌ Failed to create seizure event: {e}")
                                event_id = str(uuid.uuid4())
                            
                            # 🔥 CAPTURE NGAY TẠI THỜI ĐIỂM PHÁT HIỆN CO GIẬT!
                            snapshot_ids = self.capture_5_snapshots_immediate(
                                current_frame=frame,  # Frame ĐANG phát hiện seizure
                                event_type='seizure',
                                confidence=final_seizure_confidence,
                                event_id=event_id,
                                metadata={
                                    'motion_level': motion_level,
                                    'detection_type': 'confirmation',
                                    'confirmation_frames': self.detection_history['seizure_confirmation_frames'],
                                    'temporal_ready': seizure_result.get('temporal_ready', False),
                                    'person_bbox': person_bbox,
                                    'keypoints': seizure_result.get('keypoints')
                                },
                                other_cameras=other_cameras  # 🎥 Pass multi-camera info
                            )
                            
                            # Update event with snapshot_id
                            snapshot_id = None
                            if snapshot_ids and len(snapshot_ids) > 0:
                                snapshot_id = snapshot_ids[0]
                                self.event_publisher.postgresql_service.update_event_snapshot(event_id, snapshot_id)
                                print(f"📸 {len(snapshot_ids)} Seizure images saved and linked to event!")
                            
                            # Send notification (dispatcher won't create duplicate event now)
                            # ❌ REMOVED: Duplicate event creation via dispatcher
                            # Event already created above with:
                            # - Cooldown check (45s)
                            # - 5 snapshots (temporal mode)
                            # - Vietnamese caption
                            # - Database event_detections record
                            # Dispatcher (publish_seizure_detection) is disabled!
                            
                            print(f"✅ Seizure event created: {event_id[:8] if event_id else 'N/A'}... (already saved to DB with cooldown + 5 snapshots)")
                            
                            # try:
                            #     context_data = {
                            #         'motion_level': motion_level,
                            #         'detection_type': 'confirmation',
                            #         'confirmation_frames': self.detection_history['seizure_confirmation_frames'],
                            #         'temporal_ready': seizure_result.get('temporal_ready', False),
                            #         'processing_time': time.time() - seizure_time_start,
                            #         'frame_number': self.stats['total_frames'],
                            #         'snapshot_id': snapshot_id,
                            #         'event_id': event_id,
                            #         'image_path': alert_image_path,
                            #         'description': f'Seizure activity detected with {final_seizure_confidence:.1%} confidence'
                            #     }
                            #     
                            #     response = self.event_publisher.publish_seizure_detection(
                            #         confidence=final_seizure_confidence,
                            #         bounding_boxes=bounding_boxes,
                            #         context=context_data
                            #     )
                            #     
                            #     if response.get('alert_created'):
                            #         print(f"📡 Seizure alert created: Priority {response.get('priority_level')}")
                            #     else:
                            #         print(f"📵 Seizure alert skipped: Lower priority than existing alerts")
                            #         
                            # except Exception as e:
                            #     print(f"Error publishing seizure detection: {e}")
                            
                            # RESET confirmation frames after detection
                            self.detection_history['seizure_confirmation_frames'] = 0
                        else:
                            # Still in cooldown period - log once per second
                            if self.stats['total_frames'] % 30 == 0:
                                time_left = SEIZURE_COOLDOWN - (current_time - self.stats['last_seizure_time'])
                                print(f"⏱️ Seizure detection in COOLDOWN: {time_left:.1f}s remaining")
                            result['seizure_confidence'] = final_seizure_confidence
                            # Reset confirmation frames khi trong cooldown
                            self.detection_history['seizure_confirmation_frames'] = 0
                    elif final_seizure_confidence > warning_threshold and motion_level > 0.2:  # Giảm motion threshold
                        result['seizure_confidence'] = final_seizure_confidence
                        self.stats['seizure_warnings'] += 1
                        print(f"⚠️ SEIZURE WARNING! Confidence: {final_seizure_confidence:.2f} | Motion: {motion_level:.2f}")
                        print(f"📊 Alert Level: Warning | Emergency Type: Seizure Warning")
                    else:
                        result['seizure_confidence'] = final_seizure_confidence
                        
            except Exception as e:
                print(f"Seizure detection error: {str(e)}")
                self.stats['pose_extraction_failures'] += 1
        else:
            pass  # Seizure detector is None - no need to log
                
        self.performance['seizure_detection_time'] = time.time() - seizure_time_start
        
        # 🔥 NEW: 5-LEVEL STATUS DETERMINATION
        # danger, warning, suspect, normal, unknown
        
        # Get fall_type for danger classification
        fall_type = result.get('fall_type', 'unknown')
        
        if result['seizure_detected'] and result['fall_detected']:
            # Cả 2 đều detected - so sánh confidence
            if result['seizure_confidence'] > result['fall_confidence']:
                result['alert_level'] = 'danger'  # danger: Co giật xác nhận
                result['emergency_type'] = 'seizure'
                self.stats['critical_alerts'] += 1
                print(f"⚖️ Both detected, seizure wins ({result['seizure_confidence']:.2f} > {result['fall_confidence']:.2f})")
            else:
                # Fall detected - check fall_type for severity
                if fall_type == 'slow_collapse':
                    result['alert_level'] = 'danger'  # danger: Đột quỵ - nằm bất động
                    print(f"⚖️ Both detected, SLOW COLLAPSE (stroke) - DANGER!")
                else:
                    result['alert_level'] = 'danger'  # danger: Té ngã xác nhận
                result['emergency_type'] = 'fall'
                print(f"⚖️ Both detected, fall wins ({result['fall_confidence']:.2f} > {result['seizure_confidence']:.2f})")
                
        elif result['seizure_detected']:
            result['alert_level'] = 'danger'  # danger: Co giật xác nhận
            result['emergency_type'] = 'seizure'
            self.stats['critical_alerts'] += 1
            
        elif result['fall_detected']:
            # Fall detected - classify based on fall_type
            if fall_type == 'slow_collapse':
                result['alert_level'] = 'danger'  # danger: Đột quỵ - nằm bất động
                result['emergency_type'] = 'fall_stroke'  # Specific type
            else:
                result['alert_level'] = 'danger'  # danger: Té ngã xác nhận
                result['emergency_type'] = 'fall'
                
        # WARNING: Có dấu hiệu nhưng chưa chắc chắn
        elif result['seizure_confidence'] > 0.45 and motion_level > 0.7:
            result['alert_level'] = 'warning'  # warning: Nghi ngờ co giật
            result['emergency_type'] = 'seizure_warning'
            
        elif result['fall_confidence'] >= 0.40 and result['fall_confidence'] < 0.60:
            result['alert_level'] = 'warning'  # warning: Nghi ngờ té
            result['emergency_type'] = 'fall_warning'
            
        # SUSPECT: Chuyển động đáng ngờ
        elif result['seizure_confidence'] >= 0.15 and result['seizure_confidence'] < 0.45:
            result['alert_level'] = 'suspect'  # suspect: Chuyển động bất thường
            result['emergency_type'] = 'abnormal_movement'
            
        elif result['fall_confidence'] >= 0.20 and result['fall_confidence'] < 0.40:
            result['alert_level'] = 'suspect'  # suspect: Có thể sắp té
            result['emergency_type'] = 'pre_fall_risk'
            
        # NORMAL: Confidence quá thấp = Không có sự cố = Bình thường
        else:
            # Luôn set normal nếu không rơi vào case nào ở trên
            result['alert_level'] = 'normal'  # normal: Hoạt động bình thường (không phát hiện sự cố)
            result['emergency_type'] = 'normal_activity'
        
        # 🔥 OPTIMIZED: SMART NORMAL LOGGING
        # Chiến lược tối ưu cho NORMAL để giảm storage:
        # 1. Chỉ log khi có CHUYỂN ĐỘNG (motion > threshold)
        # 2. Log mỗi 5 GIÂY thay vì 1 giây (giảm 80% storage)
        # 3. KHÔNG log snapshot nếu giống snapshot trước đó (similarity check)
        # 4. Aggregate summary mỗi 5 phút
        
        # NORMAL: Log bất cứ khi nào có motion (không giới hạn thời gian)
        # Tần suất sẽ được kiểm soát bởi duplicate check trong DB (5 giây)
        should_log_normal = motion_level > 0.01  # Chỉ cần có motion nhẹ là log
        
        # 🔥 DEBUG: Log để track NORMAL status mỗi 30 frames
        if result['alert_level'] == 'normal' and self.stats['total_frames'] % 30 == 0:
            print(f"📊 NORMAL Status: frames={self.stats['total_frames']}, motion={motion_level:.3f}, should_log={should_log_normal}")
            print(f"   Motion check: {motion_level:.3f} > 0.01 = {motion_level > 0.01}")
        
        if result['alert_level'] != 'normal' or should_log_normal:
            self.stats['total_alerts'] += 1
            self.stats['last_alert_time'] = time.time()
            self.stats['alert_type'] = result['alert_level']
            
            # 🔥 CAPTURE IMAGES FOR ALL STATUS LEVELS
            # Tất cả status đều chụp ảnh để lưu lại hình ảnh
            snapshot_ids = []
            try:
                if self.camera_id and self.user_id and frame is not None:
                    # Determine confidence based on alert level
                    if result['alert_level'] in ['danger']:
                        log_confidence = max(result['fall_confidence'], result['seizure_confidence'])
                    elif result['alert_level'] in ['warning']:
                        log_confidence = max(result['fall_confidence'], result['seizure_confidence'], 0.4)
                    elif result['alert_level'] in ['suspect']:
                        log_confidence = max(result['fall_confidence'], result['seizure_confidence'], 0.2)
                    else:
                        log_confidence = 0.1  # normal or unknown
                    
                    # 📸 CAPTURE IMAGES: 5 ảnh cho danger/warning, 1 ảnh cho suspect/normal/unknown
                    if result['alert_level'] in ['danger', 'warning']:
                        # Capture 5 images with multi-angle (if available)
                        print(f"📸 Capturing 5 images for {result['alert_level'].upper()} status...")
                        snapshot_ids = self.capture_5_snapshots_immediate(
                            current_frame=frame,
                            event_type=result.get('emergency_type', 'activity'),
                            confidence=log_confidence,
                            metadata={
                                'alert_level': result['alert_level'],
                                'fall_confidence': result['fall_confidence'],
                                'seizure_confidence': result['seizure_confidence'],
                                'motion_level': motion_level,
                                'fall_type': result.get('fall_type'),
                                'fall_duration': result.get('fall_duration'),
                                'fall_velocity': result.get('fall_velocity')
                            },
                            other_cameras=other_cameras if 'other_cameras' in locals() else None
                        )
                    else:
                        # 🔥 ALWAYS capture snapshot for ALL events (including low-motion NORMAL)
                        # Mọi event đều phải có ảnh để ghi lại trạng thái thực tế
                        print(f"📸 Capturing snapshot for {result['alert_level'].upper()} status (motion={motion_level:.3f})...")
                        if self.snapshot_service:
                                snapshot_id, _ = self.snapshot_service.create_detection_snapshot(
                                    camera_id=self.camera_id,
                                    user_id=self.user_id,
                                    event_type=result.get('emergency_type', 'normal_activity'),
                                    confidence=log_confidence,
                                    frame=frame,
                                    metadata={
                                        'alert_level': result['alert_level'],
                                        'fall_confidence': result['fall_confidence'],
                                        'seizure_confidence': result['seizure_confidence'],
                                        'motion_level': motion_level,
                                        'detection_timestamp': datetime.now().isoformat()
                                    }
                                )
                                snapshot_ids = [snapshot_id] if snapshot_id else []
            except Exception as snap_err:
                print(f"⚠️ Failed to capture snapshot: {snap_err}")
            
            # 🔥 Log ALL events to database with snapshot_id
            try:
                # Only log if we have valid IDs
                if self.camera_id and self.user_id:
                    # Create event data for all activities
                    activity_event_data = {
                        'event_type': result.get('emergency_type', 'normal_activity'),
                        'description': f"{result['alert_level'].upper()}: {result.get('emergency_type', 'activity')}",
                        'detection_data': {
                            'alert_level': result['alert_level'],
                            'fall_confidence': result['fall_confidence'],
                            'seizure_confidence': result['seizure_confidence'],
                            'motion_level': motion_level,
                            'fall_type': result.get('fall_type'),
                            'fall_duration': result.get('fall_duration'),
                            'fall_velocity': result.get('fall_velocity'),
                            'detection_timestamp': datetime.now().isoformat()
                        },
                        'confidence': log_confidence,
                        'context': {
                            'alert_level': result['alert_level'],
                            'emergency_type': result.get('emergency_type'),
                            'fall_type': result.get('fall_type'),
                            'fall_duration': result.get('fall_duration'),
                            'fall_velocity': result.get('fall_velocity'),
                            'motion_level': motion_level
                        },
                        'camera_id': self.camera_id,
                        'user_id': self.user_id
                    }
                    
                    # 🔥 FIX DUPLICATE: Log ONLY suspect/normal to DB
                    # danger/warning already handled in specific detection logic (line 334, 583)
                    if result['alert_level'] in ['suspect', 'normal']:
                        if result['alert_level'] != 'normal' or should_log_normal:
                            # Debug log - log mỗi 90 frames để không spam console
                            if result['alert_level'] == 'normal' and self.stats['total_frames'] % 90 == 0:
                                print(f"📝 Logging NORMAL event: motion={motion_level:.3f}, should_log={should_log_normal}")
                            
                            event_result = self.event_publisher.postgresql_service.publish_event_detection(activity_event_data)
                            
                            if event_result:
                                # Log success
                                event_id = event_result.get('event_id') if isinstance(event_result, dict) else str(event_result)
                                print(f"✅ {result['alert_level'].upper()} event logged: {event_id[:8]}...")
                                
                                # Link snapshot to event (if available)
                                if snapshot_ids and len(snapshot_ids) > 0:
                                    snapshot_id = snapshot_ids[0]
                                    try:
                                        self.event_publisher.postgresql_service.update_event_snapshot(event_id, snapshot_id)
                                        print(f"   📸 Linked {len(snapshot_ids)} snapshot(s)")
                                    except Exception as link_err:
                                        print(f"   ⚠️ Failed to link snapshot: {link_err}")
                                else:
                                    print(f"   ℹ️ No snapshot (motion too low)")
                            else:
                                print(f"❌ Failed to log {result['alert_level']} event")
                        
            except Exception as log_err:
                print(f"⚠️ Failed to log activity to database: {log_err}")
            
            # Send FCM notification ONLY for danger/warning (not suspect/normal/unknown)
            if result['alert_level'] in ['danger', 'warning']:
                self.send_emergency_notification(result)
            
        self.performance['total_detection_time'] = time.time() - start_time
        return result

    def calculate_motion_level_person(self, person_detections):
        """Calculate motion level based on person detections như file mẫu - FIXED"""
        # Use actual motion calculation instead of variance
        if hasattr(self, '_prev_frame') and self._prev_frame is not None:
            current_frame = getattr(self, '_current_frame', None)
            if current_frame is not None:
                motion = self.calculate_motion_level(self._prev_frame, current_frame)
                # DEBUG: Log motion calculation (disabled to reduce noise)
                # if self.stats['total_frames'] % 30 == 0:
                #     print(f"🎯 Motion Calculation: {motion:.3f}")
                return motion
        
        # Fallback: use motion variance if no frame data
        if not self.detection_history['motion_levels'] or len(self.detection_history['motion_levels']) < 2:
            return 0.1  # Default low motion
        
        recent_motions = self.detection_history['motion_levels'][-5:]
        if len(recent_motions) < 2:
            return 0.1
            
        motion_variance = np.var(recent_motions)
        return min(float(motion_variance * 10), 1.0)  # Scale and cap at 1.0

    def enhance_detection_with_motion(self, base_confidence, motion_level, detection_type):
        """Enhanced detection with motion analysis - MORE SENSITIVE VERSION"""
        # MORE GENEROUS: Lower thresholds and better enhancement
        if detection_type == 'fall':
            # Fall detection benefits from motion - MORE SENSITIVE
            if motion_level > 0.1:  # Giảm từ 0.3 xuống 0.1
                enhancement = min(0.3, motion_level * 0.5)  # Tăng enhancement
                return min(1.0, base_confidence + enhancement)
            elif motion_level < 0.02:  # Very very low motion
                return base_confidence * 0.95  # Minimal penalty
        elif detection_type == 'seizure':
            # Seizure detection - MORE SENSITIVE
            if motion_level > 0.05:  # Giảm từ 0.2 xuống 0.05
                enhancement = min(0.2, motion_level * 0.4)  # Tăng enhancement
                return min(1.0, base_confidence + enhancement)
            elif motion_level < 0.01:  # Very very low motion
                return base_confidence * 0.9  # Minimal penalty
        
        return base_confidence

    def smooth_detection_confidence(self, confidence, detection_type):
        """Smooth detection confidence over time như file mẫu"""
        history_key = f'{detection_type}_confidences'
        
        if history_key not in self.detection_history:
            self.detection_history[history_key] = []
            
        self.detection_history[history_key].append(confidence)
        if len(self.detection_history[history_key]) > 10:
            self.detection_history[history_key].pop(0)
            
        # Simple moving average
        return np.mean(self.detection_history[history_key])

    def save_alert_image(self, frame, alert_type, confidence=None):
        """Save alert image with optional confidence như file mẫu"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            confidence_str = f"_conf_{confidence:.2f}" if confidence is not None else ""
            filename = f"{alert_type}_{timestamp}{confidence_str}.jpg"
            filepath = os.path.join(self.alert_save_path, filename)
            cv2.imwrite(filepath, frame)
            print(f"Alert image saved: {filepath}")
        except Exception as e:
            print(f"Error saving alert image: {e}")

    def update_statistics(self, detection_result, person_count):
        """Update simplified statistics"""
        # Frame counting
        self.stats['frames_processed'] += 1
        self.stats['persons_detected'] += person_count
        
        # FPS calculation 
        current_time = time.time()
        self.stats['frame_times'].append(current_time)
        if len(self.stats['frame_times']) > 30:
            self.stats['frame_times'].pop(0)
            
        if len(self.stats['frame_times']) > 1:
            time_diff = self.stats['frame_times'][-1] - self.stats['frame_times'][0]
            self.stats['fps'] = len(self.stats['frame_times']) / time_diff if time_diff > 0 else 0

    def calculate_motion_level(self, prev_frame, current_frame):
        """Calculate motion level between frames như file mẫu"""
        if prev_frame is None:
            return 0.0
            
        # Resize frames for faster processing
        height, width = prev_frame.shape[:2]
        small_height, small_width = height // 4, width // 4
        
        prev_small = cv2.resize(prev_frame, (small_width, small_height))
        curr_small = cv2.resize(current_frame, (small_width, small_height))
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Calculate motion level as percentage of changed pixels
        motion_pixels = np.sum(diff > 30)  # Threshold for significant change
        total_pixels = diff.shape[0] * diff.shape[1]
        motion_level = motion_pixels / total_pixels
        
        return motion_level

    def visualize_dual_detection(self, frame, detection_result, person_detections):
        """Visualize dual detection với full overlay như file mẫu"""
        frame_vis = frame.copy()
        
        # Vẽ bounding box cho từng người
        for person in person_detections:
            bbox = person['bbox']
            confidence = person.get('confidence', 0)
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)
            if detection_result.get('alert_level') == 'critical':
                color = (0, 0, 255)
            elif detection_result.get('alert_level') == 'high':
                color = (0, 165, 255)
            elif detection_result.get('alert_level') == 'warning':
                color = (0, 255, 255)
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_vis, f"Person: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Vẽ keypoints với improved accuracy
        keypoints = detection_result.get('keypoints')
        if keypoints is not None:
            conf_threshold = 0.5  # Increased threshold for better accuracy
            
            # Draw keypoints with better color coding
            for i, (kx, ky, kconf) in enumerate(keypoints):
                if kconf > conf_threshold:
                    # Improved color coding based on confidence
                    if kconf > 0.8:
                        color = (0, 255, 0)    # Bright green for very high confidence
                    elif kconf > 0.6:
                        color = (0, 255, 255)  # Yellow for high confidence  
                    else:
                        color = (0, 165, 255)  # Orange for medium confidence
                    
                    # Size based on confidence
                    radius = 4 if kconf > 0.7 else 3
                    cv2.circle(frame_vis, (int(kx), int(ky)), radius, color, -1)
                    
                    # Better keypoint labels (COCO-17 format)
                    keypoint_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 
                                     'l_shldr', 'r_shldr', 'l_elbow', 'r_elbow', 
                                     'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 
                                     'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
                    
                    if i < len(keypoint_names) and kconf > 0.6:
                        cv2.putText(frame_vis, keypoint_names[i], (int(kx), int(ky-8)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

            # Improved skeleton connections
            connections = [
                # Head connections
                (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes, eyes-ears
                
                # Torso connections  
                (5, 6),   # shoulder to shoulder
                (5, 11), (6, 12),  # shoulders to hips
                (11, 12), # hip to hip
                
                # Arms
                (5, 7), (7, 9),   # left arm: shoulder -> elbow -> wrist
                (6, 8), (8, 10),  # right arm: shoulder -> elbow -> wrist
                
                # Legs
                (11, 13), (13, 15), # left leg: hip -> knee -> ankle
                (12, 14), (14, 16), # right leg: hip -> knee -> ankle
            ]
            
            # Draw skeleton with color coding for body parts
            for i, (p1, p2) in enumerate(connections):
                if (p1 < len(keypoints) and p2 < len(keypoints) and 
                    keypoints[p1][2] > conf_threshold and keypoints[p2][2] > conf_threshold):
                    
                    pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                    pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                    
                    # Color coding for different body parts
                    if i < 4:  # Head connections
                        color = (255, 0, 255)  # Magenta
                    elif i < 8:  # Torso connections
                        color = (0, 255, 255)  # Cyan
                    elif i < 10: # Left arm
                        color = (0, 255, 0)    # Green
                    elif i < 12: # Right arm
                        color = (255, 255, 0)  # Yellow
                    elif i < 14: # Left leg
                        color = (255, 0, 0)    # Blue
                    else:        # Right leg
                        color = (0, 0, 255)    # Red
                    
                    # Thickness based on confidence
                    min_conf = min(keypoints[p1][2], keypoints[p2][2])
                    thickness = 3 if min_conf > 0.7 else 2
                    cv2.line(frame_vis, pt1, pt2, color, thickness)

        # Vẽ thông tin cảnh báo, motion, confirmation frames
        alert_y = 30
        if detection_result.get('fall_detected'):
            cv2.putText(frame_vis, f"FALL DETECTED: {detection_result['fall_confidence']:.2f}", (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        elif detection_result.get('fall_confidence', 0) > 0.25:
            cv2.putText(frame_vis, f"Fall Warning: {detection_result['fall_confidence']:.2f}", (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += 30

        if detection_result.get('seizure_detected'):
            cv2.putText(frame_vis, f"SEIZURE DETECTED: {detection_result['seizure_confidence']:.2f}", (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        elif detection_result.get('seizure_confidence', 0) > 0.5 and self.detection_history['motion_levels']:
            current_motion = self.detection_history['motion_levels'][-1]
            if current_motion > 0.8:
                cv2.putText(frame_vis, f"⚠️ Seizure Warning: {detection_result['seizure_confidence']:.2f}", (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                alert_y += 30

        if self.detection_history['motion_levels']:
            current_motion = self.detection_history['motion_levels'][-1]
            motion_color = (0, 255, 0) if current_motion < 0.3 else (0, 255, 255) if current_motion < 0.7 else (0, 0, 255)
            cv2.putText(frame_vis, f"Motion Level: {current_motion:.2f}", (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
            alert_y += 25

        fall_conf = self.detection_history.get('fall_confirmation_frames', 0)
        seizure_conf = self.detection_history.get('seizure_confirmation_frames', 0)
        if fall_conf > 0 or seizure_conf > 0:
            cv2.putText(frame_vis, f"🔍 Conf Frames - Fall:{fall_conf} Seizure:{seizure_conf}", (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            alert_y += 20

        # Hiển thị buffer seizure nếu chưa đủ window
        if detection_result.get('seizure_ready') is False:
            frames_needed = 25 - len(self.detection_history['seizure_confidences'])
            cv2.putText(frame_vis, f"📊 Seizure Buffer: {frames_needed} frames needed", (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return frame_vis

    def create_normal_camera_window(self, frame, person_detections):
        """Create normal camera window như file mẫu"""
        frame_normal = frame.copy()
        
        for person in person_detections:
            bbox = person['bbox']
            confidence = person.get('confidence', 0)
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)
            cv2.rectangle(frame_normal, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_normal, f"Person: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame_normal, "Normal Camera View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        person_count = len(person_detections)
        status_text = f"Persons Detected: {person_count}"
        cv2.putText(frame_normal, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_normal, timestamp, (10, frame_normal.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_normal

    def draw_statistics_overlay(self, frame, stats):
        """Simplified statistics overlay - only essential info"""
        frame_vis = frame.copy()
        h, w = frame.shape[:2]
        panel_width = 250
        panel_height = 180
        panel_x = w - panel_width - 10
        panel_y = 10

        # Semi-transparent background
        overlay = frame_vis.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        frame_vis = cv2.addWeighted(frame_vis, 0.7, overlay, 0.3, 0)

        # Calculate runtime
        runtime = time.time() - stats['start_time']
        
        # Simplified statistics text
        stats_text = [
            "🏥 HEALTHCARE MONITOR",
            f"Runtime: {runtime/60:.1f}min | FPS: {stats['fps']:.1f}",
            f"Frames: {stats['frames_processed']}/{stats['total_frames']}",
            "",
            "🩹 FALL DETECTION:",
            f"Detected: {stats['fall_detections']}",
            f"Frames: {self.detection_history.get('fall_confirmation_frames', 0)}",
            "",
            "🧠 SEIZURE DETECTION:",
            f"Detected: {stats['seizure_detections']}",
            f"Warnings: {stats['seizure_warnings']}",
            f"Frames: {self.detection_history.get('seizure_confirmation_frames', 0)}",
            "",
            " ALERTS:",
            f"Critical: {stats['critical_alerts']} | Total: {stats['total_alerts']}",
            f"Status: {stats['alert_type']}"
        ]

        # Draw statistics
        text_y = panel_y + 15
        for line in stats_text:
            if line == "":
                text_y += 5
                continue
            color = (255, 255, 255)
            if "🚨" in line:
                color = (0, 0, 255)
            elif "🩹" in line or "🧠" in line:
                color = (0, 255, 255)
            cv2.putText(frame_vis, line, (panel_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            text_y += 12

        return frame_vis

    def print_final_statistics(self):
        """Print simplified final statistics"""
        runtime = time.time() - self.stats['start_time']
        print("\n" + "="*50)
        print("🏥 HEALTHCARE MONITOR - FINAL STATISTICS")
        print("="*50)
        print(f"📊 Runtime: {runtime/60:.1f} minutes | FPS: {self.stats['fps']:.1f}")
        print(f"📊 Frames: {self.stats['frames_processed']}/{self.stats['total_frames']}")
        print(f"📊 Persons Detected: {self.stats['persons_detected']}")
        print()
        print("🩹 FALL DETECTION:")
        print(f"   Detected: {self.stats['fall_detections']}")
        print(f"   Last: {datetime.fromtimestamp(self.stats['last_fall_time']).strftime('%H:%M:%S') if self.stats['last_fall_time'] else 'None'}")
        print()
        print("🧠 SEIZURE DETECTION:")
        print(f"   Detected: {self.stats['seizure_detections']}")
        print(f"   Warnings: {self.stats['seizure_warnings']}")
        print(f"   Last: {datetime.fromtimestamp(self.stats['last_seizure_time']).strftime('%H:%M:%S') if self.stats['last_seizure_time'] else 'None'}")
        print()
        print("🚨 ALERTS:")
        print(f"   Critical: {self.stats['critical_alerts']} | Total: {self.stats['total_alerts']}")
        print(f"   Status: {self.stats['alert_type']}")
        print("="*50)

    def send_emergency_notification(self, detection_result):
        """
        Log emergency events (notifications handled by NestJS backend)
        
        Args:
            detection_result: Detection result containing event info
        """
        try:
            # Only log critical and high alerts
            alert_level = detection_result.get('alert_level', 'normal')
            emergency_type = detection_result.get('emergency_type')
            
            if alert_level in ['critical', 'high'] and emergency_type:
                
                if emergency_type in ['fall', 'fall_detected']:
                    confidence = detection_result.get('fall_confidence', 0.0)
                    event_type = 'fall'
                elif emergency_type in ['seizure', 'seizure_detected']:
                    confidence = detection_result.get('seizure_confidence', 0.0) 
                    event_type = 'seizure'
                else:
                    return  # Skip other types for now
                
                # Prepare additional data
                additional_data = {
                    'alert_level': alert_level,
                    'emergency_type': emergency_type,
                    'location': 'Healthcare Room',  # You can customize this
                    'camera_id': getattr(self.camera, 'camera_id', 'unknown'),
                    'detection_time': datetime.now().strftime('%H:%M:%S')
                }
                
                # Log emergency event (FCM removed - handled by NestJS backend)
                print(f"🚨 Emergency Event Detected: {event_type} (confidence: {confidence:.2f})")
                print(f"   Alert Level: {alert_level}")
                print(f"   Location: {additional_data['location']}")
                print(f"   Time: {additional_data['detection_time']}")
                
        except Exception as e:
            print(f"❌ Emergency Event Logging Error: {e}")

    def capture_5_snapshots_immediate(self, current_frame, event_type, confidence, event_id=None, metadata=None, other_cameras=None):
        """
        🔥 CAPTURE NGAY TẠI THỜI ĐIỂM PHÁT HIỆN EVENT!
        - Image 1: Frame hiện tại (đang phát hiện)
        - Image 2-5: MULTI-ANGLE (if same room) or TEMPORAL (if different rooms)
        
        Args:
            current_frame: Frame đang phát hiện event (QUAN TRỌNG!)
            event_type: fall, seizure, etc.
            confidence: Detection confidence
            event_id: Event ID to link
            metadata: Additional metadata
            other_cameras: List of other camera dicts in same room (for multi-angle)
            
        Returns:
            list of snapshot IDs
        """
        if not self.camera_id or not self.user_id:
            print(f"⚠️ Cannot save snapshots - missing camera_id or user_id")
            return []
        
        snapshot_ids = []
        
        # Determine capture mode
        has_multi_camera = other_cameras and len(other_cameras) > 0
        capture_mode = 'multi_angle' if has_multi_camera else 'temporal'
        
        if has_multi_camera:
            print(f"📸📹 MULTI-ANGLE CAPTURE: {len(other_cameras) + 1} cameras (same room)")
        else:
            print(f"📸 TEMPORAL CAPTURE: Single camera")
        
        try:
            # Create ONE snapshot record for all 5 images
            main_snapshot_id = None
            if self.snapshot_service and current_frame is not None:
                snapshot_metadata = {
                    'detection_time': datetime.now().isoformat(),
                    'total_images': 5,
                    'capture_mode': capture_mode,
                    'multi_angle': has_multi_camera,
                    'camera_count': len(other_cameras) + 1 if has_multi_camera else 1,
                    **(metadata or {})
                }
                
                if event_id:
                    snapshot_metadata['event_id'] = event_id
                
                # 🎬 LẤY 2 FRAMES TRƯỚC ĐÓ TỪ BUFFER để thấy hành động dẫn tới sự kiện
                frames_to_capture = []
                
                # Lấy tối đa 2 frames trước từ buffer
                buffer_len = len(self._frame_buffer)
                if buffer_len >= 3:
                    # Có đủ buffer: lấy frame-2 và frame-1 (2 frames trước current)
                    frames_to_capture = [
                        ('frame_minus_2', self._frame_buffer[-3], -2 * 33),  # ~66ms trước (2 frames @ 30fps)
                        ('frame_minus_1', self._frame_buffer[-2], -1 * 33),  # ~33ms trước (1 frame)
                        ('current', current_frame, 0)  # Frame hiện tại
                    ]
                    print(f"📸 Using buffer: [frame-2, frame-1, current] for temporal context")
                elif buffer_len >= 2:
                    # Có 2 frames: lấy frame-1 và current
                    frames_to_capture = [
                        ('frame_minus_1', self._frame_buffer[-2], -1 * 33),
                        ('current', current_frame, 0)
                    ]
                    print(f"📸 Using buffer: [frame-1, current] (limited buffer)")
                else:
                    # Không đủ buffer: chỉ dùng current frame
                    frames_to_capture = [('current', current_frame, 0)]
                    print(f"⚠️ Insufficient buffer, using current frame only")
                
                # 🔥 IMAGE 1 (PRIMARY): Frame-2 (hoặc frame sớm nhất có)
                primary_label, primary_frame, time_offset = frames_to_capture[0]
                main_snapshot_id, first_image_id = self.snapshot_service.create_detection_snapshot(
                    camera_id=self.camera_id,
                    user_id=self.user_id,
                    event_type=event_type,
                    confidence=confidence,
                    frame=primary_frame,  # Frame trước sự kiện!
                    metadata={
                        **snapshot_metadata, 
                        'sequence_number': 1, 
                        'camera_angle': 'primary',
                        'time_offset_ms': time_offset,
                        'frame_label': primary_label,
                        'is_detection_frame': (primary_label == 'current')
                    }
                )
                snapshot_ids.append(main_snapshot_id)
                print(f"✅ Image 1/5 ({primary_label}): {main_snapshot_id[:8]}...")
                
                # 🎬 IMAGE 2-3: Lấy tiếp từ buffer nếu có (frame-1 và current frame)
                image_sequence = 2
                for idx in range(1, min(len(frames_to_capture), 3)):  # Max 2 frames nữa (để có 3 total)
                    label, frame, time_offset = frames_to_capture[idx]
                    
                    self.snapshot_service.add_image_to_snapshot(
                        snapshot_id=main_snapshot_id,
                        frame=frame,
                        camera_id=self.camera_id,
                        user_id=self.user_id,
                        event_type=event_type,
                        confidence=confidence,
                        is_primary=(label == 'current'),
                        metadata={
                            **snapshot_metadata,
                            'sequence_number': image_sequence,
                            'camera_angle': 'primary',
                            'time_offset_ms': time_offset,
                            'frame_label': label,
                            'is_detection_frame': (label == 'current')
                        }
                    )
                    print(f"✅ Image {image_sequence}/5 ({label})")
                    image_sequence += 1
            
            # 🔥 IMAGE 4-5: MULTI-ANGLE or TEMPORAL
            if main_snapshot_id:
                if has_multi_camera:
                    # 📹 MULTI-ANGLE MODE: Capture từ nhiều camera
                    self._capture_multi_angle_images(
                        main_snapshot_id, event_type, confidence,
                        other_cameras, snapshot_metadata, start_sequence=image_sequence
                    )
                else:
                    # ⏱️ TEMPORAL MODE: Capture theo thời gian
                    self._capture_temporal_images(
                        main_snapshot_id, event_type, confidence, snapshot_metadata, start_sequence=image_sequence
                    )
                
                print(f"📸 Captured 1 snapshot with {image_sequence + 2} images ({capture_mode} mode)!")
            
        except Exception as e:
            print(f"❌ Error capturing immediate snapshots: {e}")
            import traceback
            traceback.print_exc()
        
        return snapshot_ids

    def _capture_multi_angle_images(self, snapshot_id, event_type, confidence, 
                                    other_cameras, snapshot_metadata, start_sequence=2):
        """
        Capture Images 4-5 (or from start_sequence) from multiple cameras (same room)
        
        Strategy:
        - Image N: Camera 2 @ T+0.0s (other angle, simultaneous)
        - Image N+1: Camera 1 @ T+0.5s
        - Image N+2: Camera 2 @ T+0.5s (if needed)
        """
        try:
            # Setup cameras
            primary_rtsp = getattr(self.camera, 'rtsp_url', None)
            primary_cap = cv2.VideoCapture(primary_rtsp) if primary_rtsp else None
            
            # Get first other camera
            other_camera = other_cameras[0] if other_cameras else None
            other_cap = None
            
            if other_camera and 'rtsp_url' in other_camera:
                other_cap = cv2.VideoCapture(other_camera['rtsp_url'])
                print(f"🎥 Opened secondary camera: {other_camera.get('camera_id', 'unknown')}")
            
            # IMAGE N: Other camera @ T+0
            seq_num = start_sequence
            if other_cap and other_cap.isOpened():
                ret, frame = other_cap.read()
                if ret and frame is not None:
                    self.snapshot_service.add_image_to_snapshot(
                        snapshot_id=snapshot_id,
                        frame=frame,
                        camera_id=other_camera.get('camera_id'),
                        user_id=self.user_id,
                        event_type=event_type,
                        confidence=confidence,
                        is_primary=False,
                        metadata={
                            'sequence_number': seq_num,
                            'camera_angle': 'secondary',
                            'time_offset_ms': 0,
                            'capture_method': 'multi_angle'
                        }
                    )
                    print(f"✅ Image {seq_num}/5 (Cam2 T+0.0s)")
                    seq_num += 1
            
            # Sleep 0.5s before next batch
            time.sleep(0.5)
            
            # IMAGE N+1: Primary camera @ T+0.5s
            if primary_cap and primary_cap.isOpened() and seq_num <= 5:
                ret, frame = primary_cap.read()
                if ret and frame is not None:
                    self.snapshot_service.add_image_to_snapshot(
                        snapshot_id=snapshot_id,
                        frame=frame,
                        camera_id=self.camera_id,
                        user_id=self.user_id,
                        event_type=event_type,
                        confidence=confidence,
                        is_primary=False,
                        metadata={
                            'sequence_number': seq_num,
                            'camera_angle': 'primary',
                            'time_offset_ms': 500,
                            'capture_method': 'multi_angle'
                        }
                    )
                    print(f"✅ Image {seq_num}/5 (Cam1 T+0.5s)")
                    seq_num += 1
            
            # IMAGE N+2: Other camera @ T+0.5s (if needed)
            if other_cap and other_cap.isOpened() and seq_num <= 5:
                ret, frame = other_cap.read()
                if ret and frame is not None:
                    self.snapshot_service.add_image_to_snapshot(
                        snapshot_id=snapshot_id,
                        frame=frame,
                        camera_id=other_camera.get('camera_id'),
                        user_id=self.user_id,
                        event_type=event_type,
                        confidence=confidence,
                        is_primary=False,
                        metadata={
                            'sequence_number': seq_num,
                            'camera_angle': 'secondary',
                            'time_offset_ms': 500,
                            'capture_method': 'multi_angle'
                        }
                    )
                    print(f"✅ Image {seq_num}/5 (Cam2 T+0.5s)")
                    seq_num += 1
            
            # Cleanup
            if primary_cap:
                primary_cap.release()
            if other_cap:
                other_cap.release()
                
        except Exception as e:
            print(f"❌ Multi-angle capture error: {e}")
            import traceback
            traceback.print_exc()

    def _capture_temporal_images(self, snapshot_id, event_type, confidence, snapshot_metadata, start_sequence=2):
        """
        Capture remaining images (from start_sequence to 5) from single camera over time
        
        Strategy:
        - Capture additional frames at 0.3s intervals to reach 5 total images
        """
        try:
            rtsp_url = getattr(self.camera, 'rtsp_url', None)
            if not rtsp_url:
                print("⚠️ No RTSP URL for temporal capture")
                return
            
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"⚠️ Could not open RTSP for temporal capture")
                return
            
            # Capture remaining images (from start_sequence to 5)
            num_images_needed = 5 - start_sequence + 1
            for i in range(num_images_needed):
                seq_num = start_sequence + i
                if seq_num > 5:  # Don't exceed 5 images
                    break
                
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    print(f"⚠️ Failed to capture frame {seq_num}/5 from RTSP")
                    time.sleep(0.3)
                    continue
                
                # Add image to snapshot
                try:
                    self.snapshot_service.add_image_to_snapshot(
                        snapshot_id=snapshot_id,
                        frame=frame,
                        camera_id=self.camera_id,
                        user_id=self.user_id,
                        event_type=event_type,
                        confidence=confidence,
                        is_primary=False,
                        metadata={
                            'sequence_number': seq_num,
                            'camera_angle': 'primary',
                            'time_offset_ms': i * 300,
                            'capture_method': 'temporal'
                        }
                    )
                    print(f"✅ Image {seq_num}/5 (T+{i*0.3:.1f}s)")
                except Exception as e:
                    print(f"❌ Failed to add image {i+1}/5: {e}")
                
                # Delay before next capture
                if i < 4:
                    time.sleep(0.3)
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Temporal capture error: {e}")
            import traceback
            traceback.print_exc()

    def capture_5_snapshots_from_rtsp(self, event_type, confidence, event_id=None, metadata=None):
        """
        Capture 5 snapshots directly from RTSP stream (no processing lag)
        Giãn cách đều trong khoảng 2 giây (mỗi 0.5s chụp 1 ảnh)
        
        Args:
            event_type: Type of detection (fall, seizure, etc.)
            confidence: Detection confidence
            event_id: Event ID to link snapshots to (optional)
            metadata: Additional metadata
            
        Returns:
            list of snapshot IDs
        """
        if not self.camera_id or not self.user_id:
            print(f"⚠️ Cannot save snapshots - missing camera_id or user_id")
            return []
        
        snapshot_ids = []
        rtsp_url = getattr(self.camera, 'rtsp_url', None)
        
        if not rtsp_url:
            print(f"⚠️ No RTSP URL available for camera")
            return []
        
        print(f"📸 Capturing 5 snapshots from RTSP for {event_type}...")
        
        try:
            # Open RTSP stream directly
            cap = cv2.VideoCapture(rtsp_url)
            
            if not cap.isOpened():
                print(f"❌ Failed to open RTSP stream: {rtsp_url}")
                return []
            
            # Create ONE snapshot record for all 5 images
            main_snapshot_id = None
            if self.snapshot_service:
                # Create the main snapshot record first
                snapshot_metadata = {
                    'detection_time': datetime.now().isoformat(),
                    'total_images': 5,
                    **(metadata or {})
                }
                
                if event_id:
                    snapshot_metadata['event_id'] = event_id
                
                # Capture first frame to create snapshot
                ret, first_frame = cap.read()
                if ret and first_frame is not None:
                    main_snapshot_id, first_image_id = self.snapshot_service.create_detection_snapshot(
                        camera_id=self.camera_id,
                        user_id=self.user_id,
                        event_type=event_type,
                        confidence=confidence,
                        frame=first_frame,
                        metadata={**snapshot_metadata, 'sequence_number': 1}
                    )
                    snapshot_ids.append(main_snapshot_id)
                    print(f"✅ Snapshot created: {main_snapshot_id[:8]}... (Image 1/5)")
                    time.sleep(0.5)
            
            # Capture remaining 4 frames and add to same snapshot
            if main_snapshot_id:
                for i in range(1, 5):  # Images 2-5
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        print(f"⚠️ Failed to capture frame {i+1}/5")
                        continue
                    
                    # Add image to existing snapshot
                    try:
                        image_id = self.snapshot_service.add_image_to_snapshot(
                            snapshot_id=main_snapshot_id,
                            frame=frame,
                            camera_id=self.camera_id,
                            user_id=self.user_id,
                            event_type=event_type,
                            confidence=confidence,
                            is_primary=False,
                            metadata={
                                'sequence_number': i + 1,
                                'detection_time': datetime.now().isoformat()
                            }
                        )
                        print(f"✅ Image {i+1}/5 added: {image_id[:8]}...")
                    except Exception as e:
                        print(f"❌ Failed to add image {i+1}/5: {e}")
                    
                    # Delay 0.5s before next capture (except last frame)
                    if i < 4:
                        time.sleep(0.5)
            
            cap.release()
            print(f"📸 Captured 1 snapshot with {5 if main_snapshot_id else 0} images successfully!")
            
        except Exception as e:
            print(f"❌ Error capturing RTSP snapshots: {e}")
        
        return snapshot_ids

    def save_detection_snapshot(self, frame, event_type, confidence, metadata=None):
        """
        Save detection snapshot to MinIO and database
        
        Args:
            frame: OpenCV frame to save
            event_type: Type of detection (fall, seizure, etc.)
            confidence: Detection confidence
            metadata: Additional metadata
        """
        if not self.camera_id or not self.user_id:
            print(f"⚠️ Cannot save snapshot - missing camera_id or user_id")
            return None
            
        try:
            if self.snapshot_service:
                snapshot_id, image_id = self.snapshot_service.create_detection_snapshot(
                    camera_id=self.camera_id,
                    user_id=self.user_id,
                    event_type=event_type,
                    confidence=confidence,
                    frame=frame,
                    metadata={
                        'detection_time': datetime.now().isoformat(),
                        'frame_number': self.stats['total_frames'],
                        'processing_stats': {
                            'fps': self.stats['fps'],
                            'total_detections': self.stats[f'{event_type}_detections']
                        },
                        **(metadata or {})
                    }
                )
                
                print(f"📸 {event_type.upper()} snapshot saved: {snapshot_id[:8]}... (confidence: {confidence:.3f})")
                return snapshot_id
            else:
                print(f"⚠️ Snapshot service not available - saving locally only")
                # Fallback: save to local alerts folder
                import cv2
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{event_type}_{self.camera_id}_{timestamp}_{confidence:.3f}.jpg"
                local_path = os.path.join(self.alert_save_path, filename)
                cv2.imwrite(local_path, frame)
                print(f"📸 {event_type.upper()} image saved locally: {filename}")
                return f"local_{timestamp}"
                
        except Exception as e:
            print(f"❌ Error saving {event_type} snapshot: {e}")
            # Fallback: save to local alerts folder
            try:
                import cv2
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{event_type}_{self.camera_id}_{timestamp}_{confidence:.3f}_fallback.jpg"
                local_path = os.path.join(self.alert_save_path, filename)
                cv2.imwrite(local_path, frame)
                print(f"📸 {event_type.upper()} image saved locally (fallback): {filename}")
                return f"fallback_{timestamp}"
            except Exception as fallback_error:
                print(f"❌ Fallback save also failed: {fallback_error}")
                return None
