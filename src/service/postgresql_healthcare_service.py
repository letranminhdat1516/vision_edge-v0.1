"""
PostgreSQL Direct Connection Service for Healthcare Monitoring
Uses session pooler for IPv4 compatibility
"""

import uuid
import json
import logging
import time
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from psycopg2 import pool
import threading
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration
from service.database_config_service import config_loader

try:
    from config.supabase_config import supabase_config
except ImportError:
    try:
        from src.config.supabase_config import supabase_config
    except ImportError:
        # Fallback configuration if config module not available
        class FallbackConfig:
            def __init__(self):
                self.database_url = os.getenv('DATABASE_URL', '')
        supabase_config = FallbackConfig()

logger = logging.getLogger(__name__)

class PostgreSQLHealthcareService:
    """Direct PostgreSQL service for healthcare events using session pooler"""
    
    def __init__(self):
        self.database_url = supabase_config.database_url
        self.connection_pool = None
        
        # Initialize Vietnamese Caption Service for alert messages
        try:
            from service.ai_vision_description_service import ProfessionalVietnameseCaptionPipeline
            self.vietnamese_caption = ProfessionalVietnameseCaptionPipeline()
            logger.info("📝 Vietnamese Caption Service: Enabled for alert messages")
        except ImportError as e:
            self.vietnamese_caption = None
            logger.warning(f"📝 Vietnamese Caption Service: Disabled - {e}")
        
        # Initialize Snapshot Service for MinIO image upload
        try:
            from infrastructure.services.snapshot_service import get_snapshot_service
            self.snapshot_service = get_snapshot_service(self.database_url)
            logger.info("📸 Snapshot Service: Enabled for MinIO image upload")
        except ImportError as e:
            self.snapshot_service = None
            logger.warning(f"📸 Snapshot Service: Disabled - {e}")
        self.is_connected = False
        self.polling_threads = {}
        self.event_handlers = {}
        self.last_check_times = {}
        
        # Alternative connection parameters
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD') 
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'postgres')
        
        # Initialize connection
        self._initialize_connection()
        
        # Note: We now use real database cameras instead of ensuring default entities
    
    def _initialize_connection(self):
        """Initialize PostgreSQL connection pool"""
        try:
            # Try individual parameters first (preferred for pooler)
            if self.db_host and self.db_user:
                logger.info("Attempting connection using individual parameters")
                try:
                    self.connection_pool = SimpleConnectionPool(
                        minconn=1,
                        maxconn=10,
                        host=self.db_host,
                        port=int(self.db_port),
                        database=self.db_name,
                        user=self.db_user,
                        password=self.db_password,
                        cursor_factory=RealDictCursor,
                        connect_timeout=10
                    )
                    
                    # Test connection
                    conn = self.connection_pool.getconn()
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        if result:
                            logger.info("✅ PostgreSQL connected successfully via pooler")
                            self.is_connected = True
                            self.return_connection(conn)
                            return
                    
                    self.return_connection(conn)
                    
                except Exception as e:
                    logger.warning(f"Individual parameters connection failed: {e}")
                    if self.connection_pool:
                        try:
                            self.connection_pool.closeall()
                        except:
                            pass
                        self.connection_pool = None
            
            # Fallback to DATABASE_URL parsing
            if not self.database_url:
                logger.error("No valid connection parameters configured")
                return
            
            logger.info("Attempting connection using DATABASE_URL")
            
            # Try IPv6 if hostname resolution fails
            original_url = self.database_url
            ipv6_url = original_url.replace('db.undznprwlqjpnxqsgyiv.supabase.co', '[2406:da18:243:7412:68f3:999f:785b:e90d]')
            
            for attempt, url in enumerate([original_url, ipv6_url], 1):
                try:
                    logger.info(f"Attempting URL connection {attempt}/2: {'original' if attempt == 1 else 'IPv6'}")
                    
                    # Parse database URL
                    parsed = urlparse(url)
                    
                    # Create connection pool với size lớn hơn để tránh exhausted
                    self.connection_pool = SimpleConnectionPool(
                        minconn=2,
                        maxconn=50,  # Tăng từ 10 lên 50 connections
                        host=parsed.hostname,
                        port=parsed.port or 5432,
                        database=parsed.path[1:] if parsed.path else 'postgres',
                        user=parsed.username,
                        password=parsed.password,
                        cursor_factory=RealDictCursor,
                        connect_timeout=10
                    )
                    
                    # Test connection
                    conn = self.connection_pool.getconn()
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        if result:
                            logger.info(f"✅ PostgreSQL connected successfully via URL {'original' if attempt == 1 else 'IPv6'}")
                            self.is_connected = True
                            self.return_connection(conn)
                            return
                        else:
                            logger.error("❌ PostgreSQL connection test failed")
                    
                    self.return_connection(conn)
                    
                except Exception as e:
                    logger.warning(f"URL connection attempt {attempt} failed: {e}")
                    if self.connection_pool:
                        try:
                            self.connection_pool.closeall()
                        except:
                            pass
                        self.connection_pool = None
                    continue
            
            logger.error("❌ All connection attempts failed")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL connection: {e}")
            self.is_connected = False
    
    def get_connection(self):
        """Get connection from pool"""
        if self.connection_pool:
            return self.connection_pool.getconn()
        return None
    
    def return_connection(self, conn):
        """Return connection to pool"""
        if self.connection_pool and conn:
            self.connection_pool.putconn(conn)
    
    def subscribe_to_events(self, table: str, event_type: str, handler):
        """Subscribe to table changes using polling"""
        if not self.is_connected:
            logger.error("PostgreSQL not connected")
            return
        
        try:
            subscription_key = f"{table}_{event_type}"
            self.event_handlers[subscription_key] = handler
            self.last_check_times[subscription_key] = datetime.now(timezone.utc)
            
            # Start polling thread
            polling_thread = threading.Thread(
                target=self._poll_table_changes,
                args=(table, event_type, handler),
                daemon=True
            )
            polling_thread.start()
            
            self.polling_threads[subscription_key] = polling_thread
            logger.info(f"✅ Started polling for {table} {event_type} events")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {table}: {e}")
    
    def _poll_table_changes(self, table: str, event_type: str, handler):
        """Poll table for new records"""
        subscription_key = f"{table}_{event_type}"
        
        while subscription_key in self.polling_threads:
            try:
                if not self.is_connected:
                    logger.warning(f"Not connected, stopping poll for {subscription_key}")
                    break
                
                conn = self.get_connection()
                if not conn:
                    logger.error("Could not get database connection")
                    time.sleep(5)
                    continue
                
                try:
                    last_check = self.last_check_times.get(subscription_key)
                    
                    with conn.cursor() as cursor:
                        if event_type in ['INSERT', '*']:
                            # Query for new records since last check
                            if last_check:
                                cursor.execute(
                                    f"SELECT * FROM {table} WHERE created_at > %s ORDER BY created_at ASC",
                                    (last_check,)
                                )
                            else:
                                # First time - get latest 5 records
                                cursor.execute(
                                    f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT 5"
                                )
                            
                            records = cursor.fetchall()
                            
                            for record in records:
                                # Convert record to dict
                                record_dict = dict(record) if record else {}
                                
                                event_data = {
                                    'event_type': 'INSERT',
                                    'table': table,
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'new_data': record_dict,
                                    'old_data': {}
                                }
                                
                                # Call handler in separate thread
                                threading.Thread(
                                    target=handler,
                                    args=(event_data,),
                                    daemon=True
                                ).start()
                    
                    # Update last check time
                    self.last_check_times[subscription_key] = datetime.now(timezone.utc)
                    
                finally:
                    self.return_connection(conn)
                
                # Wait before next poll
                time.sleep(3)  # Poll every 3 seconds
                
            except Exception as e:
                logger.error(f"Error polling {table}: {e}")
                time.sleep(5)
    
    def _get_user_camera_id(self, user_id: str) -> Optional[str]:
        """Get first camera_id for a user"""
        print(f"🔍 DEBUG: _get_user_camera_id called with user_id: {user_id}")
        conn = self.get_connection()
        if not conn:
            print("🔍 DEBUG: No database connection available")
            return None
        
        try:
            with conn.cursor() as cursor:
                # First, check what cameras exist for this user
                cursor.execute(
                    "SELECT camera_id, camera_name, user_id, status FROM cameras WHERE user_id = %s",
                    (user_id,)
                )
                all_cameras = cursor.fetchall()
                print(f"🔍 DEBUG: All cameras for user {user_id}: {all_cameras}")
                
                # Get first active camera
                cursor.execute(
                    "SELECT camera_id FROM cameras WHERE user_id = %s AND status = 'active' LIMIT 1",
                    (user_id,)
                )
                result = cursor.fetchone()
                camera_id = str(result['camera_id']) if result else None  # Use 'camera_id' not 'id'
                print(f"🔍 DEBUG: Found user camera_id: {camera_id}")
                return camera_id
        except Exception as e:
            print(f"🔍 DEBUG: Error getting user camera: {e}")
            return None
        finally:
            self.return_connection(conn)
    
    def _get_camera_name(self, camera_id: str) -> Optional[str]:
        """Get camera name from camera_id"""
        if not camera_id:
            return None
            
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT camera_name FROM cameras WHERE camera_id = %s",
                    (camera_id,)
                )
                result = cursor.fetchone()
                return result['camera_name'] if result else None
        except Exception as e:
            logger.warning(f"Error getting camera name: {e}")
            return None
        finally:
            self.return_connection(conn)
    
    def _get_any_camera_id(self) -> Optional[str]:
        """Get any available camera_id as fallback"""
        print(f"🔍 DEBUG: _get_any_camera_id called")
        conn = self.get_connection()
        if not conn:
            print("🔍 DEBUG: No database connection for _get_any_camera_id")
            return None

        try:
            with conn.cursor() as cursor:
                # Check what cameras exist at all
                cursor.execute("SELECT camera_id, camera_name, status FROM cameras LIMIT 5")
                all_cameras = cursor.fetchall()
                print(f"🔍 DEBUG: Available cameras: {all_cameras}")
                
                cursor.execute("SELECT camera_id FROM cameras WHERE status = 'active' LIMIT 1")
                result = cursor.fetchone()
                camera_id = str(result['camera_id']) if result else None  # Use 'camera_id' not 'id'
                print(f"🔍 DEBUG: Found any camera_id: {camera_id}")
                return camera_id
        except Exception as e:
            print(f"🔍 DEBUG: Error getting any camera: {e}")
            return None
        finally:
            self.return_connection(conn)
    
    def _create_minimal_snapshot(self, camera_id: str, user_id: str) -> Optional[str]:
        """Create minimal snapshot with just required fields"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            snapshot_id = str(uuid.uuid4())
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO snapshots (snapshot_id, camera_id, user_id) 
                    VALUES (%s, %s, %s) 
                    RETURNING snapshot_id
                """, (snapshot_id, camera_id, user_id))
                
                result = cursor.fetchone()
                conn.commit()
                return str(result['snapshot_id']) if result else None
        except Exception as e:
            logger.error(f"Error creating minimal snapshot: {e}")
            conn.rollback()
            return None
        finally:
            self.return_connection(conn)

    def _create_default_snapshot(self, camera_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
        """Create a default snapshot record with validated IDs"""
        conn = self.get_connection()
        if not conn:
            return None
        
        # Validate required IDs - if any is None/empty, skip snapshot creation
        if not all([camera_id, user_id]):
            logger.warning("⚠️ Missing required IDs for snapshot creation, skipping...")
            return None
        
        try:
            snapshot_id = str(uuid.uuid4())
            
            with conn.cursor() as cursor:
                insert_sql = """
                INSERT INTO snapshots (
                    snapshot_id, camera_id, user_id,
                    metadata, capture_type, captured_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s
                ) RETURNING snapshot_id
                """
                
                cursor.execute(insert_sql, (
                    snapshot_id,
                    camera_id,
                    user_id,
                    json.dumps({'type': 'default_snapshot', 'created_by': 'system'}),
                    'alert_triggered',
                    datetime.now(timezone.utc)
                ))
                
                result = cursor.fetchone()
                conn.commit()
                
                if result:
                    return result['snapshot_id'] if isinstance(result, dict) else result[0]
                    
        except Exception as e:
            logger.error(f"Error creating default snapshot: {e}")
            import traceback
            traceback.print_exc()
            if conn:
                conn.rollback()
        finally:
            self.return_connection(conn)
        
        return None
    
    def _determine_event_status(self, confidence: float, event_type: str) -> str:
        """
        Determine event status based on confidence and event type
        Aligned with healthcare_event_publisher SEVERITY_THRESHOLDS
        
        Args:
            confidence: Detection confidence (0.0 - 1.0)
            event_type: Type of event ('fall', 'abnormal_behavior', etc.)
            
        Returns:
            Status: 'normal', 'warning', or 'danger'
        """
        if event_type == 'fall':
            if confidence >= 0.60:        # high threshold for falls
                return 'danger'     
            elif confidence >= 0.40:      # medium threshold for falls
                return 'warning'    
            else:
                return 'normal'     # low confidence = normal monitoring
                
        elif event_type in ['abnormal_behavior', 'seizure']:
            if confidence >= 0.50:        # high threshold for seizures
                return 'danger'     
            elif confidence >= 0.30:      # medium threshold for seizures  
                return 'warning'    
            else:
                return 'normal'     # low confidence = normal monitoring
                
        else:
            # Unknown event type - use conservative thresholds
            if confidence >= 0.60:
                return 'danger'    
            elif confidence >= 0.40:
                return 'warning'
            else:
                return 'normal'
    
    def _calculate_reliability_score(self, confidence: float, event_type: str, 
                                     bounding_boxes: list = None, context: dict = None) -> float:
        """
        Calculate reliability score (độ nguy hiểm) based on multiple factors
        
        Công thức tính độ nguy hiểm:
        - Base score từ confidence (0-100)
        - Event type multiplier (fall/seizure nguy hiểm hơn)
        - Detection quality (số lượng bounding boxes, kích thước)
        - Context factors (location, time, history)
        
        Args:
            confidence: AI confidence score (0.0 - 1.0)
            event_type: Type of event detected
            bounding_boxes: List of detected objects
            context: Additional context data
            
        Returns:
            Reliability score (0.0 - 1.0): 1.0 = cực kỳ nguy hiểm, 0.0 = không nguy hiểm
        """
        
        # 1. BASE SCORE từ confidence (40% trọng số)
        base_score = confidence * 0.4
        
        # 2. EVENT TYPE SEVERITY (30% trọng số)
        event_severity = {
            'fall': 0.30,              # Té ngã: rất nguy hiểm
            'abnormal_behavior': 0.28,  # Bất thường: rất nguy hiểm
            'seizure': 0.28,           # Bất thường: rất nguy hiểm
            'manual_emergency': 0.30,   # Khẩn cấp thủ công: rất nguy hiểm
            'sleep': 0.05,             # Ngủ: ít nguy hiểm
            'normal_activity': 0.02    # Hoạt động bình thường: không nguy hiểm
        }
        severity_score = event_severity.get(event_type, 0.15)  # Default: mức trung bình
        
        # 3. DETECTION QUALITY (15% trọng số)
        quality_score = 0.0
        if bounding_boxes and len(bounding_boxes) > 0:
            # Có detection objects
            quality_score = 0.10
            
            # Bonus nếu có nhiều detections (người té có thể detect nhiều pose)
            if len(bounding_boxes) >= 2:
                quality_score += 0.03
            
            # Bonus nếu có keypoints (pose data)
            if any('keypoints' in bbox for bbox in bounding_boxes):
                quality_score += 0.02
        
        # 4. CONTEXT FACTORS (15% trọng số)
        context_score = 0.0
        if context:
            # Alert level từ detection
            alert_level = context.get('alert_level', '')
            if alert_level == 'critical':
                context_score = 0.15
            elif alert_level == 'high':
                context_score = 0.12
            elif alert_level == 'warning':
                context_score = 0.08
            
            # Consecutive detections (liên tục phát hiện = nguy hiểm hơn)
            if context.get('consecutive_detections', 0) >= 3:
                context_score += 0.03
        
        # TỔNG ĐIỂM
        total_score = base_score + severity_score + quality_score + context_score
        
        # Clamp giá trị trong khoảng [0.0, 1.0]
        reliability_score = min(max(total_score, 0.0), 1.0)
        
        # Log để debug
        logger.debug(f"🎯 Reliability Score Calculation:")
        logger.debug(f"   Base (confidence {confidence:.2%}): {base_score:.3f}")
        logger.debug(f"   Severity ({event_type}): {severity_score:.3f}")
        logger.debug(f"   Quality (boxes={len(bounding_boxes) if bounding_boxes else 0}): {quality_score:.3f}")
        logger.debug(f"   Context: {context_score:.3f}")
        logger.debug(f"   📊 TOTAL RELIABILITY: {reliability_score:.3f} ({reliability_score*100:.1f}%)")
        
        return reliability_score
    
    def update_event_snapshot(self, event_id: str, snapshot_id: str) -> bool:
        """
        Update event with snapshot_id after snapshot is created separately
        
        Args:
            event_id: Event UUID to update
            snapshot_id: Snapshot UUID to link
            
        Returns:
            bool: True if successful
        """
        if not self.is_connected:
            logger.error("PostgreSQL not connected")
            return False
        
        conn = self.get_connection()
        if not conn:
            logger.error("Could not get database connection")
            return False
        
        try:
            with conn.cursor() as cursor:
                update_sql = "UPDATE event_detections SET snapshot_id = %s WHERE event_id = %s"
                cursor.execute(update_sql, (snapshot_id, event_id))
                conn.commit()
                logger.info(f"✅ Updated event {event_id} with snapshot {snapshot_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to update event snapshot: {e}")
            conn.rollback()
            return False
        finally:
            self.return_connection(conn)
    
    def _generate_event_description(self, event_type: str, confidence: float, image_path: str, fallback_description: str, camera_name: str = None) -> str:
        """
        Generate intelligent action message for event_description field
        This should contain the FULL intelligent action with Vietnamese caption
        
        Args:
            event_type: Type of event (fall, abnormal_behavior, etc.)
            confidence: Detection confidence
            image_path: Path to event image/snapshot
            fallback_description: Original description as fallback
            camera_name: Optional camera name for location context
            
        Returns:
            Full intelligent action message (like: "🆘 KHẨN CẤP - CO GIẬT: Two young men are đứng trong phòng...")
        """
        try:
            # Debug logging for test description detection
            print(f" _generate_event_description called:")
            print(f"   event_type: {event_type}")
            print(f"   confidence: {confidence}")
            print(f"   fallback_description: '{fallback_description}'")
            
            # For test events, use the test description directly to create intelligent action
            if fallback_description and ('Một người' in fallback_description or 'Hai người' in fallback_description or 'Một em bé' in fallback_description or 'Một phụ nữ' in fallback_description):
                print(f"🧪 DETECTED TEST DESCRIPTION - Using for intelligent action: {fallback_description}")
                
                # Create intelligent action using test description
                if event_type in ['abnormal_behavior', 'seizure']:
                    if confidence >= 0.50:
                        result = f"🆘 KHẨN CẤP - CO GIẬT: {fallback_description} - CẦN ĐIỀU TRỊ Y TẾ NGAY!"
                    elif confidence >= 0.30:
                        result = f"⚠️ CẢNH BÁO CO GIẬT: {fallback_description} - Cần theo dõi chặt chẽ"
                    else:
                        result = f"📊 QUAN SÁT: {fallback_description} - Tiếp tục theo dõi"
                elif event_type == 'fall':
                    if confidence >= 0.60:
                        result = f"🚨 KHẨN CẤP - TÉ NGÃ: {fallback_description} - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC!"
                    elif confidence >= 0.40:
                        result = f"⚠️ CẢNH BÁO TÉ NGÃ: {fallback_description} - Cần theo dõi"
                    else:
                        result = f"📊 THEO DÕI: {fallback_description} - Quan sát"
                
                print(f"🎯 RETURNING TEST-BASED ACTION: {result}")
                return result
            
            print(f"⚠️ FALLBACK_DESCRIPTION does not match test patterns, using BLIP captioning...")
            
            # Try to generate intelligent action with Vietnamese caption
            # If image_path not provided, try to find latest alert image
            image_file_to_use = image_path
            if not image_file_to_use or not os.path.exists(image_file_to_use):
                logger.warning(f"⚠️ No valid image_path provided (got: {image_path})")
                logger.warning(f"⚠️ SKIPPING caption generation - will use fallback description")
                # 🔥 FIX: Do NOT use old images from alerts folder!
                # This was causing captions like "a camera is shown" or "man standing"
                # when the actual event had different content.
                # If no image_path, return fallback description instead.
                return fallback_description if fallback_description else f"Fall detected with {confidence:.1%} confidence"
            
            if image_file_to_use and os.path.exists(image_file_to_use):
                logger.info(f"🔍 Attempting to generate Vietnamese caption for image: {image_file_to_use}")
                # Try to use BLIP + Translation pipeline for full intelligent action
                try:
                    if self.vietnamese_caption is not None:
                        logger.info("✅ Vietnamese caption service is available, generating caption...")
                        # Generate Vietnamese caption from image with event_type for accurate medical context
                        vietnamese_result = self.vietnamese_caption.generate_professional_caption(
                            image_file_to_use,
                            event_type=event_type,  # Pass event_type to avoid filename confusion
                            camera_name=camera_name,  # Pass camera name for location context
                            confidence=confidence  # Pass confidence to smart caption replacement
                        )
                        vietnamese_caption = vietnamese_result[0] if isinstance(vietnamese_result, tuple) else vietnamese_result
                        
                        logger.info(f"📝 Generated Vietnamese caption: {vietnamese_caption}")
                        
                        if vietnamese_caption and len(vietnamese_caption.strip()) > 0:
                            # Create full intelligent action message like in main.py
                            if event_type in ['abnormal_behavior', 'seizure']:
                                if confidence >= 0.50:
                                    result = f"🆘 KHẨN CẤP - CO GIẬT: {vietnamese_caption} - CẦN ĐIỀU TRỊ Y TẾ NGAY!"
                                    logger.info(f"🚨 Generated seizure action: {result}")
                                    return result
                                elif confidence >= 0.30:
                                    result = f"⚠️ CẢNH BÁO CO GIẬT: {vietnamese_caption} - Cần theo dõi chặt chẽ"
                                    logger.info(f"⚠️ Generated seizure warning action: {result}")
                                    return result
                                else:
                                    result = f"📊 QUAN SÁT: {vietnamese_caption} - Tiếp tục theo dõi"
                                    logger.info(f"📊 Generated observation action: {result}")
                                    return result
                            elif event_type == 'fall':
                                if confidence >= 0.60:
                                    result = f"🚨 KHẨN CẤP - TÉ NGÃ: {vietnamese_caption} - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC!"
                                    logger.info(f"🚨 Generated fall emergency action: {result}")
                                    return result
                                elif confidence >= 0.40:
                                    result = f"⚠️ CẢNH BÁO TÉ NGÃ: {vietnamese_caption} - Cần theo dõi"
                                    logger.info(f"⚠️ Generated fall warning action: {result}")
                                    return result
                                else:
                                    result = f"📊 THEO DỔI: {vietnamese_caption} - Quan sát"
                                    logger.info(f"📊 Generated fall observation action: {result}")
                                    return result
                    else:
                        logger.warning("⚠️ Vietnamese caption service is not available")
                except Exception as e:
                    logger.warning(f"Failed to generate intelligent action: {e}")
            else:
                logger.warning(f"⚠️ No valid image file found for caption generation")
            
            # Fallback to simple action messages if Vietnamese caption fails
            logger.info("📋 Using fallback action messages")
            if event_type == 'fall':
                if confidence >= 0.60:
                    return f"🚨 KHẨN CẤP - TÉ NGÃ: Phát hiện té ngã nghiêm trọng - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC!"
                elif confidence >= 0.40:
                    return f"⚠️ CẢNH BÁO TÉ NGÃ: Phát hiện té ngã - Cần kiểm tra"
                else:
                    return f"📊 THEO DÕI: Nghi ngờ té ngã - Quan sát"
                    
            elif event_type in ['abnormal_behavior', 'seizure']:
                if confidence >= 0.50:
                    return f"🆘 KHẨN CẤP - CO GIẬT: Phát hiện co giật nghiêm trọng - CẦN ĐIỀU TRỊ Y TẾ NGAY!"
                elif confidence >= 0.30:
                    return f"⚠️ CẢNH BÁO CO GIẬT: Phát hiện co giật - Cần theo dõi chặt chẽ"
                else:
                    return f"📊 QUAN SÁT: Nghi ngờ co giật - Tiếp tục theo dõi"
                    
            else:
                # Unknown event type
                return f"🔍 PHÁT HIỆN: Sự kiện {event_type} - Cần đánh giá"
                
        except Exception as e:
            logger.error(f"❌ Error generating intelligent action: {e}")
            # Final fallback
            return fallback_description or f"Phát hiện sự kiện {event_type} (tin cậy: {confidence:.1%})"
    
    def generate_vietnamese_caption(self, image_path: str, event_type: str, confidence: float) -> str:
        """
        Generate Vietnamese caption for alert messages using BLIP model
        
        Args:
            image_path: Path to the image for captioning
            event_type: Type of event (fall, seizure, etc.)
            confidence: Detection confidence
            
        Returns:
            Vietnamese caption describing what's happening in the image
        """
        try:
            if self.vietnamese_caption is None:
                # Fallback: simple Vietnamese description
                if event_type == 'fall':
                    return f"Phát hiện té ngã (tin cậy: {confidence:.0%})"
                elif event_type in ['abnormal_behavior', 'seizure']:
                    return f"Phát hiện co giật (tin cậy: {confidence:.0%})"
                else:
                    return f"Phát hiện sự kiện {event_type} (tin cậy: {confidence:.0%})"
            
            # Use BLIP model for Vietnamese captioning
            vietnamese_result = self.vietnamese_caption.generate_professional_caption(image_path)
            vietnamese_description = vietnamese_result[0] if isinstance(vietnamese_result, tuple) else vietnamese_result
            
            if vietnamese_description and len(vietnamese_description.strip()) > 0:
                logger.info(f"✅ Generated Vietnamese caption: {vietnamese_description[:50]}...")
                return vietnamese_description
            else:
                # Fallback if BLIP fails
                logger.warning("BLIP returned empty caption, using fallback")
                if event_type == 'fall':
                    return f"Phát hiện té ngã (tin cậy: {confidence:.0%})"
                elif event_type in ['abnormal_behavior', 'seizure']:
                    return f"Phát hiện hành vi bất thường (tin cậy: {confidence:.0%})"
                else:
                    return f"Phát hiện sự kiện {event_type} (tin cậy: {confidence:.0%})"
                    
        except Exception as e:
            logger.error(f"❌ Error generating Vietnamese caption: {e}")
            # Fallback description
            if event_type == 'fall':
                return f"Phát hiện té ngã (tin cậy: {confidence:.0%})"
            elif event_type in ['abnormal_behavior', 'seizure']:
                return f"Phát hiện co giật (tin cậy: {confidence:.0%})"
            else:
                return f"Phát hiện sự kiện {event_type} (tin cậy: {confidence:.0%})"
    
    def publish_event_detection(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert event detection into database"""
        
        # Add unique detection key for duplicate prevention
        import time
        detection_key = f"{event_data.get('event_type')}_{event_data.get('confidence', 0):.3f}_{int(time.time() * 1000)}"
        logger.info(f"🔍 Publishing event detection: {detection_key}")
        
        if not self.is_connected:
            logger.error("PostgreSQL not connected")
            return None
        
        # DON'T get connection here - helper functions manage their own connections
        
        try:
            # Get user's real camera_id from database
            user_id = event_data.get('user_id')
            camera_id = event_data.get('camera_id')
            
            # If no user_id, get from environment
            if not user_id:
                user_id = os.getenv('DEFAULT_USER_ID')
                logger.info(f"🔧 Using DEFAULT_USER_ID from env: {user_id}")
            
            # If no camera_id provided, get user's first camera (uses own connection)
            if not camera_id and user_id:
                camera_id = self._get_user_camera_id(user_id)
                print(f"🔧 Got user camera_id: {camera_id}")
            
            # If still no camera_id, get any available camera
            if not camera_id:
                camera_id = self._get_any_camera_id()
                print(f"🔧 Got fallback camera_id: {camera_id}")
            
            print(f"🔧 Final IDs - user_id: {user_id}, camera_id: {camera_id}")
            
            # Upload image to MinIO if frame is provided
            snapshot_id = None
            image_id = None
            cloud_url = None
            
            if self.snapshot_service and event_data.get('frame') is not None:
                try:
                    import numpy as np
                    frame = event_data.get('frame')
                    
                    # Ensure frame is valid numpy array
                    if isinstance(frame, np.ndarray) and frame.size > 0:
                        logger.info(f"📸 Uploading {event_data.get('event_type')} image to MinIO...")
                        
                        snapshot_id, image_id = self.snapshot_service.create_detection_snapshot(
                            camera_id=camera_id,
                            user_id=user_id,
                            event_type=event_data.get('event_type', 'unknown'),
                            confidence=event_data.get('confidence', 0.0),
                            frame=frame,
                            metadata={
                                'detection_data': event_data.get('detection_data', {}),
                                'bounding_boxes': event_data.get('bounding_boxes', [])
                            }
                        )
                        
                        logger.info(f"✅ MinIO upload successful! snapshot_id: {snapshot_id}, image_id: {image_id}")
                    else:
                        logger.warning("⚠️ Invalid frame data - skipping MinIO upload")
                        
                except Exception as upload_error:
                    logger.error(f"❌ MinIO upload failed: {upload_error}")
                    import traceback
                    traceback.print_exc()
            
            # Use snapshot_id from parameter if provided (means snapshot created separately)
            if not snapshot_id:
                snapshot_id = event_data.get('snapshot_id')
            
            # Only create default snapshot if still no snapshot_id (legacy compatibility)
            if not snapshot_id:
                snapshot_id = self._create_default_snapshot(
                    camera_id=camera_id,
                    user_id=user_id
                )
            
            # If snapshot creation failed, try to create one with minimal data
            if not snapshot_id and camera_id and user_id:
                snapshot_id = self._create_minimal_snapshot(camera_id, user_id)
                
            # If still failed, create dummy snapshot
            if not snapshot_id:
                snapshot_id = str(uuid.uuid4())
                logger.warning("Using dummy snapshot_id due to snapshot creation failure")
            
            # Get camera name for location context
            camera_name = None
            if camera_id:
                camera_name = self._get_camera_name(camera_id)
                logger.info(f"📍 Camera location: {camera_name}")
            
            # Generate Vietnamese description for the event
            print(f"🔥 DEBUG BEFORE _generate_event_description:")
            print(f"   event_data description: '{event_data.get('description', '')}'")
            
            vietnamese_description = self._generate_event_description(
                event_data.get('event_type', ''),
                event_data.get('confidence', 0.0),
                event_data.get('image_path', ''),
                event_data.get('description', ''),
                camera_name=camera_name
            )
            
            print(f"🔥 DEBUG AFTER _generate_event_description:")
            print(f"   vietnamese_description: '{vietnamese_description}'")
            
            # Validate event description - don't save if NULL or empty
            if not vietnamese_description or vietnamese_description.strip() == '' or vietnamese_description.lower() == 'null':
                logger.warning(f"❌ Skipping event detection save - empty event_description for {event_data.get('event_type', 'unknown')}")
                return None
                
            # Check for recent duplicate events (same type, user, camera within 5 seconds)
            dup_conn = None
            try:
                dup_conn = self.get_connection()
                if dup_conn:
                    with dup_conn.cursor() as cursor:
                        duplicate_check_sql = """
                        SELECT event_id FROM event_detections 
                        WHERE event_type = %s AND user_id = %s AND camera_id = %s 
                        AND detected_at > NOW() - INTERVAL '5 seconds'
                        ORDER BY detected_at DESC LIMIT 1
                        """
                        cursor.execute(duplicate_check_sql, (
                            event_data.get('event_type'),
                            user_id,
                            camera_id
                        ))
                        recent_event = cursor.fetchone()
                        
                        if recent_event and recent_event[0]:  # FIX: Check if result exists AND has event_id
                            logger.warning(f"❌ Skipping duplicate event detection - similar {event_data.get('event_type')} within 5 seconds")
                            self.return_connection(dup_conn)
                            return {'event_id': recent_event[0], 'duplicate_skipped': True}
                    
                    # Return connection if no duplicate found
                    self.return_connection(dup_conn)
                        
            except Exception as dup_error:
                logger.warning(f"Duplicate check failed: {dup_error}")
                if dup_conn:
                    self.return_connection(dup_conn)
            
            # Validate final IDs (user_id and camera_id already processed above)
            
            # Calculate reliability score (độ nguy hiểm)
            reliability_score = self._calculate_reliability_score(
                confidence=event_data.get('confidence', 0.0),
                event_type=event_data.get('event_type', ''),
                bounding_boxes=event_data.get('bounding_boxes', []),
                context=event_data.get('context', {})
            )
            
            # FIX: Remove numpy arrays from context before JSON serialization
            context_data = event_data.get('context', {}).copy()
            if 'frame' in context_data:
                del context_data['frame']  # Remove frame (numpy ndarray) to avoid JSON serialization error
            if 'original_frame' in context_data:
                del context_data['original_frame']
            
            # Prepare record with validated values
            record = {
                'event_id': str(uuid.uuid4()),
                'user_id': user_id,
                'camera_id': camera_id,
                'snapshot_id': snapshot_id,
                'event_type': event_data.get('event_type'),
                'event_description': vietnamese_description,  # Use Vietnamese description
                'detection_data': json.dumps(event_data.get('detection_data', {})),
                'ai_analysis_result': json.dumps(event_data.get('ai_analysis', {})),
                'confidence_score': str(event_data.get('confidence', 0.0)),  # FIX: Convert to string for DB
                'bounding_boxes': json.dumps(event_data.get('bounding_boxes', [])),
                'status': self._determine_event_status(
                    event_data.get('confidence', 0.0),
                    event_data.get('event_type', '')
                ),
                'context_data': json.dumps(context_data),  # FIX: Use cleaned context without frames,
                'detected_at': datetime.now(timezone.utc),
                'created_at': datetime.now(timezone.utc),
                # Required fields with NOT NULL constraint
                'lifecycle_state': 'NOTIFIED',  # Initial state when event is created
                'confirmation_state': 'DETECTED',  # Fixed: Use valid enum value (was PENDING_CONFIRMATION)
                'verification_status': 'PENDING',  # Waiting for verification
                'escalation_count': 0,  # No escalations yet
                'is_canceled': False,  # Not canceled
                'notification_attempts': 0,  # Will be incremented when notification sent
                'reliability_score': str(reliability_score)  # FIX: Convert to string for DB
            }
            
            # Get NEW connection for INSERT (previous conn may be closed by helper functions)
            insert_conn = self.get_connection()
            if not insert_conn:
                logger.error("Could not get database connection for INSERT")
                return None
            
            try:
                with insert_conn.cursor() as cursor:
                    insert_sql = """
                    INSERT INTO event_detections (
                        event_id, user_id, camera_id, snapshot_id,
                        event_type, event_description, detection_data, ai_analysis_result,
                        confidence_score, bounding_boxes, status, context_data,
                        detected_at, created_at,
                        lifecycle_state, confirmation_state, verification_status,
                        escalation_count, is_canceled, notification_attempts,
                        reliability_score
                    ) VALUES (
                        %(event_id)s, %(user_id)s, %(camera_id)s, %(snapshot_id)s,
                        %(event_type)s, %(event_description)s, %(detection_data)s, %(ai_analysis_result)s,
                        %(confidence_score)s, %(bounding_boxes)s, %(status)s, %(context_data)s,
                        %(detected_at)s, %(created_at)s,
                        %(lifecycle_state)s, %(confirmation_state)s, %(verification_status)s,
                        %(escalation_count)s, %(is_canceled)s, %(notification_attempts)s,
                        %(reliability_score)s
                    ) RETURNING *
                    """
                    
                    cursor.execute(insert_sql, record)
                    result = cursor.fetchone()
                    insert_conn.commit()
                    
                    if result:
                        logger.info(f"✅ Event detection published: {record['event_type']} with confidence {record['confidence_score']}")
                        print(f"💾 ✅ DATABASE SAVE SUCCESS!")
                        print(f"   Event ID: {record['event_id']}")
                        print(f"   Event Type: {record['event_type']}")
                        print(f"   Status: {record['status']}")
                        print(f"   Confidence: {record['confidence_score']}")
                        print(f"   🎯 Reliability (Độ nguy hiểm): {record['reliability_score']}")
                        print(f"   Description: {record['event_description'][:100]}...")
                        
                        self.return_connection(insert_conn)
                        # DON'T return conn - it was already returned by helper functions
                        return dict(result)
                    else:
                        logger.error("❌ Failed to publish event detection")
                        print(f"❌ DATABASE SAVE FAILED - No result returned")
                        self.return_connection(insert_conn)
                        # DON'T return conn - it was already returned by helper functions
                        return None
                        
            except Exception as insert_error:
                logger.error(f"Error during INSERT: {insert_error}")
                import traceback
                traceback.print_exc()
                insert_conn.rollback()
                self.return_connection(insert_conn)
                # DON'T return conn - it was already returned by helper functions
                return None
                    
        except Exception as e:
            logger.error(f"Error publishing event detection: {e}")
            import traceback
            traceback.print_exc()
            # DON'T rollback or return conn - it was already handled by helper functions
            return None
    
    def publish_alert(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert alert into event_detections table instead of alerts"""
        if not self.is_connected:
            logger.error("PostgreSQL not connected")
            return None
        
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            # Get real user_id and camera_id
            user_id = alert_data.get('user_id')
            camera_id = alert_data.get('camera_id')
            
            # If no user_id, get from environment
            if not user_id:
                user_id = os.getenv('DEFAULT_USER_ID')
            
            # If no camera_id, get user's camera
            if not camera_id and user_id:
                camera_id = self._get_user_camera_id(user_id)
            
            # If still no camera_id, get any camera
            if not camera_id:
                camera_id = self._get_any_camera_id()
            
            # Create snapshot_id
            if camera_id and user_id:
                snapshot_id = self._create_minimal_snapshot(camera_id, user_id)
            else:
                snapshot_id = None
            if not snapshot_id:
                snapshot_id = str(uuid.uuid4())
                logger.warning("Using dummy snapshot_id for alert")
            
            # Convert alert data to event_detection format
            record = {
                'event_id': str(uuid.uuid4()),
                'user_id': user_id,
                'camera_id': camera_id,
                'snapshot_id': snapshot_id,
                'event_type': alert_data.get('alert_type', 'alert'),
                'confidence_score': alert_data.get('confidence', 0.8),
                'detection_data': json.dumps({
                    'alert_type': alert_data.get('alert_type'),
                    'severity': alert_data.get('severity', 'medium'),
                    'alert_message': alert_data.get('message'),
                    'alert_data': alert_data.get('alert_data', {})
                }),
                'created_at': datetime.now(timezone.utc),
                'detected_at': datetime.now(timezone.utc),
                # Required fields with NOT NULL constraint
                'lifecycle_state': 'NOTIFIED',
                'confirmation_state': 'DETECTED',  # Fixed: Use valid enum value
                'verification_status': 'PENDING',
                'escalation_count': 0,
                'is_canceled': False,
                'notification_attempts': 0,
                'event_description': alert_data.get('message', 'Alert notification'),
                'status': 'danger' if alert_data.get('severity') == 'critical' else 'warning'
            }
            
            with conn.cursor() as cursor:
                insert_sql = """
                INSERT INTO event_detections (
                    event_id, user_id, camera_id, snapshot_id, event_type, confidence_score,
                    detection_data, created_at, detected_at,
                    lifecycle_state, confirmation_state, verification_status,
                    escalation_count, is_canceled, notification_attempts,
                    event_description, status
                ) VALUES (
                    %(event_id)s, %(user_id)s, %(camera_id)s, %(snapshot_id)s, %(event_type)s, %(confidence_score)s,
                    %(detection_data)s, %(created_at)s, %(detected_at)s,
                    %(lifecycle_state)s, %(confirmation_state)s, %(verification_status)s,
                    %(escalation_count)s, %(is_canceled)s, %(notification_attempts)s,
                    %(event_description)s, %(status)s
                ) RETURNING *
                """
                
                cursor.execute(insert_sql, record)
                result = cursor.fetchone()
                conn.commit()
                
                if result:
                    logger.info(f"✅ Alert published to event_detections: {record['event_type']} - {alert_data.get('severity', 'medium')}")
                    return dict(result)
                    
        except Exception as e:
            logger.error(f"Error publishing alert to event_detections: {e}")
            conn.rollback()
            return None
        finally:
            self.return_connection(conn)
    
    def publish_snapshot(self, snapshot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert snapshot into database"""
        if not self.is_connected:
            logger.error("PostgreSQL not connected")
            return None
        
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            record = {
                'snapshot_id': str(uuid.uuid4()),
                'camera_id': snapshot_data.get('camera_id', str(uuid.uuid4())),
                'room_id': snapshot_data.get('room_id', str(uuid.uuid4())),
                'user_id': snapshot_data.get('user_id'),
                'image_path': snapshot_data.get('image_path'),
                'cloud_url': snapshot_data.get('cloud_url'),
                'metadata': json.dumps(snapshot_data.get('metadata', {})),
                'capture_type': snapshot_data.get('capture_type', 'alert'),
                'captured_at': datetime.now(timezone.utc),
                'is_processed': False
            }
            
            with conn.cursor() as cursor:
                insert_sql = """
                INSERT INTO snapshots (
                    snapshot_id, camera_id, room_id, user_id, image_path,
                    cloud_url, metadata, capture_type, captured_at, is_processed
                ) VALUES (
                    %(snapshot_id)s, %(camera_id)s, %(room_id)s, %(user_id)s, %(image_path)s,
                    %(cloud_url)s, %(metadata)s, %(capture_type)s, %(captured_at)s, %(is_processed)s
                ) RETURNING *
                """
                
                cursor.execute(insert_sql, record)
                result = cursor.fetchone()
                conn.commit()
                
                if result:
                    logger.info(f"✅ Snapshot published: {record['image_path']}")
                    return dict(result)
                    
        except Exception as e:
            logger.error(f"Error publishing snapshot: {e}")
            conn.rollback()
            return None
        finally:
            self.return_connection(conn)
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events from database"""
        if not self.is_connected:
            return []
        
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM event_detections ORDER BY created_at DESC LIMIT %s",
                    (limit,)
                )
                results = cursor.fetchall()
                return [dict(row) for row in results] if results else []
                
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def close(self):
        """Close all connections"""
        try:
            # Stop all polling threads
            threads_to_stop = list(self.polling_threads.keys())
            for subscription_key in threads_to_stop:
                del self.polling_threads[subscription_key]
            
            # Close connection pool
            if self.connection_pool:
                self.connection_pool.closeall()
            
            self.is_connected = False
            logger.info("PostgreSQL service closed")
            
        except Exception as e:
            logger.error(f"Error closing PostgreSQL service: {e}")

# Global service instance
postgresql_service = PostgreSQLHealthcareService()
