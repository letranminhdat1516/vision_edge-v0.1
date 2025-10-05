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
from service.config_loader import config_loader

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
        
        # Ensure default entities exist in database
        self._ensure_default_entities()
    
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
                    
                    # Create connection pool
                    self.connection_pool = SimpleConnectionPool(
                        minconn=1,
                        maxconn=10,
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
    
    def _create_default_snapshot(self, camera_id: Optional[str] = None, room_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
        """Create a default snapshot record with proper fallback values"""
        conn = self.get_connection()
        if not conn:
            return None
        
        # Get default IDs from config
        db_config = config_loader.get_database_config()
        fall_defaults = db_config.get('default_ids', {}).get('fall_detection', {})
        
        # Use provided IDs or fallback to config values, validate not None
        final_camera_id = camera_id or fall_defaults.get('camera_id')
        final_room_id = room_id or fall_defaults.get('room_id')
        final_user_id = user_id or fall_defaults.get('user_id')
        
        # Validate UUIDs - if any is None/empty, skip snapshot creation
        if not all([final_camera_id, final_room_id, final_user_id]):
            logger.warning("⚠️ Missing required IDs for snapshot creation, skipping...")
            return None
        
        try:
            snapshot_id = str(uuid.uuid4())
            
            with conn.cursor() as cursor:
                insert_sql = """
                INSERT INTO snapshots (
                    snapshot_id, camera_id, room_id, user_id,
                    image_path, metadata, capture_type, captured_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                ) RETURNING snapshot_id
                """
                
                cursor.execute(insert_sql, (
                    snapshot_id,
                    final_camera_id,
                    final_room_id, 
                    final_user_id,
                    f'default_{snapshot_id}.jpg',  # Default image path
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
    
    def _generate_event_description(self, event_type: str, confidence: float, image_path: str, fallback_description: str) -> str:
        """
        Generate intelligent action message for event_description field
        This should contain the FULL intelligent action with Vietnamese caption
        
        Args:
            event_type: Type of event (fall, abnormal_behavior, etc.)
            confidence: Detection confidence
            image_path: Path to event image/snapshot
            fallback_description: Original description as fallback
            
        Returns:
            Full intelligent action message (like: "🆘 KHẨN CẤP - CO GIẬT: Two young men are đứng trong phòng...")
        """
        try:
            # Debug logging for test description detection
            print(f"� _generate_event_description called:")
            print(f"   event_type: {event_type}")
            print(f"   confidence: {confidence}")
            print(f"   fallback_description: '{fallback_description}'")
            
            # For test events, use the test description directly to create intelligent action
            if fallback_description and ('Một người' in fallback_description or 'Hai người' in fallback_description or 'Một em bé' in fallback_description or 'Một phụ nữ' in fallback_description):
                print(f"🧪 DETECTED TEST DESCRIPTION - Using for intelligent action: {fallback_description}")
                
                # Create intelligent action using test description
                if event_type in ['abnormal_behavior', 'seizure']:
                    if confidence >= 0.50:
                        result = f"🆘 KHẨN CẤP - CO GIẬT: {fallback_description} - CẦN ĐIỀU TRỊ Y TẾ NGAY! (Tin cậy: {confidence:.0%})"
                    elif confidence >= 0.30:
                        result = f"⚠️ CẢNH BÁO BẤT THƯỜNG: {fallback_description} - Cần theo dõi chặt chẽ (Tin cậy: {confidence:.0%})"
                    else:
                        result = f"📊 QUAN SÁT: {fallback_description} - Tiếp tục theo dõi (Tin cậy: {confidence:.0%})"
                elif event_type == 'fall':
                    if confidence >= 0.60:
                        result = f"🚨 KHẨN CẤP - TÉ NGÃ: {fallback_description} - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC! (Tin cậy: {confidence:.0%})"
                    elif confidence >= 0.40:
                        result = f"⚠️ CẢNH BÁO TÉ NGÃ: {fallback_description} - Cần theo dõi (Tin cậy: {confidence:.0%})"
                    else:
                        result = f"📊 THEO DÕI: {fallback_description} - Quan sát (Tin cậy: {confidence:.0%})"
                
                print(f"🎯 RETURNING TEST-BASED ACTION: {result}")
                return result
            
            print(f"⚠️ FALLBACK_DESCRIPTION does not match test patterns, using BLIP captioning...")
            
            # Try to generate intelligent action with Vietnamese caption
            # If image_path not provided, try to find latest alert image
            image_file_to_use = image_path
            if not image_file_to_use or not os.path.exists(image_file_to_use):
                # Try to find latest alert image
                try:
                    import glob
                    from pathlib import Path
                    
                    # Try multiple alert directories
                    alert_dirs = [
                        "examples/data/saved_frames/alerts",
                        "data/saved_frames/alerts",
                        os.path.join(os.getcwd(), "examples/data/saved_frames/alerts"),
                        os.path.join(os.getcwd(), "data/saved_frames/alerts")
                    ]
                    
                    for alerts_dir in alert_dirs:
                        alerts_path = Path(alerts_dir)
                        if alerts_path.exists():
                            image_files = list(alerts_path.glob("*.jpg"))
                            if image_files:
                                # Get most recent image
                                image_file_to_use = str(max(image_files, key=lambda p: p.stat().st_ctime))
                                logger.info(f"🔍 Found latest alert image: {image_file_to_use}")
                                break
                except Exception as e:
                    logger.warning(f"⚠️ Could not find alert image: {e}")
            
            if image_file_to_use and os.path.exists(image_file_to_use):
                logger.info(f"🔍 Attempting to generate Vietnamese caption for image: {image_file_to_use}")
                # Try to use BLIP + Translation pipeline for full intelligent action
                try:
                    if self.vietnamese_caption is not None:
                        logger.info("✅ Vietnamese caption service is available, generating caption...")
                        # Generate Vietnamese caption from image
                        vietnamese_result = self.vietnamese_caption.generate_professional_caption(image_file_to_use)
                        vietnamese_caption = vietnamese_result[0] if isinstance(vietnamese_result, tuple) else vietnamese_result
                        
                        logger.info(f"📝 Generated Vietnamese caption: {vietnamese_caption}")
                        
                        if vietnamese_caption and len(vietnamese_caption.strip()) > 0:
                            # Create full intelligent action message like in main.py
                            if event_type in ['abnormal_behavior', 'seizure']:
                                if confidence >= 0.50:
                                    result = f"🆘 KHẨN CẤP - CO GIẬT: {vietnamese_caption} 🚨 Cảnh báo: Phát hiện co giật - Độ tin cậy: 0.0% - CẦN ĐIỀU TRỊ Y TẾ NGAY! (Tin cậy: {confidence:.0%})"
                                    logger.info(f"🚨 Generated seizure action: {result}")
                                    return result
                                elif confidence >= 0.30:
                                    result = f"⚠️ CẢNH BÁO BẤT THƯỜNG: {vietnamese_caption} ⚠️ Cảnh báo: Phát hiện hành vi bất thường - Độ tin cậy: {confidence:.1%} - Cần theo dõi chặt chẽ (Tin cậy: {confidence:.0%})"
                                    logger.info(f"⚠️ Generated abnormal action: {result}")
                                    return result
                                else:
                                    result = f"📊 QUAN SÁT: {vietnamese_caption} - Nghi ngờ hành vi bất thường - Độ tin cậy: {confidence:.1%} - Tiếp tục theo dõi (Tin cậy: {confidence:.0%})"
                                    logger.info(f"📊 Generated observation action: {result}")
                                    return result
                            elif event_type == 'fall':
                                if confidence >= 0.60:
                                    result = f"🚨 KHẨN CẤP - TÉ NGÃ: {vietnamese_caption} 🚨 Cảnh báo: Phát hiện té ngã - Độ tin cậy: 0.0% - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC! (Tin cậy: {confidence:.0%})"
                                    logger.info(f"🚨 Generated fall emergency action: {result}")
                                    return result
                                elif confidence >= 0.40:
                                    result = f"⚠️ CẢNH BÁO TÉ NGÃ: {vietnamese_caption} ⚠️ Cảnh báo: Phát hiện ngã đổ - Độ tin cậy: 0.0% - Cần theo dõi (Tin cậy: {confidence:.0%})"
                                    logger.info(f"⚠️ Generated fall warning action: {result}")
                                    return result
                                else:
                                    result = f"📊 THEO DÕI: {vietnamese_caption} - Nghi ngờ té ngã - Độ tin cậy: {confidence:.1%} - Quan sát (Tin cậy: {confidence:.0%})"
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
                    return f"🚨 KHẨN CẤP - TÉ NGÃ: Phát hiện té ngã nghiêm trọng - Độ tin cậy: {confidence:.1%} - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC!"
                elif confidence >= 0.40:
                    return f"⚠️ CẢNH BÁO TÉ NGÃ: Phát hiện té ngã - Độ tin cậy: {confidence:.1%} - Cần kiểm tra"
                else:
                    return f"📊 THEO DÕI: Nghi ngờ té ngã - Độ tin cậy: {confidence:.1%} - Quan sát"
                    
            elif event_type in ['abnormal_behavior', 'seizure']:
                if confidence >= 0.50:
                    return f"🆘 KHẨN CẤP - CO GIẬT: Phát hiện co giật nghiêm trọng - Độ tin cậy: {confidence:.1%} - CẦN ĐIỀU TRỊ Y TẾ NGAY!"
                elif confidence >= 0.30:
                    return f"⚠️ CẢNH BÁO BẤT THƯỜNG: Phát hiện hành vi bất thường - Độ tin cậy: {confidence:.1%} - Cần theo dõi chặt chẽ"
                else:
                    return f"📊 QUAN SÁT: Nghi ngờ hành vi bất thường - Độ tin cậy: {confidence:.1%} - Tiếp tục theo dõi"
                    
            else:
                # Unknown event type
                return f"🔍 PHÁT HIỆN: Sự kiện {event_type} - Độ tin cậy: {confidence:.1%} - Cần đánh giá"
                
        except Exception as e:
            logger.error(f"❌ Error generating intelligent action: {e}")
            # Final fallback
            return fallback_description or f"Phát hiện sự kiện {event_type} (độ tin cậy: {confidence:.1%})"
    
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
                    return f"Phát hiện té ngã với độ tin cậy {confidence:.1%}"
                elif event_type in ['abnormal_behavior', 'seizure']:
                    return f"Phát hiện co giật với độ tin cậy {confidence:.1%}"
                else:
                    return f"Phát hiện sự kiện {event_type} với độ tin cậy {confidence:.1%}"
            
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
                    return f"Phát hiện té ngã với độ tin cậy {confidence:.1%}"
                elif event_type in ['abnormal_behavior', 'seizure']:
                    return f"Phát hiện co giật với độ tin cậy {confidence:.1%}"
                else:
                    return f"Phát hiện sự kiện {event_type} với độ tin cậy {confidence:.1%}"
                    
        except Exception as e:
            logger.error(f"❌ Error generating Vietnamese caption: {e}")
            # Fallback description
            if event_type == 'fall':
                return f"Phát hiện té ngã với độ tin cậy {confidence:.1%}"
            elif event_type in ['abnormal_behavior', 'seizure']:
                return f"Phát hiện co giật với độ tin cậy {confidence:.1%}"
            else:
                return f"Phát hiện sự kiện {event_type} với độ tin cậy {confidence:.1%}"
    
    def publish_event_detection(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert event detection into database"""
        if not self.is_connected:
            logger.error("PostgreSQL not connected")
            return None
        
        conn = self.get_connection()
        if not conn:
            logger.error("Could not get database connection")
            return None
        
        try:
            # Get default IDs from config
            db_config = config_loader.get_database_config()
            fall_defaults = db_config.get('default_ids', {}).get('fall_detection', {})
            
            # Create snapshot first
            snapshot_id = event_data.get('snapshot_id') or self._create_default_snapshot(
                camera_id=event_data.get('camera_id') or fall_defaults.get('camera_id'),
                room_id=event_data.get('room_id') or fall_defaults.get('room_id'),
                user_id=event_data.get('user_id') or fall_defaults.get('user_id')
            )
            
            # If snapshot creation failed, create a dummy UUID to avoid NULL constraint
            if not snapshot_id:
                snapshot_id = str(uuid.uuid4())
                logger.warning("Using dummy snapshot_id due to snapshot creation failure")
            
            # Generate Vietnamese description for the event
            print(f"🔥 DEBUG BEFORE _generate_event_description:")
            print(f"   event_data description: '{event_data.get('description', '')}'")
            
            vietnamese_description = self._generate_event_description(
                event_data.get('event_type', ''),
                event_data.get('confidence', 0.0),
                event_data.get('image_path', ''),
                event_data.get('description', '')
            )
            
            print(f"🔥 DEBUG AFTER _generate_event_description:")
            print(f"   vietnamese_description: '{vietnamese_description}'")
            
            
            # Get default IDs from config for event detection
            event_type = event_data.get('event_type', '')
            if event_type == 'seizure':
                event_defaults = db_config.get('default_ids', {}).get('seizure_detection', {})
            else:
                event_defaults = db_config.get('default_ids', {}).get('fall_detection', {})
            
            # Get IDs with validation
            user_id = event_data.get('user_id') or event_defaults.get('user_id')
            camera_id = event_data.get('camera_id') or event_defaults.get('camera_id')
            room_id = event_data.get('room_id') or event_defaults.get('room_id')
            
            # Validate all required IDs exist
            if not all([user_id, camera_id, room_id]):
                logger.warning("⚠️ Missing required IDs for event detection, using dummy values...")
                user_id = user_id or str(uuid.uuid4())
                camera_id = camera_id or str(uuid.uuid4())
                room_id = room_id or str(uuid.uuid4())
            
            # Prepare record with validated values
            record = {
                'event_id': str(uuid.uuid4()),
                'user_id': user_id,
                'camera_id': camera_id,
                'room_id': room_id,
                'snapshot_id': snapshot_id,
                'event_type': event_data.get('event_type'),
                'event_description': vietnamese_description,  # Use Vietnamese description
                'detection_data': json.dumps(event_data.get('detection_data', {})),
                'ai_analysis_result': json.dumps(event_data.get('ai_analysis', {})),
                'confidence_score': float(event_data.get('confidence', 0.0)),
                'bounding_boxes': json.dumps(event_data.get('bounding_boxes', [])),
                'status': self._determine_event_status(
                    event_data.get('confidence', 0.0),
                    event_data.get('event_type', '')
                ),
                'context_data': json.dumps(event_data.get('context', {})),
                'detected_at': datetime.now(timezone.utc),
                'created_at': datetime.now(timezone.utc)
            }
            
            with conn.cursor() as cursor:
                insert_sql = """
                INSERT INTO event_detections (
                    event_id, user_id, camera_id, room_id, snapshot_id,
                    event_type, event_description, detection_data, ai_analysis_result,
                    confidence_score, bounding_boxes, status, context_data,
                    detected_at, created_at
                ) VALUES (
                    %(event_id)s, %(user_id)s, %(camera_id)s, %(room_id)s, %(snapshot_id)s,
                    %(event_type)s, %(event_description)s, %(detection_data)s, %(ai_analysis_result)s,
                    %(confidence_score)s, %(bounding_boxes)s, %(status)s, %(context_data)s,
                    %(detected_at)s, %(created_at)s
                ) RETURNING *
                """
                
                cursor.execute(insert_sql, record)
                result = cursor.fetchone()
                conn.commit()
                
                if result:
                    logger.info(f"✅ Event detection published: {record['event_type']} with confidence {record['confidence_score']}")
                    return dict(result)
                else:
                    logger.error("❌ Failed to publish event detection")
                    return None
                    
        except Exception as e:
            logger.error(f"Error publishing event detection: {e}")
            conn.rollback()
            return None
        finally:
            self.return_connection(conn)
    
    def publish_alert(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert alert into database"""
        if not self.is_connected:
            logger.error("PostgreSQL not connected")
            return None
        
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            record = {
                'alert_id': str(uuid.uuid4()),
                'event_id': alert_data.get('event_id'),
                'user_id': alert_data.get('user_id', str(uuid.uuid4())),
                'alert_type': alert_data.get('alert_type'),
                'severity': alert_data.get('severity', 'medium'),
                'alert_message': alert_data.get('message'),
                'alert_data': json.dumps(alert_data.get('alert_data', {})),
                'status': 'active',
                'created_at': datetime.now(timezone.utc)
            }
            
            with conn.cursor() as cursor:
                insert_sql = """
                INSERT INTO alerts (
                    alert_id, event_id, user_id, alert_type, severity,
                    alert_message, alert_data, status, created_at
                ) VALUES (
                    %(alert_id)s, %(event_id)s, %(user_id)s, %(alert_type)s, %(severity)s,
                    %(alert_message)s, %(alert_data)s, %(status)s, %(created_at)s
                ) RETURNING *
                """
                
                cursor.execute(insert_sql, record)
                result = cursor.fetchone()
                conn.commit()
                
                if result:
                    logger.info(f"✅ Alert published: {record['alert_type']} - {record['severity']}")
                    return dict(result)
                    
        except Exception as e:
            logger.error(f"Error publishing alert: {e}")
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
    
    def _ensure_default_entities(self):
        """Ensure default users, cameras, rooms exist in database"""
        try:
            db_config = config_loader.get_database_config()
            default_ids = db_config.get('default_ids', {})
            
            # Get default IDs for both fall and seizure detection
            fall_defaults = default_ids.get('fall_detection', {})
            seizure_defaults = default_ids.get('seizure_detection', {})
            
            all_defaults = [fall_defaults, seizure_defaults]
            
            # Create default entities if they don't exist
            for defaults in all_defaults:
                if defaults:
                    self._create_default_user(defaults.get('user_id'))
                    self._create_default_room(defaults.get('room_id'))
                    self._create_default_camera(defaults.get('camera_id'), defaults.get('room_id'))
                    
            logger.info("✅ Default entities verified/created")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not ensure default entities: {e}")
    
    def _create_default_user(self, user_id: str):
        """Create default user if not exists"""
        if not user_id:
            return
            
        conn = self.get_connection()
        if not conn:
            return
            
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
                if cursor.fetchone():
                    return  # User already exists
                
                # Create default user
                cursor.execute("""
                    INSERT INTO users (user_id, username, email, role, created_at) 
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                """, (
                    user_id,
                    'admin_demo',
                    'admin@vision-edge.com',
                    'admin',
                    datetime.now(timezone.utc)
                ))
                conn.commit()
                logger.info(f"✅ Created default user: {user_id}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not create default user {user_id}: {e}")
        finally:
            self.return_connection(conn)
    
    def _create_default_room(self, room_id: str):
        """Create default room if not exists"""
        if not room_id:
            return
            
        conn = self.get_connection()
        if not conn:
            return
            
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT room_id FROM rooms WHERE room_id = %s", (room_id,))
                if cursor.fetchone():
                    return  # Room already exists
                
                # Create default room
                cursor.execute("""
                    INSERT INTO rooms (room_id, room_name, location, capacity, room_type, created_at) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (room_id) DO NOTHING
                """, (
                    room_id,
                    'Healthcare Room A101',
                    'First Floor - Healthcare Wing',
                    4,
                    'patient_room',
                    datetime.now(timezone.utc)
                ))
                conn.commit()
                logger.info(f"✅ Created default room: {room_id}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not create default room {room_id}: {e}")
        finally:
            self.return_connection(conn)
    
    def _create_default_camera(self, camera_id: str, room_id: str):
        """Create default camera if not exists"""
        if not camera_id or not room_id:
            return
            
        conn = self.get_connection()
        if not conn:
            return
            
        try:
            # Get default user_id from config
            db_config = config_loader.get_database_config()
            default_user_id = db_config.get('default_ids', {}).get('fall_detection', {}).get('user_id')
            
            if not default_user_id:
                logger.warning(f"⚠️ No default user_id found for camera creation")
                return
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT camera_id FROM cameras WHERE camera_id = %s", (camera_id,))
                if cursor.fetchone():
                    return  # Camera already exists
                
                # Create default camera
                cursor.execute("""
                    INSERT INTO cameras (camera_id, user_id, room_id, camera_name, camera_type, ip_address, status, created_at) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (camera_id) DO NOTHING
                """, (
                    camera_id,
                    default_user_id,  # Add user_id
                    room_id,
                    f'Healthcare Camera - Room {room_id[-3:]}',
                    'rtsp',  # Use valid enum value for RTSP cameras
                    '192.168.8.122',
                    'active',
                    datetime.now(timezone.utc)
                ))
                conn.commit()
                logger.info(f"✅ Created default camera: {camera_id}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not create default camera {camera_id}: {e}")
        finally:
            self.return_connection(conn)

# Global service instance
postgresql_service = PostgreSQLHealthcareService()
