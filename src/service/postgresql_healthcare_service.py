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

from config.supabase_config import supabase_config

logger = logging.getLogger(__name__)

class PostgreSQLHealthcareService:
    """Direct PostgreSQL service for healthcare events using session pooler"""
    
    def __init__(self):
        self.database_url = supabase_config.database_url
        self.connection_pool = None
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
        
        # Use provided IDs or fallback to known valid ones
        final_camera_id = camera_id or '3c0b0000-0000-4000-8000-000000000001'
        final_room_id = room_id or '2d0a0000-0000-4000-8000-000000000001' 
        final_user_id = user_id or '34e92ef3-1300-40d0-a0e0-72989cf30121'
        
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
            # Prepare record with proper fallback values
            record = {
                'event_id': str(uuid.uuid4()),
                'user_id': event_data.get('user_id') or '34e92ef3-1300-40d0-a0e0-72989cf30121',  # Default admin user
                'camera_id': event_data.get('camera_id') or '3c0b0000-0000-4000-8000-000000000001',  # Default camera
                'room_id': event_data.get('room_id') or '2d0a0000-0000-4000-8000-000000000001',  # Default room
                'snapshot_id': event_data.get('snapshot_id') or self._create_default_snapshot(
                    camera_id=event_data.get('camera_id'),
                    room_id=event_data.get('room_id'),
                    user_id=event_data.get('user_id')
                ),
                'event_type': event_data.get('event_type'),
                'event_description': event_data.get('description'),
                'detection_data': json.dumps(event_data.get('detection_data', {})),
                'ai_analysis_result': json.dumps(event_data.get('ai_analysis', {})),
                'confidence_score': float(event_data.get('confidence', 0.0)),
                'bounding_boxes': json.dumps(event_data.get('bounding_boxes', [])),
                'status': 'detected',
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

# Global service instance
postgresql_service = PostgreSQLHealthcareService()
