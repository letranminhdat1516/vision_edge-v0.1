"""
Supabase Realtime Service for Healthcare Monitoring
Handles real-time communication with Supabase for events, alerts, and system status
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from supabase import create_client, Client
import threading
import logging
import time

from config.supabase_config import supabase_config

logger = logging.getLogger(__name__)

class SupabaseRealtimeService:
    """Service for handling Supabase realtime operations"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.service_client: Optional[Client] = None
        self.is_connected = False
        self.polling_threads = {}
        self.event_handlers = {}
        self.last_check_times = {}
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Supabase clients"""
        try:
            # Regular client for realtime subscriptions
            client_config = supabase_config.get_client_config()
            self.client = create_client(client_config['url'], client_config['key'])
            
            # Service role client for admin operations
            service_config = supabase_config.get_service_config()
            self.service_client = create_client(service_config['url'], service_config['key'])
            
            logger.info("Supabase clients initialized successfully")
            self.is_connected = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase clients: {e}")
            self.is_connected = False
    
    def subscribe_to_events(self, table: str, event_type: str, handler: Callable):
        """
        Subscribe to realtime events for a specific table using polling approach
        
        Args:
            table: Table name to subscribe to
            event_type: Type of event ('INSERT', 'UPDATE', 'DELETE', '*')
            handler: Callback function to handle events
        """
        if not self.client or not self.is_connected:
            logger.error("Supabase client not initialized")
            return
        
        try:
            subscription_key = f"{table}_{event_type}"
            self.event_handlers[subscription_key] = handler
            self.last_check_times[subscription_key] = datetime.now(timezone.utc)
            
            # Start polling thread for this table
            polling_thread = threading.Thread(
                target=self._poll_table_changes,
                args=(table, event_type, handler),
                daemon=True
            )
            polling_thread.start()
            
            self.polling_threads[subscription_key] = polling_thread
            logger.info(f"Started polling for {table} {event_type} events")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {table} events: {e}")
    
    def _poll_table_changes(self, table: str, event_type: str, handler: Callable):
        """Poll table for changes"""
        subscription_key = f"{table}_{event_type}"
        
        while subscription_key in self.polling_threads:
            try:
                if not self.client or not self.is_connected:
                    logger.warning(f"Client not connected, stopping poll for {subscription_key}")
                    break
                
                # Get last check time
                last_check = self.last_check_times.get(subscription_key)
                
                # Query for new records since last check
                if event_type in ['INSERT', '*']:
                    query = self.client.table(table).select('*')
                    if last_check:
                        query = query.gte('created_at', last_check.isoformat())
                    
                    result = query.order('created_at', desc=False).execute()
                    
                    if result.data:
                        for record in result.data:
                            event_data = {
                                'event_type': 'INSERT',
                                'table': table,
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'new_data': record,
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
                
                # Wait before next poll
                time.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                logger.error(f"Error polling {table}: {e}")
                time.sleep(5)  # Wait longer on error
    
    def publish_event_detection(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Publish a new event detection to the database
        
        Args:
            event_data: Event detection data
            
        Returns:
            Inserted record data or None if failed
        """
        if not self.service_client:
            logger.error("Service client not initialized")
            return None
        
        try:
            # Prepare event detection record
            record = {
                'event_id': str(uuid.uuid4()),
                'user_id': event_data.get('user_id'),
                'camera_id': event_data.get('camera_id'),
                'room_id': event_data.get('room_id'),
                'snapshot_id': event_data.get('snapshot_id'),
                'event_type': event_data.get('event_type'),
                'event_description': event_data.get('description'),
                'detection_data': event_data.get('detection_data', {}),
                'ai_analysis_result': event_data.get('ai_analysis', {}),
                'confidence_score': event_data.get('confidence', 0.0),
                'bounding_boxes': event_data.get('bounding_boxes', []),
                'status': 'detected',
                'context_data': event_data.get('context', {}),
                'detected_at': datetime.now(timezone.utc).isoformat(),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Insert into database
            result = self.service_client.table('event_detections').insert(record).execute()
            
            if result.data:
                logger.info(f"Event detection published: {record['event_type']} with confidence {record['confidence_score']}")
                return result.data[0]
            else:
                logger.error("Failed to publish event detection")
                return None
                
        except Exception as e:
            logger.error(f"Error publishing event detection: {e}")
            return None
    
    def publish_alert(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Publish a new alert to the database
        
        Args:
            alert_data: Alert data
            
        Returns:
            Inserted alert record or None if failed
        """
        if not self.service_client:
            logger.error("Service client not initialized")
            return None
        
        try:
            # Prepare alert record
            record = {
                'alert_id': str(uuid.uuid4()),
                'event_id': alert_data.get('event_id'),
                'user_id': alert_data.get('user_id'),
                'alert_type': alert_data.get('alert_type'),
                'severity': alert_data.get('severity', 'medium'),
                'alert_message': alert_data.get('message'),
                'alert_data': alert_data.get('alert_data', {}),
                'status': 'active',
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Insert into database
            result = self.service_client.table('alerts').insert(record).execute()
            
            if result.data:
                logger.info(f"Alert published: {record['alert_type']} - {record['severity']}")
                return result.data[0]
            else:
                logger.error("Failed to publish alert")
                return None
                
        except Exception as e:
            logger.error(f"Error publishing alert: {e}")
            return None
    
    def publish_snapshot(self, snapshot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Publish a new snapshot record to the database
        
        Args:
            snapshot_data: Snapshot data
            
        Returns:
            Inserted snapshot record or None if failed
        """
        if not self.service_client:
            logger.error("Service client not initialized")
            return None
        
        try:
            # Prepare snapshot record
            record = {
                'snapshot_id': str(uuid.uuid4()),
                'camera_id': snapshot_data.get('camera_id'),
                'room_id': snapshot_data.get('room_id'),
                'user_id': snapshot_data.get('user_id'),
                'image_path': snapshot_data.get('image_path'),
                'cloud_url': snapshot_data.get('cloud_url'),
                'metadata': snapshot_data.get('metadata', {}),
                'capture_type': snapshot_data.get('capture_type', 'alert'),
                'captured_at': datetime.now(timezone.utc).isoformat(),
                'is_processed': False
            }
            
            # Insert into database
            result = self.service_client.table('snapshots').insert(record).execute()
            
            if result.data:
                logger.info(f"Snapshot published: {record['image_path']}")
                return result.data[0]
            else:
                logger.error("Failed to publish snapshot")
                return None
                
        except Exception as e:
            logger.error(f"Error publishing snapshot: {e}")
            return None
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events from database"""
        if not self.client:
            return []
        
        try:
            result = (
                self.client
                .table('event_detections')
                .select('*')
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    def unsubscribe_all(self):
        """Stop all polling threads"""
        try:
            # Stop all polling threads
            threads_to_stop = list(self.polling_threads.keys())
            for subscription_key in threads_to_stop:
                del self.polling_threads[subscription_key]
                logger.info(f"Stopped polling for {subscription_key}")
            
            self.event_handlers.clear()
            self.last_check_times.clear()
            
        except Exception as e:
            logger.error(f"Error stopping polling threads: {e}")
    
    def close(self):
        """Close all connections and clean up"""
        self.unsubscribe_all()
        self.is_connected = False
        logger.info("Supabase realtime service closed")

# Global service instance
realtime_service = SupabaseRealtimeService()
