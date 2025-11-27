"""
Emergency Alarm Handler - PostgreSQL LISTEN/NOTIFY with psycopg3
Sử dụng PostgreSQL native LISTEN/NOTIFY để nhận realtime events
Không polling, hiệu suất cao, độ trễ thấp (< 50ms)
"""

import psycopg
import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
import os
from infrastructure.services.audio_alert_service import audio_alert_service

logger = logging.getLogger(__name__)

class EmergencyAlarmHandlerPsycopg:
    """Handler sử dụng PostgreSQL LISTEN/NOTIFY với psycopg3"""
    
    def __init__(self, postgresql_service=None):
        self.postgresql_service = postgresql_service
        self.is_running = False
        self.processed_events = set()
        self.last_cleanup_time = datetime.now()
        
        # PostgreSQL connection for LISTEN
        self.listen_conn: Optional[psycopg.Connection] = None
        
        # Database credentials - MUST use DIRECT connection (port 5432) for LISTEN/NOTIFY
        # Pooler (port 6543) does NOT support LISTEN/NOTIFY
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_name = os.getenv('DB_NAME', 'postgres')
        
        # Force port 5432 for LISTEN/NOTIFY (direct connection)
        self.database_url = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
        
        # Channel name - must match trigger function
        self.channel_name = 'system_alarm_channel'  # Match notify_alarm_trigger()
        self.stop_channel_name = 'system_alarm_stop_channel'  # For lifecycle_state changes
        
        logger.info("🎧 Emergency Alarm Handler initialized (PostgreSQL LISTEN/NOTIFY - psycopg3)")
        logger.info(f"   Using DIRECT connection (port 5432) for LISTEN/NOTIFY")
    
    def set_postgresql_service(self, service):
        """Set PostgreSQL service cho các update operations"""
        self.postgresql_service = service
        logger.info("✅ PostgreSQL service connected")
    
    async def start_listening(self):
        """Bắt đầu lắng nghe PostgreSQL notifications"""
        
        # Chạy listener trong thread riêng (vì psycopg đồng bộ)
        listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listener_thread.start()
        
        logger.info("=" * 80)
        logger.info("🚀 EMERGENCY ALARM HANDLER STARTED (LISTEN/NOTIFY)")
        logger.info("=" * 80)
        logger.info(f"📡 Alarm Channel: {self.channel_name}")
        logger.info(f"📡 Stop Channel: {self.stop_channel_name}")
        logger.info("💡 Waiting for notifications from PostgreSQL triggers...")
        logger.info("=" * 80)
        
        # Keep running và cleanup cache
        self.is_running = True
        while self.is_running:
            import asyncio
            await asyncio.sleep(60)
            
            if (datetime.now() - self.last_cleanup_time).seconds > 300:
                self._cleanup_processed_cache()
    
    def _listen_loop(self):
        """Main listener loop (chạy trong thread riêng)"""
        
        retry_delay = 5
        
        while True:
            try:
                logger.info(f"🔌 Connecting to PostgreSQL for LISTEN/NOTIFY...")
                logger.info(f"   URL: {self.database_url[:50]}...")
                
                # Connect to PostgreSQL
                self.listen_conn = psycopg.connect(self.database_url, autocommit=True)
                
                logger.info("✅ PostgreSQL connection established!")
                
                with self.listen_conn.cursor() as cur:
                    # Start listening to both channels
                    cur.execute(f"LISTEN {self.channel_name};")
                    cur.execute(f"LISTEN {self.stop_channel_name};")
                    logger.info(f"✅ Listening on channels:")
                    logger.info(f"   - {self.channel_name} (alarm activation)")
                    logger.info(f"   - {self.stop_channel_name} (alarm stop)")
                    logger.info("⚡ Ready to receive instant notifications!")
                    
                    # Poll for notifications
                    while self.is_running:
                        # Wait for notification with timeout (1 second)
                        gen = self.listen_conn.notifies(timeout=1.0)
                        
                        for notify in gen:
                            # Process notification
                            self._handle_notification(notify)
                        
                        # Small sleep to prevent tight loop
                        time.sleep(0.01)
                
            except psycopg.OperationalError as e:
                logger.error(f"❌ Connection lost: {e}")
                logger.info(f"🔄 Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
            except psycopg.InterfaceError as e:
                logger.error(f"❌ Interface error: {e}")
                logger.info(f"🔄 Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"❌ Unexpected error in listener: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(retry_delay)
            
            finally:
                if self.listen_conn:
                    try:
                        self.listen_conn.close()
                    except:
                        pass
    
    def _handle_notification(self, notify):
        """
        Xử lý notification từ PostgreSQL
        
        Args:
            notify: psycopg Notify object
                - channel: tên channel
                - payload: JSON string
        """
        try:
            logger.info("=" * 80)
            logger.info(f"🔔 NOTIFICATION RECEIVED!")
            logger.info(f"   Channel: {notify.channel}")
            logger.info(f"   Payload: {notify.payload[:200]}...")
            logger.info("=" * 80)
            
            # Parse JSON payload
            data = json.loads(notify.payload)
            
            event_id = data.get('event_id')
            state = data.get('state')
            action = data.get('action')
            message = data.get('message', '')
            
            # Check which channel sent the notification
            if notify.channel == self.stop_channel_name:
                # Stop alarm request - ANY state change should stop alarm
                logger.info(f"🔇 STOP ALARM REQUEST received")
                logger.info(f"   Event ID: {event_id}")
                logger.info(f"   New State: {state}")
                logger.info(f"   Reason: {message}")
                self._process_alarm_stop_sync(data)
                return
            
            # Avoid duplicates for alarm activation
            if event_id in self.processed_events:
                logger.info(f"⏭️  Event {event_id} already processed, skipping")
                return
            
            logger.info(f"📋 Event Details:")
            logger.info(f"   Event ID: {event_id}")
            logger.info(f"   User ID: {data.get('user_id')}")
            logger.info(f"   Camera ID: {data.get('camera_id')}")
            logger.info(f"   State: {state}")
            logger.info(f"   Message: {message}")
            
            # Process ONLY alarm activation - trigger only fires for ALARM_ACTIVATED
            if state == 'ALARM_ACTIVATED':
                self._process_alarm_activated_sync(data)
            else:
                logger.info(f"ℹ️ State {state} - no action needed")

            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON payload: {e}")
            logger.error(f"   Raw payload: {notify.payload}")
        except Exception as e:
            logger.error(f"❌ Error handling notification: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _process_emergency_request_sync(self, event_data: Dict[str, Any]):
        """Xử lý manual_emergency event (synchronous)"""
        try:
            event_id = str(event_data.get('event_id', ''))
            user_id = str(event_data.get('user_id', ''))
            
            self.processed_events.add(event_id)
            
            logger.info(f"🚨 Processing MANUAL EMERGENCY: {event_id}")
            
            # Trigger alarm (dùng asyncio.run để chạy async function)
            import asyncio
            alarm_result = asyncio.run(audio_alert_service.play_emergency_alarm(
                user_id=user_id,
                triggered_by='manual_emergency',
                duration=10
            ))
            
            if alarm_result['success']:
                logger.info("✅ ✅ ✅ ALARM ACTIVATED SUCCESSFULLY! ✅ ✅ ✅")
                logger.info(f"   Volume: {alarm_result.get('volume', 1.0) * 100:.0f}%")
                logger.info(f"   Duration: {alarm_result.get('duration', 10)}s")
                
                # Update event status
                self._update_event_status(
                    event_id=event_id,
                    lifecycle_state='ACKNOWLEDGED',
                    status='danger',
                    confirmation_state='CONFIRMED_BY_CUSTOMER',
                    verification_status='APPROVED',
                    notes=f"ALARM ACTIVATED at {datetime.now()}: {event_data.get('event_description', '')}"
                )
            else:
                logger.error(f"❌ ALARM FAILED: {alarm_result['message']}")
                
                # Update to error state
                self._update_event_status(
                    event_id=event_id,
                    lifecycle_state='CANCELED',
                    notes=f"Alarm activation failed: {alarm_result['message']}"
                )
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error processing emergency: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _process_alarm_activated_sync(self, event_data: Dict[str, Any]):
        """Xử lý alarm_activated event - CHỈ PHÁ CÒI, KHÔNG UPDATE STATE"""
        try:
            event_id = str(event_data.get('event_id', ''))
            user_id = str(event_data.get('user_id', ''))
            
            self.processed_events.add(event_id)
            
            logger.info(f"🚨 Processing ALARM ACTIVATION: {event_id}")
            logger.info(f"   Old state: {event_data.get('old_lifecycle_state')}")
            logger.info(f"   New state: {event_data.get('new_lifecycle_state')}")
            
            # Trigger alarm - NO DURATION LIMIT (will play until stopped)
            import asyncio
            alarm_result = asyncio.run(audio_alert_service.play_emergency_alarm(
                user_id=user_id,
                triggered_by='alarm_activation',
                duration=0  # 0 = infinite, no auto-stop
            ))
            
            if alarm_result['success']:
                logger.info("✅ ✅ ✅ ALARM PLAYING! ✅ ✅ ✅")
                logger.info(f"   Volume: {alarm_result.get('volume', 1.0) * 100:.0f}%")
                logger.info(f"   Duration: INFINITE (until state change)")
                logger.info(f"   📍 Alarm will play until lifecycle_state changes")
                # ❌ KHÔNG UPDATE STATE - Để mobile/user tự update khi muốn dừng
            else:
                logger.error(f"❌ ALARM FAILED: {alarm_result['message']}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error processing alarm: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _process_alarm_stop_sync(self, event_data: Dict[str, Any]):
        """Xử lý alarm stop request (synchronous)"""
        try:
            event_id = str(event_data.get('event_id', ''))
            reason = event_data.get('message', 'Unknown reason')
            
            logger.info(f"🔇 Processing ALARM STOP: {event_id}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Old state: {event_data.get('old_lifecycle_state')}")
            logger.info(f"   New state: {event_data.get('new_lifecycle_state')}")
            
            # Stop alarm
            import asyncio
            stop_result = asyncio.run(audio_alert_service.stop_alarm())
            
            if stop_result['success']:
                logger.info("✅ ✅ ✅ ALARM STOPPED SUCCESSFULLY! ✅ ✅ ✅")
                logger.info(f"   Event: {event_id[:8]}...")
                logger.info(f"   Reason: {reason}")
            else:
                logger.warning(f"⚠️ No alarm was playing: {stop_result['message']}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error stopping alarm: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def resolve_active_alarms(self, reason="Auto-resolved"):
        """
        Resolve all ALARM_ACTIVATED events to RESOLVED
        Called when alarm is stopped by system (2 people detected or situation normalized)
        """
        try:
            if not self.postgresql_service:
                logger.warning("⚠️ PostgreSQL service not available for resolve")
                return 0
            
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            # Find and update ALARM_ACTIVATED events to RESOLVED
            cursor.execute("""
                UPDATE event_detections
                SET 
                    lifecycle_state = 'RESOLVED',
                    last_action_at = NOW(),
                    notes = COALESCE(notes, '') || '\n' || %s || ' at ' || NOW()::text
                WHERE lifecycle_state = 'ALARM_ACTIVATED'
                RETURNING event_id, event_type
            """, (reason,))
            
            resolved_events = cursor.fetchall()
            conn.commit()
            cursor.close()
            self.postgresql_service.return_connection(conn)
            
            if resolved_events:
                for event_id, event_type in resolved_events:
                    logger.info(f"📝 Event {event_id[:8]}... ({event_type}) → RESOLVED")
            
            return len(resolved_events)
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve active alarms: {e}")
            return 0
    
    def _update_event_status(self, event_id: str, **kwargs):
        """Update event status in database"""
        try:
            if not self.postgresql_service:
                logger.warning("⚠️ PostgreSQL service not available for update")
                return
            
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            # Build UPDATE query
            set_clauses = ["last_action_at = NOW()"]
            params = []
            
            if 'lifecycle_state' in kwargs:
                set_clauses.append(f"lifecycle_state = %s")
                params.append(kwargs['lifecycle_state'])
            
            if 'status' in kwargs:
                set_clauses.append(f"status = %s")
                params.append(kwargs['status'])
            
            if 'confirmation_state' in kwargs:
                set_clauses.append(f"confirmation_state = %s")
                params.append(kwargs['confirmation_state'])
            
            if 'verification_status' in kwargs:
                set_clauses.append(f"verification_status = %s")
                params.append(kwargs['verification_status'])
            
            if 'notes' in kwargs:
                set_clauses.append(f"notes = %s")
                params.append(kwargs['notes'])
            
            params.append(event_id)
            
            query = f"""
                UPDATE event_detections
                SET {', '.join(set_clauses)}
                WHERE event_id = %s
            """
            
            cursor.execute(query, params)
            conn.commit()
            cursor.close()
            self.postgresql_service.return_connection(conn)
            
            logger.info(f"✅ Event {event_id[:8]}... updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to update event: {e}")
    
    def _cleanup_processed_cache(self):
        """Cleanup cache"""
        if len(self.processed_events) > 1000:
            logger.info(f"🧹 Cleaning cache ({len(self.processed_events)} items)")
            self.processed_events.clear()
            self.last_cleanup_time = datetime.now()
    
    def stop(self):
        """Stop handler"""
        self.is_running = False
        
        if self.listen_conn:
            try:
                self.listen_conn.close()
                logger.info("✅ PostgreSQL connection closed")
            except:
                pass
        
        logger.info("🛑 Emergency Alarm Handler stopped")


# Singleton
emergency_alarm_handler = EmergencyAlarmHandlerPsycopg()
