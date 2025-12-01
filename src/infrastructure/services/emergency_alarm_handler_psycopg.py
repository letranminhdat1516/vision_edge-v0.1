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
        
        # Channel names - listen to API triggers (không phụ thuộc lifecycle_state)
        self.trigger_channel_name = 'system_alarm_trigger_channel'  # API trigger alarm
        self.stop_channel_name = 'system_alarm_stop_channel'  # API stop alarm
        
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
        logger.info(f"📡 Trigger Channel: {self.trigger_channel_name}")
        logger.info(f"📡 Stop Channel: {self.stop_channel_name}")
        logger.info("💡 Waiting for API trigger notifications (independent of lifecycle_state)...")
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
                    cur.execute(f"LISTEN {self.trigger_channel_name};")
                    cur.execute(f"LISTEN {self.stop_channel_name};")
                    logger.info(f"✅ Listening on channels:")
                    logger.info(f"   - {self.trigger_channel_name} (API trigger)")
                    logger.info(f"   - {self.stop_channel_name} (API stop)")
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
            action = data.get('action')
            user_id = data.get('user_id')
            camera_id = data.get('camera_id')
            
            # Check which channel sent the notification
            if notify.channel == self.stop_channel_name:
                # Stop alarm request
                logger.info(f"🔇 STOP ALARM REQUEST received")
                logger.info(f"   Event ID: {event_id}")
                logger.info(f"   Action: {action}")
                logger.info(f"   Reason: {data.get('reason', 'N/A')}")
                self._process_alarm_stop_sync(data)
                return
            
            # Trigger channel
            if notify.channel == self.trigger_channel_name:
                # Avoid duplicates
                if event_id and event_id in self.processed_events:
                    logger.info(f"⏭️  Event {event_id} already processed, skipping")
                    return
                
                logger.info(f"📋 Trigger Request:")
                logger.info(f"   Event ID: {event_id}")
                logger.info(f"   User ID: {user_id}")
                logger.info(f"   Camera ID: {camera_id}")
                logger.info(f"   Action: {action}")
                logger.info(f"   Triggered by: {data.get('triggered_by', 'unknown')}")
                
                # Process alarm trigger
                if action == 'TRIGGER_ALARM':
                    self._process_alarm_trigger_sync(data)
                else:
                    logger.warning(f"⚠️ Unknown action: {action}")

            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON payload: {e}")
            logger.error(f"   Raw payload: {notify.payload}")
        except Exception as e:
            logger.error(f"❌ Error handling notification: {e}")
            import traceback
            logger.error(traceback.format_exc())
    

    
    def _process_alarm_trigger_sync(self, event_data: Dict[str, Any]):
        """
        Xử lý alarm trigger request từ API
        CHỈ PHÁ CÒI - KHÔNG CẬP NHẬT DATABASE
        Lifecycle_state được quản lý riêng bởi API/Worker
        """
        try:
            event_id = str(event_data.get('event_id', ''))
            user_id = str(event_data.get('user_id', ''))
            triggered_by = event_data.get('triggered_by', 'api')
            
            if event_id:
                self.processed_events.add(event_id)
            
            logger.info(f"🚨 Processing ALARM TRIGGER: {event_id[:8] if event_id else 'N/A'}...")
            logger.info(f"   User: {user_id[:8] if user_id else 'N/A'}...")
            logger.info(f"   Triggered by: {triggered_by}")
            
            # Trigger alarm - NO DURATION LIMIT (will play until stopped)
            import asyncio
            alarm_result = asyncio.run(audio_alert_service.play_emergency_alarm(
                user_id=user_id or 'system',
                triggered_by=triggered_by,
                duration=0  # 0 = infinite, no auto-stop
            ))
            
            if alarm_result['success']:
                logger.info("✅ ✅ ✅ ALARM PLAYING! ✅ ✅ ✅")
                logger.info(f"   Volume: {alarm_result.get('volume', 1.0) * 100:.0f}%")
                logger.info(f"   Duration: INFINITE (until stop command)")
                logger.info(f"   📍 Alarm will play until stop_alarm() is called")
            else:
                logger.error(f"❌ ALARM FAILED: {alarm_result['message']}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error processing alarm trigger: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _process_alarm_stop_sync(self, event_data: Dict[str, Any]):
        """
        Xử lý alarm stop request từ API
        CHỈ TẮT CÒI - KHÔNG CẬP NHẬT DATABASE
        Lifecycle_state được quản lý riêng bởi API/Worker
        """
        try:
            event_id = str(event_data.get('event_id', '')) if event_data.get('event_id') else 'N/A'
            reason = event_data.get('reason', 'Unknown reason')
            stopped_by = event_data.get('stopped_by', 'api')
            
            logger.info(f"🔇 Processing ALARM STOP: {event_id[:8] if event_id != 'N/A' else 'N/A'}...")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Stopped by: {stopped_by}")
            
            # Stop alarm - NO MINIMUM DURATION CHECK
            # Minimum duration được quản lý bởi Worker service
            import asyncio
            stop_result = asyncio.run(audio_alert_service.stop_alarm())
            
            if stop_result['success']:
                logger.info("✅ ✅ ✅ ALARM STOPPED SUCCESSFULLY! ✅ ✅ ✅")
                logger.info(f"   Event: {event_id[:8] if event_id != 'N/A' else 'N/A'}...")
                logger.info(f"   Reason: {reason}")
            else:
                logger.warning(f"⚠️ No alarm was playing: {stop_result['message']}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error stopping alarm: {e}")
            import traceback
            logger.error(traceback.format_exc())
    

    
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
