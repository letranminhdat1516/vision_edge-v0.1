"""
Alarm API - REST API để trigger/stop alarm độc lập không phụ thuộc lifecycle_state

Có thể integrate vào:
1. Flask/FastAPI server (nếu muốn chạy riêng Python API)
2. Hoặc call trực tiếp từ NestJS backend (via child process hoặc HTTP)

Endpoints:
- POST /api/alarm/trigger - Trigger alarm cho event
- POST /api/alarm/stop - Stop alarm
- GET /api/alarm/status - Get alarm status
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AlarmAPI:
    """
    API service để trigger/stop alarm không phụ thuộc lifecycle_state
    """
    
    def __init__(self, postgresql_service=None):
        self.postgresql_service = postgresql_service
        logger.info("🔔 AlarmAPI initialized")
    
    def set_postgresql_service(self, service):
        """Set PostgreSQL service"""
        self.postgresql_service = service
        logger.info("✅ PostgreSQL service connected to AlarmAPI")
    
    def trigger_alarm(
        self, 
        event_id: str, 
        user_id: str, 
        camera_id: str,
        triggered_by: str = "api",
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trigger alarm cho một event cụ thể
        
        KHÔNG phụ thuộc vào lifecycle_state - có thể trigger alarm bất cứ lúc nào
        
        Args:
            event_id: ID của event
            user_id: ID của user
            camera_id: ID của camera
            triggered_by: Ai trigger ('api', 'mobile_app', 'admin', etc.)
            reason: Lý do trigger (optional)
        
        Returns:
            {
                "success": True/False,
                "message": "...",
                "event_id": "...",
                "alarm_triggered": True/False,
                "timestamp": "..."
            }
        """
        try:
            if not self.postgresql_service:
                return {
                    "success": False,
                    "message": "PostgreSQL service not available",
                    "event_id": event_id,
                    "alarm_triggered": False,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"🔔 Triggering alarm for event {event_id[:8]}...")
            logger.info(f"   User: {user_id[:8]}..., Camera: {camera_id[:8]}...")
            logger.info(f"   Triggered by: {triggered_by}")
            if reason:
                logger.info(f"   Reason: {reason}")
            
            # Gửi NOTIFY qua PostgreSQL channel để trigger alarm
            success = self._send_alarm_trigger_notification(
                event_id, user_id, camera_id, triggered_by, reason
            )
            
            if success:
                # Optional: Log alarm trigger vào database (không bắt buộc)
                self._log_alarm_trigger(event_id, user_id, triggered_by, reason)
                
                logger.info(f"✅ Alarm trigger notification sent successfully for event {event_id[:8]}...")
                return {
                    "success": True,
                    "message": "Alarm triggered successfully",
                    "event_id": event_id,
                    "alarm_triggered": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"❌ Failed to send alarm trigger notification for event {event_id[:8]}...")
                return {
                    "success": False,
                    "message": "Failed to send alarm trigger notification",
                    "event_id": event_id,
                    "alarm_triggered": False,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"❌ Error triggering alarm: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": str(e),
                "event_id": event_id,
                "alarm_triggered": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def stop_alarm(
        self,
        event_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stopped_by: str = "api",
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stop alarm
        
        Args:
            event_id: ID của event (optional - nếu không có sẽ stop all alarms)
            user_id: ID của user (optional)
            stopped_by: Ai stop ('api', 'mobile_app', 'admin', etc.)
            reason: Lý do stop (optional)
        
        Returns:
            {
                "success": True/False,
                "message": "...",
                "event_id": "...",
                "alarm_stopped": True/False,
                "timestamp": "..."
            }
        """
        try:
            if not self.postgresql_service:
                return {
                    "success": False,
                    "message": "PostgreSQL service not available",
                    "event_id": event_id,
                    "alarm_stopped": False,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"🔇 Stopping alarm...")
            if event_id:
                logger.info(f"   Event: {event_id[:8]}...")
            if user_id:
                logger.info(f"   User: {user_id[:8]}...")
            logger.info(f"   Stopped by: {stopped_by}")
            if reason:
                logger.info(f"   Reason: {reason}")
            
            # Gửi NOTIFY qua PostgreSQL channel để stop alarm
            success = self._send_alarm_stop_notification(
                event_id, user_id, stopped_by, reason
            )
            
            if success:
                # QUAN TRỌNG: Log alarm stop và update state → RESOLVED
                if event_id:
                    self._log_alarm_stop(event_id, stopped_by, reason)
                    logger.info(f"📝 Event {event_id[:8]}... marked as RESOLVED")
                
                logger.info(f"✅ Alarm stop notification sent successfully")
                return {
                    "success": True,
                    "message": "Alarm stopped successfully",
                    "event_id": event_id,
                    "alarm_stopped": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"❌ Failed to send alarm stop notification")
                return {
                    "success": False,
                    "message": "Failed to send alarm stop notification",
                    "event_id": event_id,
                    "alarm_stopped": False,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"❌ Error stopping alarm: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": str(e),
                "event_id": event_id,
                "alarm_stopped": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_alarm_status(self) -> Dict[str, Any]:
        """
        Get current alarm status
        
        Returns:
            {
                "is_playing": True/False,
                "active_alarms": [...],
                "timestamp": "..."
            }
        """
        try:
            from infrastructure.services.audio_alert_service import audio_alert_service
            
            # Get active alarms from database
            active_alarms = []
            if self.postgresql_service:
                active_alarms = self._get_active_alarms()
            
            return {
                "success": True,
                "is_playing": audio_alert_service.is_playing,
                "active_alarms": active_alarms,
                "audio_backend": audio_alert_service.audio_backend,
                "volume": audio_alert_service.volume,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting alarm status: {e}")
            return {
                "success": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _send_alarm_trigger_notification(
        self, 
        event_id: str, 
        user_id: str, 
        camera_id: str,
        triggered_by: str,
        reason: Optional[str]
    ) -> bool:
        """Send PostgreSQL NOTIFY to trigger alarm"""
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            payload = json.dumps({
                'event_id': event_id,
                'user_id': user_id,
                'camera_id': camera_id,
                'action': 'TRIGGER_ALARM',
                'triggered_by': triggered_by,
                'reason': reason or 'Manual trigger via API',
                'timestamp': datetime.now().isoformat()
            })
            
            # Sử dụng channel mới cho API trigger (tách biệt với auto-alarm)
            cursor.execute("SELECT pg_notify('system_alarm_trigger_channel', %s)", (payload,))
            conn.commit()
            
            cursor.close()
            self.postgresql_service.return_connection(conn)
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to send alarm trigger notification: {e}")
            return False
    
    def _send_alarm_stop_notification(
        self,
        event_id: Optional[str],
        user_id: Optional[str],
        stopped_by: str,
        reason: Optional[str]
    ) -> bool:
        """Send PostgreSQL NOTIFY to stop alarm"""
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            payload = json.dumps({
                'event_id': event_id,
                'user_id': user_id,
                'action': 'STOP_ALARM',
                'stopped_by': stopped_by,
                'reason': reason or 'Manual stop via API',
                'timestamp': datetime.now().isoformat()
            })
            
            cursor.execute("SELECT pg_notify('system_alarm_stop_channel', %s)", (payload,))
            conn.commit()
            
            cursor.close()
            self.postgresql_service.return_connection(conn)
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to send alarm stop notification: {e}")
            return False
    
    def _log_alarm_trigger(self, event_id: str, user_id: str, triggered_by: str, reason: Optional[str]):
        """
        Log alarm trigger và UPDATE lifecycle_state → ALARM_ACTIVATED
        
        Logic:
        - enabled=true → BẬT alarm + chuyển state ALARM_ACTIVATED
        - Ghi rõ trong notes là "triggered via API"
        - Update escalated_at timestamp
        """
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            # Update lifecycle_state = ALARM_ACTIVATED + notes
            update_query = """
                UPDATE event_detections
                SET 
                    lifecycle_state = 'ALARM_ACTIVATED',
                    escalated_at = NOW(),
                    last_action_at = NOW(),
                    notes = COALESCE(notes, '') || E'\\n' || 
                            '[' || NOW()::text || '] Alarm ACTIVATED via API by ' || %s ||
                            CASE WHEN %s IS NOT NULL THEN ' - Reason: ' || %s ELSE '' END
                WHERE event_id = %s
            """
            
            cursor.execute(update_query, (triggered_by, reason, reason, event_id))
            rows_updated = cursor.rowcount
            conn.commit()
            
            if rows_updated > 0:
                logger.info(f"✅ Event {event_id[:8]}... → ALARM_ACTIVATED (manual trigger via API)")
            
            cursor.close()
            self.postgresql_service.return_connection(conn)
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to log alarm trigger: {e}")
    
    def _log_alarm_stop(self, event_id: str, stopped_by: str, reason: Optional[str]):
        """
        Log alarm stop và UPDATE lifecycle_state → RESOLVED
        
        Logic:
        - enabled=false → TẮT alarm + chuyển state RESOLVED
        - Alarm đã phát, đã có tác động → coi như đã xử lý xong
        - Ghi rõ trong notes là "stopped via API"
        - Update resolved_at timestamp
        """
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            # Update lifecycle_state = RESOLVED + notes
            # ✅ Allow API to stop ALARM_ACTIVATED, NOTIFIED, and AUTOCALLED
            # Log warning when stopping AUTOCALLED (emergency services may be involved)
            update_query = """
                UPDATE event_detections
                SET 
                    lifecycle_state = 'RESOLVED',
                    last_action_at = NOW(),
                    notes = COALESCE(notes, '') || E'\\n' || 
                            '[' || NOW()::text || '] Alarm RESOLVED via API by ' || %s ||
                            CASE WHEN %s IS NOT NULL THEN ' - Reason: ' || %s ELSE '' END
                WHERE event_id = %s
                  AND lifecycle_state IN ('ALARM_ACTIVATED', 'NOTIFIED', 'AUTOCALLED')
            """
            
            cursor.execute(update_query, (stopped_by, reason, reason, event_id))
            rows_updated = cursor.rowcount
            conn.commit()
            
            if rows_updated > 0:
                logger.info(f"✅ Event {event_id[:8]}... → RESOLVED (manual stop via API)")
            
            cursor.close()
            self.postgresql_service.return_connection(conn)
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to log alarm stop: {e}")
    
    def _get_active_alarms(self) -> list:
        """Get list of active alarms from database"""
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    event_id,
                    user_id,
                    camera_id,
                    event_type,
                    status,
                    created_at,
                    escalated_at
                FROM event_detections
                WHERE lifecycle_state = 'ALARM_ACTIVATED'
                  AND is_canceled = FALSE
                ORDER BY escalated_at DESC NULLS LAST, created_at DESC
                LIMIT 10
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            alarms = []
            for row in rows:
                alarms.append({
                    'event_id': row[0],
                    'user_id': row[1],
                    'camera_id': row[2],
                    'event_type': row[3],
                    'status': row[4],
                    'created_at': row[5].isoformat() if row[5] else None,
                    'escalated_at': row[6].isoformat() if row[6] else None
                })
            
            cursor.close()
            self.postgresql_service.return_connection(conn)
            
            return alarms
        
        except Exception as e:
            logger.error(f"❌ Failed to get active alarms: {e}")
            return []


# Singleton
alarm_api = AlarmAPI()
