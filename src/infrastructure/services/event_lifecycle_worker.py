"""
Event Lifecycle Worker - Background service for auto-alarm management
Tự động trigger alarm cho events danger/warning sau 30s không xử lý
Tự động stop alarm và resolve khi có events normal sau 30s
Chạy mỗi 10 giây (giống NestJS @Cron('*/10 * * * * *'))
"""

import logging
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

class EventLifecycleWorker:
    """
    Worker service theo dõi lifecycle của events
    - Auto-alarm: Events danger/warning chưa xử lý sau 30s → trigger alarm
    - Auto-stop: Có event normal sau 30s từ danger/warning → stop alarm và resolve
    """
    
    def __init__(self, postgresql_service=None):
        self.postgresql_service = postgresql_service
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Configuration (có thể load từ .env)
        self.batch_size = int(os.getenv('EVENT_LIFECYCLE_BATCH_SIZE', '50'))
        self.check_interval = 10  # seconds - chạy mỗi 10s
        self.alarm_delay_seconds = 30  # seconds - delay trước khi auto-alarm
        self.resolve_delay_seconds = 30  # seconds - delay trước khi auto-resolve
        
        # Escalatable statuses
        self.escalatable_statuses = ['danger', 'warning']
        
        # Terminal states (không auto-alarm nếu đã ở states này)
        self.terminal_states = [
            'ACKNOWLEDGED',
            'RESOLVED',
            'EMERGENCY_RESPONSE_RECEIVED',
            'EMERGENCY_ESCALATION_FAILED',
            'CANCELED'
        ]
        
        logger.info("🔄 EventLifecycleWorker initialized")
        logger.info(f"   ⏱️  Check interval: {self.check_interval}s")
        logger.info(f"   ⏰ Alarm delay: {self.alarm_delay_seconds}s")
        logger.info(f"   ✅ Resolve delay: {self.resolve_delay_seconds}s")
    
    def set_postgresql_service(self, service):
        """Set PostgreSQL service"""
        self.postgresql_service = service
        logger.info("✅ PostgreSQL service connected to EventLifecycleWorker")
    
    def start(self):
        """Start worker trong background thread"""
        if self.is_running:
            logger.warning("EventLifecycleWorker already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("=" * 80)
        logger.info("🚀 EVENT LIFECYCLE WORKER STARTED")
        logger.info("=" * 80)
        logger.info(f"📡 Monitoring events for auto-alarm and auto-resolve")
        logger.info(f"⏱️  Running every {self.check_interval} seconds")
        logger.info("=" * 80)
    
    def _worker_loop(self):
        """Main worker loop - runs every 10 seconds"""
        while self.is_running:
            try:
                # Chạy checks
                alarm_count = self._check_and_promote_to_alarm()
                resolve_count = self._check_and_auto_resolve()
                
                if alarm_count > 0 or resolve_count > 0:
                    logger.info(f"[EventLifecycleWorker] Tick completed (alarm={alarm_count}, resolve={resolve_count})")
                
            except Exception as e:
                logger.error(f"[EventLifecycleWorker] Error in worker loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Sleep 10 seconds before next check
            time.sleep(self.check_interval)
    
    def _check_and_promote_to_alarm(self) -> int:
        """
        Kiểm tra events danger/warning chưa xử lý sau 30s → Trigger alarm
        
        Logic:
        1. Tìm events với:
           - lifecycle_state = 'NOTIFIED'
           - status IN ('danger', 'warning')
           - acknowledged_at = NULL (chưa ai xử lý)
           - is_canceled = FALSE
           - created_at <= NOW() - 30s
        
        2. Trigger alarm bằng cách gửi notification qua PostgreSQL channel
           (emergency_alarm_handler sẽ nhận và play alarm)
        
        Returns:
            Number of events promoted to alarm
        """
        if not self.postgresql_service:
            return 0
        
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            # Calculate cutoff time (30 seconds ago)
            cutoff_time = datetime.now() - timedelta(seconds=self.alarm_delay_seconds)
            
            # Find candidate events
            query = """
                SELECT 
                    event_id,
                    user_id,
                    camera_id,
                    event_type,
                    status,
                    confidence_score,
                    created_at
                FROM event_detections
                WHERE lifecycle_state = 'NOTIFIED'
                  AND status IN ('danger', 'warning')
                  AND acknowledged_at IS NULL
                  AND is_canceled = FALSE
                  AND created_at <= %s
                ORDER BY created_at ASC
                LIMIT %s
            """
            
            cursor.execute(query, (cutoff_time, self.batch_size))
            candidates = cursor.fetchall()
            
            if not candidates:
                cursor.close()
                self.postgresql_service.return_connection(conn)
                return 0
            
            logger.info(f"🔍 Found {len(candidates)} events pending alarm (>{self.alarm_delay_seconds}s old)")
            
            promoted = 0
            for row in candidates:
                # ✅ RealDictCursor returns dict-like rows, access by key
                event_id = row['event_id']
                user_id = row['user_id']
                camera_id = row['camera_id']
                event_type = row['event_type']
                status = row['status']
                confidence = row['confidence_score']
                created_at = row['created_at']
                
                try:
                    # ============================================================================
                    # QUAN TRỌNG: Đảm bảo tính toàn vẹn dữ liệu (Data Integrity)
                    # ============================================================================
                    # Bước 1: Trigger alarm TRƯỚC KHI update lifecycle_state
                    #   - Nếu trigger thất bại → không update lifecycle (tránh false positive)
                    #   - Nếu trigger thành công → mới update lifecycle = ALARM_ACTIVATED
                    #
                    # Quy tắc: lifecycle_state = ALARM_ACTIVATED <=> alarm đã được trigger
                    # ============================================================================
                    
                    # Trigger alarm via PostgreSQL NOTIFY
                    success = self._trigger_alarm_via_notify(event_id, user_id, camera_id, event_type)
                    
                    if success:
                        # ✅ Alarm đã được trigger thành công
                        # → An toàn để update lifecycle_state = ALARM_ACTIVATED
                        update_query = """
                            UPDATE event_detections
                            SET 
                                lifecycle_state = 'ALARM_ACTIVATED',
                                escalated_at = NOW(),
                                auto_escalation_reason = 'alarm_timeout',
                                last_action_at = NOW(),
                                notes = COALESCE(notes, '') || E'\\n' || 
                                        '[' || NOW()::text || '] Auto-alarm activated after 30s timeout'
                            WHERE event_id = %s
                              AND lifecycle_state = 'NOTIFIED'
                        """
                        
                        cursor.execute(update_query, (event_id,))
                        conn.commit()
                        
                        promoted += 1
                        logger.info(f"✅ Event {event_id[:8]}... → ALARM_ACTIVATED (auto-alarm after {self.alarm_delay_seconds}s)")
                        logger.info(f"   Type: {event_type}, Status: {status}, Confidence: {confidence:.2f}")
                    
                    else:
                        # ❌ Alarm trigger thất bại
                        # → KHÔNG update lifecycle_state (giữ nguyên NOTIFIED)
                        logger.warning(f"⚠️ Failed to trigger alarm for event {event_id[:8]}..., skipping lifecycle update")
                
                except Exception as e:
                    # ✅ Rollback transaction khi có lỗi (prevents "current transaction is aborted")
                    try:
                        conn.rollback()
                    except:
                        pass
                    logger.error(f"❌ Error promoting event {event_id[:8]}... to alarm: {e}")
                    continue
            
            cursor.close()
            self.postgresql_service.return_connection(conn)
            
            return promoted
        
        except Exception as e:
            # ✅ Handle SSL connection errors gracefully
            if "SSL connection has been closed" in str(e) or "connection" in str(e).lower():
                logger.warning(f"⚠️ Connection error in _check_and_promote_to_alarm: {e}")
                logger.info("🔄 Will retry on next cycle with fresh connection")
                
                # Close bad connection
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                if conn:
                    try:
                        conn.close()  # Force close bad connection
                    except:
                        pass
            else:
                logger.error(f"❌ Error in _check_and_promote_to_alarm: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            return 0
        
        finally:
            # ✅ Always cleanup resources
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    self.postgresql_service.return_connection(conn)
                except:
                    pass
    
    def _trigger_alarm_via_notify(self, event_id: str, user_id: str, camera_id: str, event_type: str) -> bool:
        """
        Trigger alarm bằng cách gửi NOTIFY qua PostgreSQL channel
        
        emergency_alarm_handler sẽ lắng nghe và play alarm
        
        Returns:
            True nếu gửi thành công, False nếu thất bại
        """
        conn = None
        cursor = None
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            # Build notification payload
            import json
            payload = json.dumps({
                'event_id': str(event_id),  # ✅ Convert UUID to string properly
                'user_id': str(user_id),
                'camera_id': str(camera_id),
                'event_type': str(event_type),
                'action': 'TRIGGER_ALARM',
                'triggered_by': 'auto_alarm_worker',
                'timestamp': datetime.now().isoformat()
            })
            
            # Send NOTIFY to PostgreSQL channel
            cursor.execute("SELECT pg_notify('system_alarm_trigger_channel', %s)", (payload,))
            conn.commit()
            
            logger.debug(f"📡 Sent alarm trigger notification for event {event_id[:8]}...")
            return True
        
        except Exception as e:
            if conn:
                try:
                    conn.rollback()  # ✅ Rollback failed transaction
                except:
                    pass
            logger.error(f"❌ Failed to send alarm trigger notification: {e}")
            return False
        
        finally:
            # ✅ CRITICAL: Always return connection to pool
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    self.postgresql_service.return_connection(conn)
                except:
                    pass
    
    def _check_and_auto_resolve(self) -> int:
        """
        Kiểm tra và tự động resolve events khi có normal status sau 30s
        
        Logic:
        1. Tìm events với:
           - lifecycle_state = 'ALARM_ACTIVATED'
           - Có event mới cùng user với status = 'normal'
           - Event normal được tạo sau event danger/warning >= 30s
        
        2. Stop alarm và update lifecycle_state → 'RESOLVED'
        
        Returns:
            Number of events resolved
        """
        if not self.postgresql_service:
            return 0
        
        conn = None
        cursor = None
        
        try:
            conn = self.postgresql_service.get_connection()
            
            # ✅ Validate connection before use (prevents SSL errors)
            if conn.closed:
                logger.warning("⚠️ Connection closed, getting new one")
                self.postgresql_service.return_connection(conn)
                conn = self.postgresql_service.get_connection()
            
            cursor = conn.cursor()
            
            # Find ALARM_ACTIVATED events có normal event sau 30s
            query = """
                WITH alarm_events AS (
                    SELECT 
                        e1.event_id,
                        e1.user_id,
                        e1.camera_id,
                        e1.created_at as alarm_time,
                        e1.event_type,
                        e1.escalated_at  -- ✅ Added: needed for ORDER BY
                    FROM event_detections e1
                    WHERE e1.lifecycle_state = 'ALARM_ACTIVATED'
                      AND e1.is_canceled = FALSE
                ),
                normal_events AS (
                    SELECT 
                        e2.user_id,
                        e2.camera_id,
                        MAX(e2.created_at) as latest_normal_time
                    FROM event_detections e2
                    WHERE e2.status = 'normal'
                    GROUP BY e2.user_id, e2.camera_id
                )
                SELECT 
                    a.event_id,
                    a.user_id,
                    a.camera_id,
                    a.event_type,
                    a.alarm_time,
                    n.latest_normal_time
                FROM alarm_events a
                JOIN normal_events n ON a.user_id = n.user_id AND a.camera_id = n.camera_id
                WHERE n.latest_normal_time > a.alarm_time
                  AND EXTRACT(EPOCH FROM (n.latest_normal_time - a.alarm_time)) >= %s
                ORDER BY a.escalated_at ASC  -- ✅ Fixed: use escalated_at instead of created_at
                LIMIT %s
            """
            
            cursor.execute(query, (self.resolve_delay_seconds, self.batch_size))
            candidates = cursor.fetchall()
            
            if not candidates:
                cursor.close()
                self.postgresql_service.return_connection(conn)
                return 0
            
            logger.info(f"🔍 Found {len(candidates)} alarms to auto-resolve (normal status after {self.resolve_delay_seconds}s)")
            
            resolved = 0
            for row in candidates:
                # ✅ RealDictCursor returns dict-like rows, access by key
                event_id = row['event_id']
                user_id = row['user_id']
                camera_id = row['camera_id']
                event_type = row['event_type']
                alarm_time = row['alarm_time']
                normal_time = row['latest_normal_time']
                
                try:
                    # Stop alarm via NOTIFY
                    self._stop_alarm_via_notify(event_id, user_id, camera_id)
                    
                    # Update lifecycle_state → RESOLVED
                    update_query = """
                        UPDATE event_detections
                        SET 
                            lifecycle_state = 'RESOLVED',
                            resolved_at = NOW(),
                            last_action_at = NOW(),
                            notes = COALESCE(notes, '') || E'\\n' || 
                                    '[' || NOW()::text || '] Auto-resolved: normal status detected after 30s'
                        WHERE event_id = %s
                          AND lifecycle_state = 'ALARM_ACTIVATED'
                    """
                    
                    cursor.execute(update_query, (event_id,))
                    conn.commit()
                    
                    resolved += 1
                    time_diff = (normal_time - alarm_time).total_seconds()
                    logger.info(f"✅ Event {event_id[:8]}... → RESOLVED (normal detected after {time_diff:.0f}s)")
                    logger.info(f"   Type: {event_type}, User: {user_id[:8]}...")
                
                except Exception as e:
                    # ✅ Rollback transaction khi có lỗi (prevents "current transaction is aborted")
                    try:
                        conn.rollback()
                    except:
                        pass
                    logger.error(f"❌ Error resolving event {event_id[:8]}...: {e}")
                    continue
            
            cursor.close()
            self.postgresql_service.return_connection(conn)
            
            return resolved
        
        except Exception as e:
            # ✅ Handle SSL connection errors gracefully
            if "SSL connection has been closed" in str(e) or "connection" in str(e).lower():
                logger.warning(f"⚠️ Connection error in _check_and_auto_resolve: {e}")
                logger.info("🔄 Will retry on next cycle with fresh connection")
                
                # Close bad connection
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                if conn:
                    try:
                        conn.close()  # Force close bad connection
                    except:
                        pass
            else:
                logger.error(f"❌ Error in _check_and_auto_resolve: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            return 0
        
        finally:
            # ✅ Always cleanup resources
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    self.postgresql_service.return_connection(conn)
                except:
                    pass
    
    def _stop_alarm_via_notify(self, event_id: str, user_id: str, camera_id: str):
        """
        Stop alarm bằng cách gửi NOTIFY qua PostgreSQL channel
        """
        conn = None
        cursor = None
        try:
            conn = self.postgresql_service.get_connection()
            cursor = conn.cursor()
            
            import json
            payload = json.dumps({
                'event_id': str(event_id),  # ✅ Convert UUID to string properly
                'user_id': str(user_id),
                'camera_id': str(camera_id),
                'action': 'STOP_ALARM',
                'reason': 'auto_resolved_normal_status',
                'timestamp': datetime.now().isoformat()
            })
            
            cursor.execute("SELECT pg_notify('system_alarm_stop_channel', %s)", (payload,))
            conn.commit()
            
            logger.debug(f"📡 Sent alarm stop notification for event {event_id[:8]}...")
        
        except Exception as e:
            if conn:
                try:
                    conn.rollback()  # ✅ Rollback failed transaction
                except:
                    pass
            logger.error(f"❌ Failed to send alarm stop notification: {e}")
        
        finally:
            # ✅ CRITICAL: Always return connection to pool
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    self.postgresql_service.return_connection(conn)
                except:
                    pass
    
    def stop(self):
        """Stop worker"""
        self.is_running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            logger.info("⏳ Waiting for worker thread to stop...")
            self.worker_thread.join(timeout=15)
        
        logger.info("🛑 EventLifecycleWorker stopped")


# Singleton
event_lifecycle_worker = EventLifecycleWorker()
