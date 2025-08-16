"""
Alert State Manager - Priority-based Alert System
Manages healthcare alerts based on severity/danger level instead of time
"""

import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    """Alert priority levels based on severity and status"""
    CRITICAL_ACTIVE = 5      # Highest - Critical + Active
    HIGH_ACTIVE = 4          # High + Active  
    MEDIUM_ACTIVE = 3        # Medium + Active
    CRITICAL_ACK = 2         # Critical + Acknowledged
    LOW_ACTIVE = 2           # Low + Active
    HIGH_ACK = 1             # High + Acknowledged
    RESOLVED = 0             # Lowest - Any resolved

class AlertStateManager:
    """Manages alert states with priority-based logic"""
    
    def __init__(self, postgresql_service):
        self.db = postgresql_service
        self.logger = logging.getLogger(__name__)
        
        # Priority mapping: (severity, status) -> priority_level
        self.PRIORITY_MAP = {
            ('critical', 'active'): AlertPriority.CRITICAL_ACTIVE.value,
            ('high', 'active'): AlertPriority.HIGH_ACTIVE.value,
            ('medium', 'active'): AlertPriority.MEDIUM_ACTIVE.value,
            ('low', 'active'): AlertPriority.LOW_ACTIVE.value,
            ('critical', 'acknowledged'): AlertPriority.CRITICAL_ACK.value,
            ('high', 'acknowledged'): AlertPriority.HIGH_ACK.value,
            ('medium', 'acknowledged'): AlertPriority.HIGH_ACK.value,
            ('low', 'acknowledged'): AlertPriority.HIGH_ACK.value,
        }
        
        # Event type to severity mapping based on confidence
        self.SEVERITY_THRESHOLDS = {
            'seizure': {
                'critical': 0.75,  # >= 75%
                'high': 0.50,      # 50-74%
                'medium': 0.30,    # 30-49%
                'low': 0.0         # < 30%
            },
            'fall': {
                'critical': 0.80,  # >= 80%
                'high': 0.60,      # 60-79%
                'medium': 0.40,    # 40-59%
                'low': 0.0         # < 40%
            },
            'abnormal_behavior': {
                'critical': 0.70,  # >= 70%
                'high': 0.50,      # 50-69%
                'medium': 0.30,    # 30-49%
                'low': 0.0         # < 30%
            }
        }
    
    def determine_alert_severity(self, event_type: str, confidence: float) -> str:
        """Determine alert severity based on event type and confidence"""
        thresholds = self.SEVERITY_THRESHOLDS.get(event_type, self.SEVERITY_THRESHOLDS['fall'])
        
        if confidence >= thresholds['critical']:
            return 'critical'
        elif confidence >= thresholds['high']:
            return 'high'
        elif confidence >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def calculate_priority(self, severity: str, status: str) -> int:
        """Calculate priority level from severity and status"""
        return self.PRIORITY_MAP.get((severity, status), 0)
    
    def get_active_alerts_by_priority(self, user_id: str, room_id: Optional[str] = None) -> List[Dict]:
        """Get active alerts ordered by priority (highest first)"""
        if not self.db.is_connected:
            self.logger.warning("Database not connected")
            return []
        
        conn = self.db.get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cursor:
                # Build query with priority calculation
                base_sql = """
                SELECT 
                    a.alert_id,
                    a.event_id,
                    a.user_id,
                    a.alert_type,
                    a.severity,
                    a.alert_message,
                    a.status,
                    a.acknowledged_by,
                    a.acknowledged_at,
                    a.resolution_notes,
                    a.created_at,
                    a.resolved_at,
                    e.event_type,
                    e.confidence_score,
                    e.detected_at,
                    e.event_description,
                    CASE 
                        WHEN a.severity = 'critical' AND a.status = 'active' THEN 5
                        WHEN a.severity = 'high' AND a.status = 'active' THEN 4
                        WHEN a.severity = 'medium' AND a.status = 'active' THEN 3
                        WHEN a.severity = 'critical' AND a.status = 'acknowledged' THEN 2
                        WHEN a.severity = 'low' AND a.status = 'active' THEN 2
                        WHEN a.severity IN ('high', 'medium', 'low') AND a.status = 'acknowledged' THEN 1
                        ELSE 0
                    END as priority_level
                FROM alerts a
                JOIN event_detections e ON a.event_id = e.event_id
                WHERE a.user_id = %s 
                  AND a.resolved_at IS NULL
                """
                
                params = [user_id]
                
                if room_id:
                    base_sql += " AND e.room_id = %s"
                    params.append(room_id)
                
                # Order by priority (highest first), then by creation time
                base_sql += " ORDER BY priority_level DESC, a.created_at DESC"
                
                cursor.execute(base_sql, params)
                alerts = cursor.fetchall()
                
                # Convert to dict list
                result = []
                for alert in alerts:
                    alert_dict = dict(alert) if hasattr(alert, 'keys') else {
                        'alert_id': alert[0], 'event_id': alert[1], 'user_id': alert[2],
                        'alert_type': alert[3], 'severity': alert[4], 'alert_message': alert[5],
                        'status': alert[6], 'acknowledged_by': alert[7], 'acknowledged_at': alert[8],
                        'resolution_notes': alert[9], 'created_at': alert[10], 'resolved_at': alert[11],
                        'event_type': alert[12], 'confidence_score': alert[13], 'detected_at': alert[14],
                        'event_description': alert[15], 'priority_level': alert[16]
                    }
                    result.append(alert_dict)
                
                self.logger.info(f"Retrieved {len(result)} alerts for user {user_id}, ordered by priority")
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []
        finally:
            self.db.return_connection(conn)
    
    def should_create_alert(self, event_data: Dict) -> bool:
        """Determine if new event should create an alert based on current active alerts"""
        user_id = event_data.get('user_id')
        if not user_id:
            return True  # Create alert if no user context
        
        # Get current active alerts
        active_alerts = self.get_active_alerts_by_priority(user_id)
        
        # Calculate new event priority
        new_severity = self.determine_alert_severity(
            event_data.get('event_type', ''),
            event_data.get('confidence_score', 0.0)
        )
        new_priority = self.calculate_priority(new_severity, 'active')
        
        # If no active alerts, create new one
        if not active_alerts:
            self.logger.info(f"No active alerts - creating new alert with severity {new_severity}")
            return True
        
        # Get highest priority of current alerts
        current_max_priority = max(alert.get('priority_level', 0) for alert in active_alerts)
        
        # Only create if new priority is higher or equal to critical/high
        should_create = (new_priority > current_max_priority or 
                        new_priority >= AlertPriority.HIGH_ACTIVE.value)
        
        self.logger.info(f"Event priority: {new_priority}, Current max: {current_max_priority}, Should create: {should_create}")
        return should_create
    
    def create_alert(self, event_data: Dict) -> Optional[str]:
        """Create new alert with proper severity based on event data"""
        if not self.db.is_connected:
            self.logger.warning("Database not connected")
            return None
        
        conn = self.db.get_connection()
        if not conn:
            return None
        
        try:
            # Determine severity from confidence score
            severity = self.determine_alert_severity(
                event_data.get('event_type', ''),
                event_data.get('confidence_score', 0.0)
            )
            
            # Generate alert message based on severity and type
            alert_message = self._generate_alert_message(
                event_data.get('event_type', ''),
                severity,
                event_data.get('confidence_score', 0.0)
            )
            
            alert_id = str(uuid.uuid4())
            
            with conn.cursor() as cursor:
                insert_sql = """
                INSERT INTO alerts (
                    alert_id, event_id, user_id, alert_type, severity,
                    alert_message, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING alert_id
                """
                
                cursor.execute(insert_sql, (
                    alert_id,
                    event_data.get('event_id'),
                    event_data.get('user_id'),
                    event_data.get('event_type', 'healthcare'),
                    severity,
                    alert_message,
                    'active',  # Always start as active
                    datetime.now(timezone.utc)
                ))
                
                result = cursor.fetchone()
                conn.commit()
                
                if result:
                    created_alert_id = result[0] if isinstance(result, (list, tuple)) else result.get('alert_id')
                    self.logger.info(f"Created alert {created_alert_id} with severity {severity}")
                    return created_alert_id
                    
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            conn.rollback()
            return None
        finally:
            self.db.return_connection(conn)
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Mark alert as acknowledged by user"""
        if not self.db.is_connected:
            return False
        
        conn = self.db.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE alerts 
                    SET status = 'acknowledged',
                        acknowledged_by = %s,
                        acknowledged_at = %s
                    WHERE alert_id = %s AND status = 'active'
                """, (user_id, datetime.now(timezone.utc), alert_id))
                
                updated = cursor.rowcount > 0
                conn.commit()
                
                if updated:
                    self.logger.info(f"Alert {alert_id} acknowledged by user {user_id}")
                
                return updated
                
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            conn.rollback()
            return False
        finally:
            self.db.return_connection(conn)
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_notes: Optional[str] = None) -> bool:
        """Mark alert as resolved"""
        if not self.db.is_connected:
            return False
        
        conn = self.db.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE alerts 
                    SET status = 'resolved',
                        resolved_at = %s,
                        resolution_notes = %s
                    WHERE alert_id = %s AND status IN ('active', 'acknowledged')
                """, (datetime.now(timezone.utc), resolution_notes, alert_id))
                
                updated = cursor.rowcount > 0
                conn.commit()
                
                if updated:
                    self.logger.info(f"Alert {alert_id} resolved by user {user_id}")
                
                return updated
                
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            conn.rollback()
            return False
        finally:
            self.db.return_connection(conn)
    
    def auto_resolve_old_alerts(self, user_id: str, max_age_hours: int = 24):
        """Auto-resolve old acknowledged alerts after specified hours"""
        if not self.db.is_connected:
            return
        
        conn = self.db.get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE alerts 
                    SET status = 'resolved',
                        resolved_at = %s,
                        resolution_notes = 'Auto-resolved due to age'
                    WHERE user_id = %s 
                      AND status = 'acknowledged'
                      AND acknowledged_at < %s - INTERVAL '%s hours'
                """, (
                    datetime.now(timezone.utc),
                    user_id,
                    datetime.now(timezone.utc),
                    max_age_hours
                ))
                
                resolved_count = cursor.rowcount
                conn.commit()
                
                if resolved_count > 0:
                    self.logger.info(f"Auto-resolved {resolved_count} old alerts for user {user_id}")
                    
        except Exception as e:
            self.logger.error(f"Error auto-resolving alerts: {e}")
            conn.rollback()
        finally:
            self.db.return_connection(conn)
    
    def _generate_alert_message(self, event_type: str, severity: str, confidence: float) -> str:
        """Generate appropriate alert message based on event type and severity"""
        confidence_pct = int(confidence * 100)
        
        if severity == 'critical':
            if event_type == 'seizure':
                return f"🚨 CRITICAL: Seizure detected ({confidence_pct}% confidence) - Immediate medical attention required"
            elif event_type == 'fall':
                return f"🚨 CRITICAL: Fall detected ({confidence_pct}% confidence) - Emergency response needed"
            else:
                return f"🚨 CRITICAL: {event_type} detected ({confidence_pct}% confidence) - Urgent attention required"
        
        elif severity == 'high':
            if event_type == 'seizure':
                return f"⚠️ HIGH: Possible seizure ({confidence_pct}% confidence) - Medical monitoring required"
            elif event_type == 'fall':
                return f"⚠️ HIGH: Possible fall ({confidence_pct}% confidence) - Assistance needed"
            else:
                return f"⚠️ HIGH: {event_type} detected ({confidence_pct}% confidence) - Attention required"
        
        elif severity == 'medium':
            return f"ℹ️ MEDIUM: {event_type} activity detected ({confidence_pct}% confidence) - Monitoring recommended"
        
        else:  # low
            return f"📊 LOW: {event_type} activity detected ({confidence_pct}% confidence) - Routine monitoring"
    
    def get_alert_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get alert statistics for dashboard"""
        if not self.db.is_connected:
            return {}
        
        conn = self.db.get_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor() as cursor:
                # Get counts by severity and status
                cursor.execute("""
                    SELECT 
                        severity, status, COUNT(*) as count
                    FROM alerts 
                    WHERE user_id = %s 
                      AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    GROUP BY severity, status
                """, (user_id,))
                
                stats = cursor.fetchall()
                
                result = {
                    'total_alerts_24h': 0,
                    'active_critical': 0,
                    'active_high': 0,
                    'acknowledged_count': 0,
                    'resolved_count': 0
                }
                
                for stat in stats:
                    severity, status, count = stat[0], stat[1], stat[2]
                    result['total_alerts_24h'] += count
                    
                    if status == 'active':
                        if severity == 'critical':
                            result['active_critical'] += count
                        elif severity == 'high':
                            result['active_high'] += count
                    elif status == 'acknowledged':
                        result['acknowledged_count'] += count
                    elif status == 'resolved':
                        result['resolved_count'] += count
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {e}")
            return {}
        finally:
            self.db.return_connection(conn)

# Global instance
alert_state_manager = None

def get_alert_state_manager(postgresql_service):
    """Get global alert state manager instance"""
    global alert_state_manager
    if alert_state_manager is None:
        alert_state_manager = AlertStateManager(postgresql_service)
    return alert_state_manager
