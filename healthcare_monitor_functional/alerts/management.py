#!/usr/bin/env python3
"""
Alert Management Functions  
Function-based alert system for healthcare monitoring
"""

import time
import json
import logging
import smtplib
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def create_alert(alert_type: str, message: str, confidence: float,
                detection_data: Dict[str, Any] = None, 
                timestamp: float = None) -> Dict[str, Any]:
    """
    Create an alert dictionary
    
    Args:
        alert_type: Type of alert ('fall', 'seizure', 'motion', 'system')
        message: Alert message
        confidence: Detection confidence (0-1)
        detection_data: Additional detection data
        timestamp: Alert timestamp (None for current time)
        
    Returns:
        Alert dictionary
    """
    if timestamp is None:
        timestamp = time.time()
    
    alert = {
        'id': f"{alert_type}_{int(timestamp)}",
        'type': alert_type,
        'message': message,
        'confidence': confidence,
        'timestamp': timestamp,
        'datetime': datetime.fromtimestamp(timestamp).isoformat(),
        'status': 'active',
        'data': detection_data or {}
    }
    
    return alert


def validate_alert(alert: Dict[str, Any]) -> bool:
    """
    Validate alert structure and data
    
    Args:
        alert: Alert dictionary
        
    Returns:
        True if alert is valid
    """
    required_fields = ['type', 'message', 'confidence', 'timestamp']
    
    for field in required_fields:
        if field not in alert:
            return False
    
    # Validate confidence range
    if not (0.0 <= alert['confidence'] <= 1.0):
        return False
    
    # Validate alert type
    valid_types = ['fall', 'seizure', 'motion', 'system', 'warning']
    if alert['type'] not in valid_types:
        return False
    
    return True


def determine_alert_priority(alert: Dict[str, Any]) -> str:
    """
    Determine alert priority based on type and confidence
    
    Args:
        alert: Alert dictionary
        
    Returns:
        Priority level ('low', 'medium', 'high', 'critical')
    """
    alert_type = alert.get('type', 'system')
    confidence = alert.get('confidence', 0.0)
    
    if alert_type in ['fall', 'seizure']:
        if confidence >= 0.8:
            return 'critical'
        elif confidence >= 0.6:
            return 'high'
        else:
            return 'medium'
    
    elif alert_type == 'motion':
        if confidence >= 0.9:
            return 'high'
        elif confidence >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    else:  # system alerts
        return 'low'


class AlertManager:
    """Manages healthcare monitoring alerts"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.active_alerts = []
        self.alert_history = []
        self.last_alert_time = {}
        self.alert_cooldown = self.config.get('alert_cooldown', 30)  # seconds
        self.max_history = self.config.get('max_history', 1000)
        
        # Notification settings
        self.email_enabled = self.config.get('email_notifications', False)
        self.webhook_enabled = self.config.get('webhook_notifications', False)
        self.log_enabled = self.config.get('log_notifications', True)
    
    def add_alert(self, alert_type: str, message: str, confidence: float,
                  detection_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Add new alert with cooldown protection
        
        Args:
            alert_type: Type of alert
            message: Alert message
            confidence: Detection confidence
            detection_data: Additional detection data
            
        Returns:
            Created alert or None if in cooldown
        """
        current_time = time.time()
        
        # Check cooldown
        if self._is_in_cooldown(alert_type, current_time):
            return None
        
        # Create alert
        alert = create_alert(alert_type, message, confidence, detection_data, current_time)
        
        if not validate_alert(alert):
            logging.error(f"Invalid alert created: {alert}")
            return None
        
        # Add priority
        alert['priority'] = determine_alert_priority(alert)
        
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Add to history
        self.alert_history.append(alert.copy())
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Update last alert time
        self.last_alert_time[alert_type] = current_time
        
        # Send notifications
        self._send_notifications(alert)
        
        logging.info(f"Alert created: {alert['type']} - {alert['message']} ({alert['confidence']:.2f})")
        
        return alert
    
    def _is_in_cooldown(self, alert_type: str, current_time: float) -> bool:
        """Check if alert type is in cooldown period"""
        last_time = self.last_alert_time.get(alert_type, 0)
        return (current_time - last_time) < self.alert_cooldown
    
    def _send_notifications(self, alert: Dict[str, Any]):
        """Send alert notifications via configured channels"""
        if self.log_enabled:
            self._log_alert(alert)
        
        if self.email_enabled:
            self._send_email_notification(alert)
        
        if self.webhook_enabled:
            self._send_webhook_notification(alert)
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to file"""
        try:
            log_message = (f"ALERT [{alert['priority'].upper()}] "
                          f"{alert['type']}: {alert['message']} "
                          f"(confidence: {alert['confidence']:.2f})")
            
            if alert['priority'] in ['critical', 'high']:
                logging.error(log_message)
            elif alert['priority'] == 'medium':
                logging.warning(log_message)
            else:
                logging.info(log_message)
                
        except Exception as e:
            logging.error(f"Failed to log alert: {str(e)}")
    
    def _send_email_notification(self, alert: Dict[str, Any]):
        """Send email notification for alert"""
        try:
            email_config = self.config.get('email', {})
            
            if not email_config.get('enabled', False):
                return
            
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username')
            password = email_config.get('password')
            recipients = email_config.get('recipients', [])
            
            if not all([smtp_server, username, password, recipients]):
                logging.warning("Email configuration incomplete")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Healthcare Monitor Alert - {alert['type'].title()}"
            
            # Email body
            body = f"""
Healthcare Monitoring Alert

Alert Type: {alert['type'].title()}
Priority: {alert['priority'].upper()}
Message: {alert['message']}
Confidence: {alert['confidence']:.2f}
Time: {alert['datetime']}

Detection Data:
{json.dumps(alert.get('data', {}), indent=2)}

Please check the monitoring system for more details.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.sendmail(username, recipients, msg.as_string())
            
            logging.info(f"Email notification sent for alert: {alert['id']}")
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {str(e)}")
    
    def _send_webhook_notification(self, alert: Dict[str, Any]):
        """Send webhook notification for alert"""
        try:
            webhook_config = self.config.get('webhook', {})
            
            if not webhook_config.get('enabled', False):
                return
            
            webhook_url = webhook_config.get('url')
            headers = webhook_config.get('headers', {'Content-Type': 'application/json'})
            timeout = webhook_config.get('timeout', 10)
            
            if not webhook_url:
                logging.warning("Webhook URL not configured")
                return
            
            # Prepare payload
            payload = {
                'alert': alert,
                'source': 'healthcare_monitor',
                'timestamp': alert['timestamp']
            }
            
            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                logging.info(f"Webhook notification sent for alert: {alert['id']}")
            else:
                logging.warning(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {str(e)}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was resolved
        """
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_time'] = time.time()
                self.active_alerts.remove(alert)
                logging.info(f"Alert resolved: {alert_id}")
                return True
        
        return False
    
    def get_active_alerts(self, alert_type: str = None) -> List[Dict[str, Any]]:
        """
        Get active alerts, optionally filtered by type
        
        Args:
            alert_type: Filter by alert type (None for all)
            
        Returns:
            List of active alerts
        """
        if alert_type is None:
            return self.active_alerts.copy()
        else:
            return [alert for alert in self.active_alerts if alert['type'] == alert_type]
    
    def get_alert_statistics(self, time_window: float = 3600) -> Dict[str, Any]:
        """
        Get alert statistics for a time window
        
        Args:
            time_window: Time window in seconds (default: 1 hour)
            
        Returns:
            Alert statistics dictionary
        """
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Filter recent alerts
        recent_alerts = [alert for alert in self.alert_history 
                        if alert['timestamp'] >= cutoff_time]
        
        # Count by type
        type_counts = {}
        priority_counts = {}
        
        for alert in recent_alerts:
            alert_type = alert['type']
            priority = alert.get('priority', 'unknown')
            
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            'time_window_hours': time_window / 3600,
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'alerts_by_type': type_counts,
            'alerts_by_priority': priority_counts,
            'last_alert_time': max([alert['timestamp'] for alert in recent_alerts]) if recent_alerts else None
        }
    
    def clear_old_alerts(self, max_age: float = 24 * 3600):
        """
        Clear old resolved alerts from active list
        
        Args:
            max_age: Maximum age in seconds (default: 24 hours)
        """
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        # Remove old resolved alerts
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.get('resolved_time', current_time) > cutoff_time
        ]
        
        logging.info(f"Cleared old alerts, {len(self.active_alerts)} active alerts remaining")


def format_alert_message(alert_type: str, confidence: float, 
                        detection_data: Dict[str, Any] = None) -> str:
    """
    Format alert message based on type and data
    
    Args:
        alert_type: Type of alert
        confidence: Detection confidence
        detection_data: Additional detection data
        
    Returns:
        Formatted alert message
    """
    if alert_type == 'fall':
        return f"Fall detected with {confidence:.1%} confidence"
    
    elif alert_type == 'seizure':
        return f"Seizure activity detected with {confidence:.1%} confidence"
    
    elif alert_type == 'motion':
        motion_level = detection_data.get('motion_level', 0.0) if detection_data else 0.0
        return f"High motion activity detected (level: {motion_level:.1%})"
    
    elif alert_type == 'system':
        return detection_data.get('message', 'System alert') if detection_data else 'System alert'
    
    else:
        return f"Alert: {alert_type} (confidence: {confidence:.1%})"


def should_suppress_alert(alert_type: str, confidence: float, 
                         recent_alerts: List[Dict[str, Any]], 
                         suppression_rules: Dict[str, Any] = None) -> bool:
    """
    Check if alert should be suppressed based on recent activity
    
    Args:
        alert_type: Type of alert to check
        confidence: Alert confidence
        recent_alerts: List of recent alerts
        suppression_rules: Custom suppression rules
        
    Returns:
        True if alert should be suppressed
    """
    if not suppression_rules:
        suppression_rules = {
            'fall': {'min_interval': 10, 'min_confidence_increase': 0.1},
            'seizure': {'min_interval': 15, 'min_confidence_increase': 0.05},
            'motion': {'min_interval': 5, 'min_confidence_increase': 0.2}
        }
    
    rules = suppression_rules.get(alert_type, {})
    min_interval = rules.get('min_interval', 5)
    min_confidence_increase = rules.get('min_confidence_increase', 0.1)
    
    current_time = time.time()
    
    for recent_alert in recent_alerts:
        if recent_alert['type'] != alert_type:
            continue
        
        time_diff = current_time - recent_alert['timestamp']
        
        # Check time interval
        if time_diff < min_interval:
            # Check if confidence increased significantly
            confidence_diff = confidence - recent_alert['confidence']
            if confidence_diff < min_confidence_increase:
                return True  # Suppress
    
    return False  # Don't suppress
