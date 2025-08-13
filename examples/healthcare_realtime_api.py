#!/usr/bin/env python3
"""
Flask API Server for Healthcare Realtime Events
Provides REST API endpoints for HTML to get real events from database
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import threading
import time
import json

# Load environment
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for HTML

# Database connection parameters
DB_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'postgres')
}

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected' if get_db_connection() else 'disconnected'
    })

@app.route('/api/events/latest')
def get_latest_events():
    """Get latest healthcare events from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Get latest events ordered by detected_at
        cursor.execute("""
            SELECT 
                event_id,
                event_type,
                confidence_score,
                detected_at,
                camera_id,
                detection_data,
                context_data,
                created_at
            FROM event_detections 
            ORDER BY detected_at DESC, created_at DESC
            LIMIT 20
        """)
        
        events = cursor.fetchall()
        
        # Convert to list of dicts
        events_list = []
        for event in events:
            event_dict = dict(event)
            # Convert datetime to string
            if event_dict['detected_at']:
                event_dict['detected_at'] = event_dict['detected_at'].isoformat()
            if event_dict['created_at']:
                event_dict['created_at'] = event_dict['created_at'].isoformat()
            events_list.append(event_dict)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'events': events_list,
            'count': len(events_list),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error getting events: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/events/new')
def get_new_events():
    """Get new events since a specific timestamp or ID"""
    try:
        # Get query parameters
        since_id = request.args.get('since_id', type=int, default=0)
        since_time = request.args.get('since_time')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Build query based on parameters
        if since_time:
            cursor.execute("""
                SELECT 
                    event_id,
                    event_type,
                    confidence_score,
                    detected_at,
                    camera_id,
                    detection_data,
                    context_data,
                    created_at
                FROM event_detections 
                WHERE detected_at > %s
                ORDER BY detected_at DESC, created_at DESC
                LIMIT 50
            """, (since_time,))
        else:
            cursor.execute("""
                SELECT 
                    event_id,
                    event_type,
                    confidence_score,
                    detected_at,
                    camera_id,
                    detection_data,
                    context_data,
                    created_at
                FROM event_detections 
                WHERE event_id::text > %s
                ORDER BY detected_at DESC, created_at DESC
                LIMIT 50
            """, (str(since_id),))
        
        events = cursor.fetchall()
        
        # Convert to list of dicts
        events_list = []
        for event in events:
            event_dict = dict(event)
            # Convert datetime to string
            if event_dict['detected_at']:
                event_dict['detected_at'] = event_dict['detected_at'].isoformat()
            if event_dict['created_at']:
                event_dict['created_at'] = event_dict['created_at'].isoformat()
            events_list.append(event_dict)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'events': events_list,
            'count': len(events_list),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error getting new events: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get healthcare statistics"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(CASE WHEN event_type = 'fall' THEN 1 END) as fall_events,
                COUNT(CASE WHEN event_type = 'abnormal_behavior' THEN 1 END) as seizure_events,
                MAX(detected_at) as latest_event_time
            FROM event_detections
        """)
        
        stats = cursor.fetchone()
        stats_dict = dict(stats)
        
        # Convert datetime to string
        if stats_dict['latest_event_time']:
            stats_dict['latest_event_time'] = stats_dict['latest_event_time'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': stats_dict,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Serve HTML page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Healthcare Real-time Monitor</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f5f5f5; 
            }
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            .status { 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px; 
                text-align: center; 
                font-weight: bold; 
            }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
            .stats { 
                display: grid; 
                grid-template-columns: repeat(4, 1fr); 
                gap: 15px; 
                margin: 20px 0; 
            }
            .stat-card { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
            }
            .stat-number { 
                font-size: 24px; 
                font-weight: bold; 
                color: #2196F3; 
            }
            .event { 
                background: #e3f2fd; 
                border-left: 4px solid #2196F3; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px; 
            }
            .event.danger { 
                background: #ffebee; 
                border-left-color: #f44336; 
            }
            .event.warning { 
                background: #fff8e1; 
                border-left-color: #ff9800; 
            }
            button { 
                padding: 10px 20px; 
                margin: 5px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                font-weight: bold; 
            }
            .start-btn { background: #4CAF50; color: white; }
            .stop-btn { background: #f44336; color: white; }
            .events-container { 
                max-height: 600px; 
                overflow-y: auto; 
            }
            .log { 
                background: #000; 
                color: #00ff00; 
                padding: 10px; 
                height: 150px; 
                overflow-y: auto; 
                font-family: monospace; 
                font-size: 12px; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Healthcare Real-time Monitor</h1>
            <p>Kết nối trực tiếp với PostgreSQL database - Nhận events thật từ hệ thống</p>
            
            <div id="status" class="status disconnected">❌ Not connected</div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number" id="total-events">0</div>
                    <div>Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="fall-events">0</div>
                    <div>Fall Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="seizure-events">0</div>
                    <div>Seizure Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="latest-time">--:--</div>
                    <div>Latest Event</div>
                </div>
            </div>
            
            <div>
                <button class="start-btn" onclick="startMonitoring()">🚀 Start Real-time Monitor</button>
                <button class="stop-btn" onclick="stopMonitoring()">⏹️ Stop</button>
                <button onclick="loadLatestEvents()" style="background: #2196F3; color: white;">🔄 Refresh</button>
                <button onclick="clearEvents()" style="background: #6c757d; color: white;">🗑️ Clear</button>
            </div>
            
            <div>
                <h3>📝 Activity Log</h3>
                <div id="log" class="log"></div>
            </div>
            
            <div>
                <h3>📱 Live Healthcare Events</h3>
                <div id="events" class="events-container"></div>
            </div>
        </div>

        <script>
            let isMonitoring = false;
            let pollInterval = null;
            let lastEventId = 0;
            let totalEvents = 0;
            let fallEvents = 0;
            let seizureEvents = 0;
            
            const statusEl = document.getElementById('status');
            const logEl = document.getElementById('log');
            const eventsEl = document.getElementById('events');
            
            function log(message) {
                const time = new Date().toLocaleTimeString();
                const logMessage = `[${time}] ${message}`;
                console.log(logMessage);
                logEl.innerHTML += logMessage + '\\n';
                logEl.scrollTop = logEl.scrollHeight;
            }
            
            function updateStatus(connected, message) {
                statusEl.className = `status ${connected ? 'connected' : 'disconnected'}`;
                statusEl.textContent = (connected ? '✅ ' : '❌ ') + message;
            }
            
            function updateStats(stats) {
                if (stats) {
                    totalEvents = stats.total_events || 0;
                    fallEvents = stats.fall_events || 0;
                    seizureEvents = stats.seizure_events || 0;
                    
                    document.getElementById('total-events').textContent = totalEvents;
                    document.getElementById('fall-events').textContent = fallEvents;
                    document.getElementById('seizure-events').textContent = seizureEvents;
                    
                    if (stats.latest_event_time) {
                        const time = new Date(stats.latest_event_time).toLocaleTimeString();
                        document.getElementById('latest-time').textContent = time;
                    }
                }
            }
            
            async function startMonitoring() {
                if (isMonitoring) return;
                
                try {
                    // Test API connection
                    const healthResponse = await fetch('/api/health');
                    if (!healthResponse.ok) throw new Error('API not available');
                    
                    const health = await healthResponse.json();
                    if (health.database !== 'connected') {
                        throw new Error('Database not connected');
                    }
                    
                    isMonitoring = true;
                    updateStatus(true, 'Connected - Polling database for real events');
                    log('🚀 Started real-time monitoring');
                    log('📡 Connected to PostgreSQL database');
                    
                    // Load initial stats and events
                    await loadStats();
                    await loadLatestEvents();
                    
                    // Start polling for new events every 2 seconds
                    pollInterval = setInterval(pollForNewEvents, 2000);
                    
                } catch (error) {
                    log(`❌ Failed to start monitoring: ${error.message}`);
                    updateStatus(false, 'Connection failed');
                }
            }
            
            function stopMonitoring() {
                if (!isMonitoring) return;
                
                isMonitoring = false;
                if (pollInterval) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                }
                updateStatus(false, 'Monitoring stopped');
                log('⏹️ Stopped real-time monitoring');
            }
            
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    if (!response.ok) throw new Error('Failed to load stats');
                    
                    const data = await response.json();
                    if (data.success) {
                        updateStats(data.stats);
                        log(`📊 Stats loaded: ${data.stats.total_events} total events`);
                    }
                } catch (error) {
                    log(`❌ Failed to load stats: ${error.message}`);
                }
            }
            
            async function loadLatestEvents() {
                try {
                    const response = await fetch('/api/events/latest');
                    if (!response.ok) throw new Error('Failed to load events');
                    
                    const data = await response.json();
                    if (data.success) {
                        // Clear existing events and add new ones
                        eventsEl.innerHTML = '';
                        data.events.forEach(event => displayEvent(event, false));
                        
                        if (data.events.length > 0) {
                            lastEventId = Math.max(...data.events.map(e => e.id));
                            log(`📋 Loaded ${data.events.length} latest events`);
                        }
                    }
                } catch (error) {
                    log(`❌ Failed to load events: ${error.message}`);
                }
            }
            
            async function pollForNewEvents() {
                if (!isMonitoring) return;
                
                try {
                    const response = await fetch(`/api/events/new?since_id=${lastEventId}`);
                    if (!response.ok) return;
                    
                    const data = await response.json();
                    if (data.success && data.events.length > 0) {
                        // Add new events to the top
                        data.events.reverse().forEach(event => {
                            displayEvent(event, true);
                            lastEventId = Math.max(lastEventId, event.id);
                        });
                        
                        // Update stats
                        await loadStats();
                        
                        log(`📨 Received ${data.events.length} new events`);
                    }
                } catch (error) {
                    log(`⚠️ Poll error: ${error.message}`);
                }
            }
            
            function displayEvent(eventData, isNew = false) {
                // Determine status
                let status = 'normal';
                if (eventData.event_type === 'fall') {
                    if (eventData.confidence_score >= 0.8) status = 'danger';
                    else if (eventData.confidence_score >= 0.6) status = 'warning';
                } else if (eventData.event_type === 'abnormal_behavior') {
                    if (eventData.confidence_score >= 0.7) status = 'danger';
                    else if (eventData.confidence_score >= 0.5) status = 'warning';
                }
                
                const eventEl = document.createElement('div');
                eventEl.className = `event ${status}`;
                
                const time = new Date(eventData.detected_at).toLocaleString('vi-VN');
                const confidence = Math.round(eventData.confidence_score * 100);
                
                // Generate Vietnamese action message
                const eventName = eventData.event_type === 'fall' ? 'té ngã' : 'co giật';
                let action = '';
                if (status === 'danger') {
                    action = `🚨 KHẨN CẤP - Phát hiện ${eventName} với độ tin cậy ${confidence}%. Cần hỗ trợ ngay lập tức!`;
                } else if (status === 'warning') {
                    action = `⚠️ Cảnh báo - Phát hiện ${eventName} với độ tin cậy ${confidence}%. Cần theo dõi.`;
                } else {
                    action = `ℹ️ Tình huống bình thường - ${eventName} được phát hiện với độ tin cậy ${confidence}%`;
                }
                
                eventEl.innerHTML = `
                    <div><strong>🚨 ${eventData.event_type.toUpperCase().replace('_', ' ')} DETECTED</strong></div>
                    <div style="margin: 5px 0; font-style: italic;">${action}</div>
                    <div><strong>📊 Confidence:</strong> ${confidence}% | <strong>Status:</strong> ${status.toUpperCase()}</div>
                    <div><strong>🕒 Time:</strong> ${time}</div>
                    <div><strong>📍 Location:</strong> ${eventData.location || 'Unknown'}</div>
                    <div><strong>📹 Camera:</strong> ${eventData.camera_id || 'Unknown'}</div>
                    <div><strong>🆔 ID:</strong> ${eventData.id}</div>
                `;
                
                if (isNew) {
                    // Add to top with animation for new events
                    eventEl.style.opacity = '0';
                    eventEl.style.transform = 'translateY(-10px)';
                    eventsEl.insertBefore(eventEl, eventsEl.firstChild);
                    
                    setTimeout(() => {
                        eventEl.style.transition = 'all 0.3s ease';
                        eventEl.style.opacity = '1';
                        eventEl.style.transform = 'translateY(0)';
                    }, 50);
                    
                    // Play sound and show notification
                    playNotificationSound(status);
                    showDesktopNotification(eventData, action);
                } else {
                    // Add normally for initial load
                    eventsEl.appendChild(eventEl);
                }
                
                // Keep only last 30 events
                while (eventsEl.children.length > 30) {
                    eventsEl.removeChild(eventsEl.lastChild);
                }
            }
            
            function playNotificationSound(status) {
                try {
                    const context = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = context.createOscillator();
                    const gainNode = context.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(context.destination);
                    
                    if (status === 'danger') {
                        oscillator.frequency.setValueAtTime(900, context.currentTime);
                        gainNode.gain.setValueAtTime(0.1, context.currentTime);
                    } else if (status === 'warning') {
                        oscillator.frequency.setValueAtTime(650, context.currentTime);
                        gainNode.gain.setValueAtTime(0.08, context.currentTime);
                    } else {
                        oscillator.frequency.setValueAtTime(450, context.currentTime);
                        gainNode.gain.setValueAtTime(0.05, context.currentTime);
                    }
                    
                    oscillator.start();
                    oscillator.stop(context.currentTime + 0.2);
                } catch (error) {
                    // Ignore audio errors
                }
            }
            
            function showDesktopNotification(eventData, action) {
                if (Notification.permission === 'granted') {
                    const title = `Healthcare Alert: ${eventData.event_type.replace('_', ' ')}`;
                    new Notification(title, {
                        body: action,
                        icon: '🏥'
                    });
                }
            }
            
            function clearEvents() {
                eventsEl.innerHTML = '';
                log('🗑️ Events display cleared');
            }
            
            // Request notification permission
            if (Notification.permission === 'default') {
                Notification.requestPermission();
            }
            
            // Initialize
            log('🏥 Healthcare Real-time Monitor initialized');
            log('💡 This connects directly to PostgreSQL database');
            log('📋 Click "Start Real-time Monitor" to begin');
            updateStatus(false, 'Ready to connect');
        </script>
    </body>
    </html>
    """)

if __name__ == '__main__':
    print("🏥 Starting Healthcare Real-time API Server")
    print("📡 This server provides real events from PostgreSQL database")
    print("🔗 Access: http://localhost:5000")
    print("📋 API Endpoints:")
    print("   - GET /api/health - Health check")
    print("   - GET /api/events/latest - Get latest events")
    print("   - GET /api/events/new?since_id=X - Get new events since ID")
    print("   - GET /api/stats - Get statistics")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
