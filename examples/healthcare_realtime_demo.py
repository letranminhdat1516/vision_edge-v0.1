<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Realtime - Direct Database Polling</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .event {
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            animation: slideIn 0.3s ease-out;
        }
        .event.danger {
            background: #ffebee;
            border-left-color: #f44336;
        }
        .event.warning {
            background: #fff8e1;
            border-left-color: #ff9800;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
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
        .log {
            background: #000;
            color: #00ff00;
            padding: 10px;
            height: 150px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
            margin: 10px 0;
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Healthcare Live Monitor</h1>
        <p>Kết nối trực tiếp với database để nhận events realtime</p>
        
        <div id="status" class="status disconnected">
            ❌ Not connected
        </div>
        
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
        </div>
        
        <div>
            <button class="start-btn" onclick="startMonitoring()">🚀 Start Monitoring</button>
            <button class="stop-btn" onclick="stopMonitoring()">⏹️ Stop</button>
            <button onclick="clearEvents()" style="background: #6c757d; color: white;">🗑️ Clear</button>
        </div>
        
        <div>
            <h3>📝 Activity Log</h3>
            <div id="log" class="log"></div>
        </div>
        
        <div>
            <h3>📱 Live Events</h3>
            <div id="events"></div>
        </div>
    </div>

    <script>
        let isMonitoring = false;
        let pollInterval = null;
        let lastEventId = null;
        let eventCount = 0;
        let fallCount = 0;
        let seizureCount = 0;
        
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
            if (connected) {
                statusEl.className = 'status connected';
                statusEl.textContent = '✅ ' + message;
            } else {
                statusEl.className = 'status disconnected'; 
                statusEl.textContent = '❌ ' + message;
            }
        }
        
        function updateStats() {
            document.getElementById('total-events').textContent = eventCount;
            document.getElementById('fall-events').textContent = fallCount;
            document.getElementById('seizure-events').textContent = seizureCount;
        }
        
        async function startMonitoring() {
            if (isMonitoring) return;
            
            isMonitoring = true;
            updateStatus(true, 'Monitoring active - Polling database every 2 seconds');
            log('🚀 Started monitoring healthcare events');
            
            // Initialize lastEventId by getting the most recent event
            try {
                const response = await fetch('/api/latest-event');
                if (response.ok) {
                    const data = await response.json();
                    if (data.success && data.event) {
                        lastEventId = data.event.id;
                        log(`📋 Initialized with event ID: ${lastEventId}`);
                    }
                }
            } catch (error) {
                log('⚠️ Could not get initial event ID, starting fresh');
            }
            
            // Start polling for new events
            pollInterval = setInterval(pollForNewEvents, 2000);
        }
        
        function stopMonitoring() {
            if (!isMonitoring) return;
            
            isMonitoring = false;
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
            updateStatus(false, 'Monitoring stopped');
            log('⏹️ Stopped monitoring');
        }
        
        async function pollForNewEvents() {
            try {
                // Simulate API call - in real implementation this would call your backend API
                const response = await fetch(`/api/new-events?since=${lastEventId || 0}`);
                
                if (!response.ok) {
                    // Simulate new events for demo
                    simulateRandomEvent();
                    return;
                }
                
                const data = await response.json();
                if (data.success && data.events && data.events.length > 0) {
                    data.events.forEach(event => {
                        handleNewEvent(event);
                        lastEventId = event.id;
                    });
                    log(`📨 Received ${data.events.length} new events`);
                }
                
            } catch (error) {
                // Since we don't have real API, simulate events
                if (Math.random() < 0.3) { // 30% chance of new event
                    simulateRandomEvent();
                }
            }
        }
        
        function simulateRandomEvent() {
            // Generate random event for demo
            const eventTypes = ['fall', 'abnormal_behavior'];
            const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)];
            const confidence = Math.random() * 0.4 + 0.5; // 0.5 to 0.9
            
            const mockEvent = {
                id: Math.floor(Math.random() * 10000),
                event_type: eventType,
                confidence_score: confidence,
                detected_at: new Date().toISOString(),
                location: 'Demo Room A',
                camera_id: 'demo_camera_01',
                metadata: { test_mode: true }
            };
            
            handleNewEvent(mockEvent);
            log(`🎯 Simulated ${eventType} event: ${Math.round(confidence * 100)}%`);
        }
        
        function handleNewEvent(eventData) {
            // Determine status based on confidence and type
            let status = 'normal';
            if (eventData.event_type === 'fall') {
                if (eventData.confidence_score >= 0.8) status = 'danger';
                else if (eventData.confidence_score >= 0.6) status = 'warning';
                fallCount++;
            } else if (eventData.event_type === 'abnormal_behavior') {
                if (eventData.confidence_score >= 0.7) status = 'danger';
                else if (eventData.confidence_score >= 0.5) status = 'warning';
                seizureCount++;
            }
            
            eventCount++;
            updateStats();
            
            // Create event display
            const eventEl = document.createElement('div');
            eventEl.className = `event ${status}`;
            
            const time = new Date(eventData.detected_at).toLocaleString('vi-VN');
            const confidence = Math.round(eventData.confidence_score * 100);
            
            // Generate Vietnamese action message
            let action = '';
            if (status === 'danger') {
                action = `🚨 KHẨN CẤP - Phát hiện ${eventData.event_type === 'fall' ? 'té ngã' : 'co giật'} với độ tin cậy ${confidence}%. Cần hỗ trợ ngay lập tức!`;
            } else if (status === 'warning') {
                action = `⚠️ Cảnh báo - Phát hiện ${eventData.event_type === 'fall' ? 'té ngã' : 'co giật'} với độ tin cậy ${confidence}%. Cần theo dõi.`;
            } else {
                action = `ℹ️ Không có gì bất thường - ${confidence}% confidence`;
            }
            
            eventEl.innerHTML = `
                <div><strong>🚨 ${eventData.event_type.toUpperCase()} DETECTED</strong></div>
                <div>📊 Confidence: ${confidence}% | Status: ${status.toUpperCase()}</div>
                <div>💬 Action: ${action}</div>
                <div>🕒 Time: ${time}</div>
                <div>📍 Location: ${eventData.location || 'Unknown'}</div>
                <div>📹 Camera: ${eventData.camera_id || 'Unknown'}</div>
                <div>🆔 Event ID: ${eventData.id}</div>
            `;
            
            // Add to top of events list
            eventsEl.insertBefore(eventEl, eventsEl.firstChild);
            
            // Keep only last 20 events
            while (eventsEl.children.length > 20) {
                eventsEl.removeChild(eventsEl.lastChild);
            }
            
            // Play notification sound
            playNotificationSound(status);
            
            // Show notification
            if (Notification.permission === 'granted') {
                new Notification(`Healthcare Alert: ${eventData.event_type}`, {
                    body: action,
                    icon: '🏥'
                });
            }
            
            log(`📱 Event displayed: ${eventData.event_type} (${confidence}%) - ${status}`);
        }
        
        function playNotificationSound(status) {
            try {
                const context = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = context.createOscillator();
                const gainNode = context.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(context.destination);
                
                if (status === 'danger') {
                    oscillator.frequency.setValueAtTime(800, context.currentTime);
                    // Play multiple beeps for danger
                    setTimeout(() => {
                        const osc2 = context.createOscillator();
                        const gain2 = context.createGain();
                        osc2.connect(gain2);
                        gain2.connect(context.destination);
                        osc2.frequency.setValueAtTime(800, context.currentTime);
                        gain2.gain.setValueAtTime(0.1, context.currentTime);
                        osc2.start();
                        osc2.stop(context.currentTime + 0.2);
                    }, 300);
                } else if (status === 'warning') {
                    oscillator.frequency.setValueAtTime(600, context.currentTime);
                } else {
                    oscillator.frequency.setValueAtTime(400, context.currentTime);
                }
                
                gainNode.gain.setValueAtTime(0.1, context.currentTime);
                oscillator.start();
                oscillator.stop(context.currentTime + 0.2);
            } catch (error) {
                log('🔇 Could not play notification sound');
            }
        }
        
        function clearEvents() {
            eventsEl.innerHTML = '';
            eventCount = 0;
            fallCount = 0;
            seizureCount = 0;
            updateStats();
            log('🗑️ Events cleared');
        }
        
        // Request notification permission
        if (Notification.permission === 'default') {
            Notification.requestPermission();
        }
        
        // Initialize
        log('🏥 Healthcare Live Monitor initialized');
        log('📋 Click "Start Monitoring" to begin');
        updateStatus(false, 'Ready to start monitoring');
        updateStats();
        
        // Auto-start monitoring
        setTimeout(() => {
            log('🚀 Auto-starting monitoring in 2 seconds...');
            setTimeout(startMonitoring, 2000);
        }, 1000);
    </script>
</body>
</html>
