"""
Mobile Supabase Realtime Integration Guide
How to receive real-time healthcare notifications on mobile devices
"""

# FLUTTER/DART IMPLEMENTATION
flutter_dart_code = '''
// pubspec.yaml - Add these dependencies
dependencies:
  supabase_flutter: ^2.0.0
  flutter_local_notifications: ^16.0.0

// main.dart - Initialize Supabase
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class HealthcareMobileApp extends StatefulWidget {
  @override
  _HealthcareMobileAppState createState() => _HealthcareMobileAppState();
}

class _HealthcareMobileAppState extends State<HealthcareMobileApp> {
  late SupabaseClient supabase;
  late FlutterLocalNotificationsPlugin notifications;
  RealtimeChannel? healthcareChannel;
  
  @override
  void initState() {
    super.initState();
    initializeSupabase();
    setupNotifications();
    setupRealtimeListener();
  }
  
  Future<void> initializeSupabase() async {
    await Supabase.initialize(
      url: 'YOUR_SUPABASE_URL',
      anonKey: 'YOUR_SUPABASE_ANON_KEY',
    );
    supabase = Supabase.instance.client;
  }
  
  Future<void> setupNotifications() async {
    notifications = FlutterLocalNotificationsPlugin();
    
    const AndroidInitializationSettings androidSettings = 
      AndroidInitializationSettings('@mipmap/ic_launcher');
    const IOSInitializationSettings iosSettings = 
      IOSInitializationSettings();
    const InitializationSettings initSettings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );
    
    await notifications.initialize(initSettings);
  }
  
  void setupRealtimeListener() {
    // Listen to event_detections table
    healthcareChannel = supabase
      .channel('healthcare_events')
      .onPostgresChanges(
        event: PostgresChangeEvent.insert,
        schema: 'public',
        table: 'event_detections',
        callback: (payload) {
          handleHealthcareEvent(payload.newRecord);
        }
      )
      .subscribe();
    
    print('🔔 Supabase Realtime listener setup complete');
  }
  
  void handleHealthcareEvent(Map<String, dynamic> eventData) {
    print('📱 Received healthcare event: $eventData');
    
    // Extract event information
    String eventType = eventData['event_type'] ?? 'unknown';
    double confidence = (eventData['confidence_score'] as num?)?.toDouble() ?? 0.0;
    String description = eventData['event_description'] ?? '';
    DateTime detectedAt = DateTime.parse(eventData['detected_at'] ?? DateTime.now().toIso8601String());
    
    // Determine status based on confidence and event type
    String status = determineStatus(eventType, confidence);
    String action = generateActionMessage(eventType, confidence, status);
    String imageUrl = generateImageUrl(eventData['event_id']);
    
    // Create mobile notification object
    Map<String, dynamic> mobileNotification = {
      'imageUrl': imageUrl,
      'status': status,
      'action': action,
      'time': detectedAt.toIso8601String(),
      'eventType': eventType,
      'confidence': confidence
    };
    
    // Show local notification
    showHealthcareNotification(mobileNotification);
    
    // Update UI state
    setState(() {
      // Add to notification list, update UI, etc.
    });
  }
  
  String determineStatus(String eventType, double confidence) {
    if (eventType == 'fall') {
      if (confidence >= 0.8) return 'danger';
      if (confidence >= 0.6) return 'warning';
      return 'normal';
    } else if (eventType == 'abnormal_behavior') {
      if (confidence >= 0.7) return 'danger';
      if (confidence >= 0.5) return 'warning';  
      return 'normal';
    }
    return 'normal';
  }
  
  String generateActionMessage(String eventType, double confidence, String status) {
    if (status == 'normal') return 'Không có gì bất thường';
    
    if (status == 'warning') {
      if (eventType == 'fall') {
        return 'Phát hiện té (${(confidence * 100).toInt()}% confidence) - Cần theo dõi';
      } else if (eventType == 'abnormal_behavior') {
        return 'Phát hiện co giật (${(confidence * 100).toInt()}% confidence) - Cần theo dõi';
      }
    }
    
    if (status == 'danger') {
      if (eventType == 'fall') {
        return '⚠️ BÁO ĐỘNG NGUY HIỂM: Phát hiện té - Yêu cầu hỗ trợ gấp!';
      } else if (eventType == 'abnormal_behavior') {
        return '🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện co giật - Yêu cầu hỗ trợ gấp!';
      }
    }
    
    return 'Đang theo dõi...';
  }
  
  String generateImageUrl(String? eventId) {
    return 'https://healthcare-system.com/snapshots/${eventId ?? 'default'}.jpg';
  }
  
  Future<void> showHealthcareNotification(Map<String, dynamic> notification) async {
    String status = notification['status'];
    String action = notification['action'];
    
    // Configure notification based on status
    AndroidNotificationDetails androidDetails;
    
    if (status == 'danger') {
      androidDetails = AndroidNotificationDetails(
        'healthcare_danger',
        'Healthcare Emergency',
        channelDescription: 'Emergency healthcare alerts',
        importance: Importance.max,
        priority: Priority.high,
        color: Colors.red,
        sound: RawResourceAndroidNotificationSound('emergency_alert'),
        enableVibration: true,
        vibrationPattern: Int64List.fromList([0, 1000, 500, 1000]),
      );
    } else if (status == 'warning') {
      androidDetails = AndroidNotificationDetails(
        'healthcare_warning',
        'Healthcare Warning',
        channelDescription: 'Healthcare warning alerts',
        importance: Importance.high,
        priority: Priority.high,
        color: Colors.orange,
        enableVibration: true,
      );
    } else {
      androidDetails = AndroidNotificationDetails(
        'healthcare_normal',
        'Healthcare Info',
        channelDescription: 'Healthcare information',
        importance: Importance.defaultImportance,
        priority: Priority.defaultPriority,
        color: Colors.blue,
      );
    }
    
    const IOSNotificationDetails iosDetails = IOSNotificationDetails(
      sound: 'default.wav',
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );
    
    NotificationDetails notificationDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );
    
    await notifications.show(
      DateTime.now().millisecondsSinceEpoch.remainder(100000),
      status == 'danger' ? '🚨 EMERGENCY' : status == 'warning' ? '⚠️ WARNING' : 'ℹ️ INFO',
      action,
      notificationDetails,
      payload: jsonEncode(notification),
    );
  }
  
  @override
  void dispose() {
    healthcareChannel?.unsubscribe();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Healthcare Monitor')),
      body: Column(
        children: [
          // Your UI components here
          Text('Listening for healthcare events...'),
          // Display notifications list, status, etc.
        ],
      ),
    );
  }
}
'''

# REACT NATIVE IMPLEMENTATION  
react_native_code = '''
// package.json - Add these dependencies
{
  "dependencies": {
    "@supabase/supabase-js": "^2.38.0",
    "@react-native-async-storage/async-storage": "^1.19.0",
    "react-native-push-notification": "^8.1.1"
  }
}

// SupabaseConfig.js
import { createClient } from '@supabase/supabase-js';
import AsyncStorage from '@react-native-async-storage/async-storage';

const supabaseUrl = 'YOUR_SUPABASE_URL';
const supabaseAnonKey = 'YOUR_SUPABASE_ANON_KEY';

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
});

// HealthcareRealtimeService.js
import { supabase } from './SupabaseConfig';
import PushNotification from 'react-native-push-notification';

class HealthcareRealtimeService {
  constructor() {
    this.channel = null;
    this.setupPushNotifications();
  }
  
  setupPushNotifications() {
    PushNotification.configure({
      onNotification: function(notification) {
        console.log('📱 Notification received:', notification);
      },
      requestPermissions: true,
    });
  }
  
  startListening() {
    this.channel = supabase
      .channel('healthcare_realtime')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public', 
          table: 'event_detections'
        },
        (payload) => this.handleHealthcareEvent(payload.new)
      )
      .subscribe();
    
    console.log('🔔 Healthcare realtime listener started');
  }
  
  handleHealthcareEvent(eventData) {
    console.log('📥 Healthcare event received:', eventData);
    
    // Extract and process event data
    const eventType = eventData.event_type || 'unknown';
    const confidence = parseFloat(eventData.confidence_score || 0);
    const description = eventData.event_description || '';
    const detectedAt = new Date(eventData.detected_at || Date.now());
    
    // Determine status and action
    const status = this.determineStatus(eventType, confidence);
    const action = this.generateActionMessage(eventType, confidence, status);
    const imageUrl = `https://healthcare-system.com/snapshots/${eventData.event_id || 'default'}.jpg`;
    
    // Create mobile notification
    const mobileNotification = {
      imageUrl,
      status,
      action,
      time: detectedAt.toISOString(),
      eventType,
      confidence
    };
    
    // Show push notification
    this.showPushNotification(mobileNotification);
    
    // Trigger app state update
    this.onEventReceived?.(mobileNotification);
  }
  
  determineStatus(eventType, confidence) {
    if (eventType === 'fall') {
      if (confidence >= 0.8) return 'danger';
      if (confidence >= 0.6) return 'warning';
      return 'normal';
    } else if (eventType === 'abnormal_behavior') {
      if (confidence >= 0.7) return 'danger';
      if (confidence >= 0.5) return 'warning';
      return 'normal';
    }
    return 'normal';
  }
  
  generateActionMessage(eventType, confidence, status) {
    if (status === 'normal') return 'Không có gì bất thường';
    
    if (status === 'warning') {
      if (eventType === 'fall') {
        return `Phát hiện té (${Math.round(confidence * 100)}% confidence) - Cần theo dõi`;
      } else if (eventType === 'abnormal_behavior') {
        return `Phát hiện co giật (${Math.round(confidence * 100)}% confidence) - Cần theo dõi`;
      }
    }
    
    if (status === 'danger') {
      if (eventType === 'fall') {
        return '⚠️ BÁO ĐỘNG NGUY HIỂM: Phát hiện té - Yêu cầu hỗ trợ gấp!';
      } else if (eventType === 'abnormal_behavior') {
        return '🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện co giật - Yêu cầu hỗ trợ gấp!';
      }
    }
    
    return 'Đang theo dõi...';
  }
  
  showPushNotification(notification) {
    const { status, action } = notification;
    
    let title = 'ℹ️ Healthcare Info';
    let priority = 'default';
    
    if (status === 'danger') {
      title = '🚨 EMERGENCY ALERT';
      priority = 'high';
    } else if (status === 'warning') {
      title = '⚠️ WARNING';
      priority = 'high';
    }
    
    PushNotification.localNotification({
      title,
      message: action,
      priority,
      vibrate: status === 'danger',
      playSound: true,
      soundName: status === 'danger' ? 'emergency.mp3' : 'default',
      userInfo: notification,
    });
  }
  
  stopListening() {
    if (this.channel) {
      supabase.removeChannel(this.channel);
      this.channel = null;
    }
    console.log('🔇 Healthcare realtime listener stopped');
  }
}

export default new HealthcareRealtimeService();

// App.js - Usage
import React, { useEffect, useState } from 'react';
import { View, Text, FlatList } from 'react-native';
import HealthcareRealtimeService from './HealthcareRealtimeService';

export default function App() {
  const [notifications, setNotifications] = useState([]);
  
  useEffect(() => {
    // Setup event handler
    HealthcareRealtimeService.onEventReceived = (notification) => {
      setNotifications(prev => [notification, ...prev]);
    };
    
    // Start listening
    HealthcareRealtimeService.startListening();
    
    return () => {
      HealthcareRealtimeService.stopListening();
    };
  }, []);
  
  return (
    <View style={{ flex: 1, padding: 20 }}>
      <Text style={{ fontSize: 18, fontWeight: 'bold' }}>
        Healthcare Monitor
      </Text>
      
      <FlatList
        data={notifications}
        keyExtractor={(item, index) => index.toString()}
        renderItem={({ item }) => (
          <View style={{ 
            padding: 15, 
            margin: 5, 
            backgroundColor: item.status === 'danger' ? '#ffebee' : 
                           item.status === 'warning' ? '#fff8e1' : '#e3f2fd',
            borderRadius: 8 
          }}>
            <Text style={{ fontWeight: 'bold' }}>
              Status: {item.status.toUpperCase()}
            </Text>
            <Text>{item.action}</Text>
            <Text style={{ fontSize: 12, color: '#666' }}>
              {new Date(item.time).toLocaleString()}
            </Text>
          </View>
        )}
      />
    </View>
  );
}
'''

print("📱 MOBILE SUPABASE REALTIME INTEGRATION GUIDE")
print("=" * 60)
print("\n🔧 IMPLEMENTATION OPTIONS:")
print("\n1. FLUTTER/DART:")
print(flutter_dart_code)
print("\n2. REACT NATIVE:")
print(react_native_code)
