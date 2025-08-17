# 🔥 FCM Firebase Setup Guide - Quick Start

## 📋 Overview
Hướng dẫn setup Firebase Cloud Messaging (FCM) cho Healthcare Emergency Alerts trong dự án Vision Edge.

## 🚀 Quick Setup (5 phút)

### 1. Tạo Firebase Project
```
1. Đi tới: https://console.firebase.google.com/
2. Click "Create a project"
3. Project name: healthcare-vision-edge (hoặc tên bạn muốn)
4. Disable Google Analytics (optional)
5. Click "Create project"
```

### 2. Enable Cloud Messaging
```
1. Firebase Console → Project Settings (⚙️ icon)
2. Tab "Cloud Messaging" 
3. Copy "Server key" và "Sender ID" (để sau)
```

### 3. Tạo Service Account
```
1. Project Settings → Service Accounts tab
2. Click "Generate new private key"
3. Download file JSON
4. Đổi tên thành: firebase-adminsdk.json
5. Đặt vào: src/config/firebase-adminsdk.json
```

### 4. Cập nhật .env
Thêm vào file `.env`:
```env
# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id-here
FIREBASE_CREDENTIALS_PATH=src/config/firebase-adminsdk.json
FIREBASE_WEB_API_KEY=your-web-api-key
FIREBASE_MESSAGING_SENDER_ID=your-sender-id

# FCM Settings
FCM_DEFAULT_TOPIC=emergency_alerts
FCM_EMERGENCY_SOUND=emergency.wav
FCM_WARNING_SOUND=warning.wav
```

### 5. Test Setup
```bash
# Kiểm tra config
python scripts/firebase_setup_checker.py

# Test FCM notifications
python examples/fcm_demo.py
```

## 📱 Lấy FCM Tokens từ Mobile App

### Android (Kotlin/Java)
```kotlin
FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
    if (!task.isSuccessful) return@addOnCompleteListener
    
    val token = task.result
    Log.d("FCM", "Token: $token")
    // Send token to your server
}
```

### iOS (Swift)
```swift
Messaging.messaging().token { token, error in
  if let error = error {
    print("Error fetching FCM token: \(error)")
  } else if let token = token {
    print("FCM token: \(token)")
    // Send token to your server
  }
}
```

### Flutter
```dart
String? token = await FirebaseMessaging.instance.getToken();
print("FCM Token: $token");
```

## 🎯 Integration trong Code

### Update Healthcare Pipeline
```python
# src/main.py
user_fcm_tokens = [
    "actual_fcm_token_from_mobile_app_1",
    "actual_fcm_token_from_mobile_app_2"
]

pipeline = AdvancedHealthcarePipeline(
    camera, video_processor, fall_detector, 
    seizure_detector, seizure_predictor, 
    alerts_folder, user_fcm_tokens  # ← FCM tokens
)
```

## 🧪 Testing

### 1. Test với Mock Notifications
```bash
python examples/fcm_demo.py
```

### 2. Test với Real Firebase
```bash
# Đảm bảo firebase-adminsdk.json đã setup
python examples/test_fcm_notification.py
```

### 3. Test End-to-End
```bash
# Chạy healthcare monitor với FCM
python src/main.py
```

## 🔧 Troubleshooting

### Firebase not initialized
```
❌ Error: FCM not initialized
✅ Solution: Kiểm tra firebase-adminsdk.json và .env config
```

### Invalid project ID
```
❌ Error: Project ID mismatch
✅ Solution: Đảm bảo FIREBASE_PROJECT_ID trong .env match với JSON file
```

### No FCM tokens
```
❌ Warning: No FCM tokens provided
✅ Solution: Lấy tokens từ mobile app và update pipeline
```

## 📁 File Structure
```
vision_edge-v0.1/
├── .env                           # Environment variables
├── src/
│   ├── config/
│   │   ├── firebase-adminsdk.json # Firebase credentials (secret)
│   │   └── firebase_setup_guide.md
│   └── service/
│       └── fcm_notification_service.py
├── scripts/
│   └── firebase_setup_checker.py
└── examples/
    ├── fcm_demo.py
    └── test_fcm_notification.py
```

## 🔒 Security Notes

- ✅ `firebase-adminsdk.json` đã được thêm vào `.gitignore`
- ✅ Sensitive data trong `.env` (không commit)
- ✅ Sử dụng environment variables thay vì hardcode
- ⚠️ Không share Firebase credentials public

## 🎯 Next Steps

1. **Mobile App Integration**: Integrate Firebase SDK vào mobile app
2. **Token Management**: Implement token registration/update system
3. **Notification Channels**: Configure emergency/warning channels
4. **Testing**: Test với real devices và notifications
5. **Production**: Deploy với proper security measures

## 💡 Tips

- Test với multiple devices để đảm bảo delivery
- Configure notification sounds khác nhau cho fall vs seizure
- Implement notification acknowledgment system
- Monitor FCM delivery statistics
- Set up notification scheduling nếu cần
