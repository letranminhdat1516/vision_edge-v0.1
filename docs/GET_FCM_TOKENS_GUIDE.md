# 📱 How to Get Real FCM Tokens - Step by Step Guide

## 🎯 Overview
Để có notifications thật, bạn cần lấy FCM registration tokens từ mobile apps. Đây là hướng dẫn chi tiết:

## 📱 Android App (Kotlin/Java)

### 1. Add Firebase to Android Project
```gradle
// app/build.gradle
implementation 'com.google.firebase:firebase-messaging:23.2.1'
implementation 'com.google.firebase:firebase-analytics:21.3.0'
```

### 2. Add google-services.json
- Download từ Firebase Console → Project Settings → General → Your apps
- Đặt vào `app/` folder

### 3. Get FCM Token (Kotlin)
```kotlin
import com.google.firebase.messaging.FirebaseMessaging

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Get FCM token
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                Log.w("FCM", "Fetching FCM registration token failed", task.exception)
                return@addOnCompleteListener
            }

            // Get new FCM registration token
            val token = task.result
            Log.d("FCM", "FCM Registration Token: $token")
            
            // TODO: Send token to your server
            sendTokenToServer(token)
        }
    }
    
    private fun sendTokenToServer(token: String) {
        // Copy token này và paste vào .env file
        println("🔥 FCM Token: $token")
        
        // Hoặc gửi lên server để lưu
        // API call to save token...
    }
}
```

### 4. Handle Token Refresh
```kotlin
class MyFirebaseMessagingService : FirebaseMessagingService() {
    override fun onNewToken(token: String) {
        Log.d("FCM", "Refreshed token: $token")
        
        // Send the new token to your server
        sendTokenToServer(token)
    }
}
```

## 📱 iOS App (Swift)

### 1. Add Firebase to iOS Project
```swift
// Add Firebase SDK via CocoaPods
pod 'Firebase/Messaging'
pod 'Firebase/Analytics'
```

### 2. Get FCM Token
```swift
import Firebase
import FirebaseMessaging

class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, 
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        
        FirebaseApp.configure()
        
        // Get FCM token
        Messaging.messaging().token { token, error in
            if let error = error {
                print("Error fetching FCM registration token: \(error)")
            } else if let token = token {
                print("🔥 FCM registration token: \(token)")
                
                // TODO: Send token to your server
                self.sendTokenToServer(token)
            }
        }
        
        return true
    }
    
    func sendTokenToServer(_ token: String) {
        // Copy token này và paste vào .env file
        print("📱 iOS FCM Token: \(token)")
    }
}
```

## 📱 Flutter App

### 1. Add Firebase to Flutter
```yaml
# pubspec.yaml
dependencies:
  firebase_core: ^2.15.1
  firebase_messaging: ^14.6.7
```

### 2. Get FCM Token
```dart
import 'package:firebase_messaging/firebase_messaging.dart';

class FCMTokenService {
  static Future<void> initializeFCM() async {
    // Get FCM token
    String? token = await FirebaseMessaging.instance.getToken();
    
    if (token != null) {
      print('🔥 FCM Token: $token');
      
      // TODO: Send to your server
      await sendTokenToServer(token);
    }
    
    // Listen for token refresh
    FirebaseMessaging.instance.onTokenRefresh.listen((newToken) {
      print('🔄 FCM Token refreshed: $newToken');
      sendTokenToServer(newToken);
    });
  }
  
  static Future<void> sendTokenToServer(String token) async {
    // Copy token này và paste vào .env file
    print('📱 Flutter FCM Token: $token');
    
    // Hoặc gửi API call
    // await api.saveToken(token);
  }
}
```

## 🔧 Quick Test Methods

### Method 1: Log Token to Console
Thêm code trên vào app → Run → Copy token từ console logs

### Method 2: Display Token in App
```kotlin
// Android
TextView tokenTextView = findViewById(R.id.tokenTextView);
tokenTextView.setText(token);
```

### Method 3: Send via API
```kotlin
// Gửi token lên test server
val retrofit = Retrofit.Builder()
    .baseUrl("https://your-test-server.com/")
    .build()

api.saveToken(token)
```

## 🔄 Update .env File

Sau khi có tokens, update `.env`:
```env
# Real FCM Tokens from mobile apps
FCM_DEVICE_TOKENS=dxxxxx:APA91bGxxx...,exxxxx:APA91bHxxx...,fxxxxx:APA91bIxxx...
FCM_CAREGIVER_TOKENS=gxxxxx:APA91bJxxx...,hxxxxx:APA91bKxxx...
FCM_EMERGENCY_TOKENS=ixxxxx:APA91bLxxx...,jxxxxx:APA91bMxxx...
```

## ✅ Verify Tokens Work

### 1. Test với script
```bash
cd vision_edge-v0.1
python examples/test_realtime_fcm.py
```

### 2. Check mobile for notifications
- Notifications should appear trên mobile devices
- Check notification sounds và vibrations
- Verify notification content

## 🚨 Common Token Formats

### Android FCM Token
```
dxxxxxxxxxxxxxx:APA91bGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### iOS FCM Token
```
exxxxxxxxxxxxxx:APA91bHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 🔒 Security Notes

- ✅ FCM tokens không phải secret credentials
- ✅ Safe to log trong development
- ⚠️ Tokens có thể expire → implement refresh logic
- 🔄 Tokens change khi app reinstall

## 📊 Test Delivery

### 1. Immediate Test
Chạy test script → check mobile ngay

### 2. Healthcare Simulation
Trigger fall/seizure detection → verify notifications

### 3. Multiple Devices
Test với nhiều devices cùng lúc

## 💡 Pro Tips

1. **Token Collection**: Tạo simple API để collect tokens từ multiple devices
2. **Token Management**: Implement database để track active tokens
3. **Testing**: Use Firebase Console để test manual notifications
4. **Monitoring**: Set up FCM delivery statistics
5. **Backup**: Keep multiple tokens per user for redundancy

## 🎯 Next Steps After Getting Tokens

1. Update `.env` với real tokens
2. Run `python examples/test_realtime_fcm.py`
3. Verify notifications trên mobile
4. Test end-to-end với healthcare detection
5. Monitor và optimize delivery rates
