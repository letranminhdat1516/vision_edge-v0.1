# Priority-Based Alert System Implementation

## 🎯 **Tổng quan hệ thống mới**

Hệ thống alert mới được thiết kế theo nguyên tắc **ưu tiên mức độ nguy hiểm** thay vì thời gian, đảm bảo các alert quan trọng không bị lấn át bởi các alert mới có mức độ thấp hơn.

---

## 🏗️ **Kiến trúc Implementation**

### **1. AlertStateManager (`alert_state_manager.py`)**
- **Chức năng chính**: Quản lý trạng thái alert theo priority
- **Priority Mapping**: Severity + Status → Priority Level (0-5)
- **Smart Alert Creation**: Chỉ tạo alert nếu priority cao hơn alert hiện tại
- **Database Integration**: Tương tác trực tiếp với PostgreSQL

### **2. Updated HealthcareEventPublisher (`healthcare_event_publisher_v2.py`)**
- **Tích hợp AlertStateManager**: Tự động kiểm tra priority trước khi tạo alert
- **Conditional Notifications**: Chỉ gửi notification khi cần thiết
- **Enhanced Response**: Thêm field `alert_created` trong response

### **3. Alert Management API (`alert_management_api.py`)**
- **Priority-ordered Endpoints**: Lấy alerts theo thứ tự priority
- **Action Endpoints**: Acknowledge và Resolve alerts
- **Statistics**: Dashboard metrics
- **Bulk Operations**: Xử lý nhiều alerts cùng lúc

---

## 📊 **Priority Level System**

```
🚨 Level 5: Critical + Active     (Seizure ≥75% or Fall ≥80%)
⚠️ Level 4: High + Active         (Seizure 50-74% or Fall 60-79%)  
ℹ️ Level 3: Medium + Active       (Seizure 30-49% or Fall 40-59%)
🔸 Level 2: Critical + Acknowledged / Low + Active
✅ Level 1: Any + Acknowledged
💤 Level 0: Any + Resolved
```

### **Priority Rules:**
1. **Chỉ tạo alert mới** nếu priority ≥ highest current priority
2. **Critical/High events** có thể override lower priority events
3. **Acknowledging** giảm priority xuống 2-3 levels
4. **Resolving** loại bỏ khỏi active list (priority = 0)

---

## 🔄 **Event Processing Workflow**

### **1. Event Detection**
```python
# Fall/Seizure detected với confidence score
confidence = 0.85
event_type = "seizure"

# Determine mobile status
status = "danger"  # for mobile display

# Determine database severity  
severity = "critical"  # for alert priority
```

### **2. Priority Check**
```python
# Check existing alerts
current_alerts = get_active_alerts_by_priority(user_id)
current_max_priority = max(alert.priority_level for alert in current_alerts)

# Calculate new event priority
new_priority = calculate_priority(severity, "active")

# Only create if higher or equal priority
if new_priority >= current_max_priority:
    create_alert(event_data)
else:
    skip_alert_creation()
```

### **3. Mobile Notification Logic**
```python
# Send notification only if:
# 1. Alert was created (priority-based)
# 2. OR confidence is very high (≥70% for fall, ≥60% for seizure)

if alert_created or confidence >= threshold:
    send_mobile_notification(response)
```

---

## 📱 **API Endpoints**

### **Alert List (Priority-Ordered)**
```
GET /api/alerts/list/<user_id>
Response: {
  "alerts": [
    {
      "alert_id": "uuid",
      "event_type": "seizure",
      "severity": "critical",
      "status": "danger",           // mobile status
      "alert_status": "active",     // db status  
      "priority_level": 5,
      "acknowledged": false,
      "message": "🚨 CRITICAL: Seizure detected...",
      "confidence": 0.85,
      "imageUrl": "https://...",
      "detected_at": "2025-08-17T10:30:00Z"
    }
  ],
  "summary": {
    "total_active": 3,
    "critical_active": 1,
    "highest_priority": 5
  },
  "sorted_by": "priority_desc"
}
```

### **Alert Actions**
```
POST /api/alerts/acknowledge
Body: {"alert_id": "uuid", "user_id": "uuid"}

POST /api/alerts/resolve  
Body: {"alert_id": "uuid", "user_id": "uuid", "notes": "Resolved manually"}

POST /api/alerts/bulk-acknowledge
Body: {"alert_ids": ["uuid1", "uuid2"], "user_id": "uuid"}
```

### **Statistics**
```
GET /api/alerts/statistics/<user_id>
Response: {
  "statistics": {
    "total_alerts_24h": 15,
    "active_critical": 2, 
    "active_high": 1,
    "acknowledged_count": 8,
    "resolved_count": 4
  }
}
```

---

## 🧪 **Testing & Validation**

### **Test Script**: `test_priority_alerts.py`
- **Scenario Testing**: Various confidence levels and event types
- **Priority Validation**: Kiểm tra logic ưu tiên  
- **Workflow Testing**: Acknowledge/Resolve actions
- **Statistics Verification**: Dashboard metrics

### **Test Scenarios:**
1. **Low Fall (45%)** → No alert (below threshold)
2. **Medium Fall (65%)** → Create alert (medium priority)
3. **High Seizure (80%)** → Create alert (critical priority) 
4. **Low Seizure (35%)** → Skip alert (lower than existing critical)
5. **Critical Fall (90%)** → Create alert (same critical level)

---

## ✅ **Benefits của hệ thống mới**

### **🎯 Healthcare-focused:**
- **Không spam notifications** với low-priority events
- **Critical events luôn được ưu tiên** 
- **Persistent dangerous state** cho đến khi được xử lý
- **Smart escalation** dựa trên medical severity

### **📊 Database-optimized:**
- **Tận dụng schema hiện tại** (`severity`, `status` enums)
- **No schema changes required**
- **Efficient priority queries** với SQL CASE statements
- **Proper audit trail** với acknowledged/resolved workflow

### **📱 Mobile-friendly:**
- **Priority-ordered alert list** thay vì chronological
- **Conditional notifications** giảm notification spam
- **Clear action workflow**: View → Acknowledge → Resolve
- **Real-time statistics** cho dashboard

---

## 🚀 **Deployment Steps**

1. **Deploy AlertStateManager**: Core priority logic
2. **Update HealthcareEventPublisher**: Integrate priority checking
3. **Deploy Alert Management API**: REST endpoints for mobile
4. **Test Priority Logic**: Run test scenarios
5. **Update Mobile App**: Consume new API endpoints

---

## 📋 **Configuration Options**

### **Confidence Thresholds (tunable):**
```python
SEVERITY_THRESHOLDS = {
    'seizure': {'critical': 0.75, 'high': 0.50, 'medium': 0.30},
    'fall': {'critical': 0.80, 'high': 0.60, 'medium': 0.40}
}
```

### **Notification Thresholds:**
```python
NOTIFICATION_THRESHOLDS = {
    'fall': 0.70,      # Send notification if ≥70% confidence
    'seizure': 0.60    # Send notification if ≥60% confidence  
}
```

### **Auto-cleanup:**
```python
AUTO_RESOLVE_HOURS = 24  # Auto-resolve acknowledged alerts after 24h
```

---

## 🎯 **Next Steps**

1. **Integration Testing**: Test với real camera stream
2. **Performance Tuning**: Optimize database queries
3. **Mobile App Updates**: Implement new API endpoints
4. **Dashboard Metrics**: Real-time statistics display
5. **Alert Rules Engine**: Configurable thresholds per user/room

Hệ thống mới đảm bảo **healthcare alerts được ưu tiên đúng mức độ nguy hiểm**, cải thiện đáng kể trải nghiệm người dùng và hiệu quả phản ứng y tế!
