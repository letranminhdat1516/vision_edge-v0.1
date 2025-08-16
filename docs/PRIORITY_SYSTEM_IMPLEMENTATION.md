# Priority-Based Alert System - Implementation Summary

## 🎯 **Phân tích về API Endpoints vs Supabase Realtime**

### **❌ Tại sao KHÔNG cần API endpoints:**

#### **1. Supabase Realtime đã đủ mạnh:**
```javascript
// Mobile app subscribe trực tiếp vào database changes
const channel = supabase
  .channel('healthcare_alerts')
  .on('postgres_changes', {
    event: 'INSERT',
    schema: 'public',
    table: 'alerts'
  }, (payload) => {
    // Receive alerts ordered by priority trong database
    handleNewAlert(payload.new);
  })
  .subscribe();
```

#### **2. Database Query với Priority:**
```sql
-- Mobile app query alerts với priority order trực tiếp
SELECT a.*, 
       CASE a.severity
           WHEN 'critical' THEN 5
           WHEN 'high' THEN 4
           WHEN 'medium' THEN 3
           WHEN 'low' THEN 2
           ELSE 1
       END as priority_level
FROM alerts a
WHERE a.user_id = $1 AND a.status = 'active'
ORDER BY priority_level DESC, a.created_at DESC
```

#### **3. Acknowledge/Resolve trực tiếp qua Supabase:**
```javascript
// Mobile app update alert status trực tiếp
await supabase
  .from('alerts')
  .update({ 
    status: 'acknowledged',
    acknowledged_by: user_id,
    acknowledged_at: new Date().toISOString()
  })
  .eq('alert_id', alert_id);
```

### **✅ Ưu điểm của Supabase Realtime:**

1. **Real-time by default** - Không cần polling API
2. **Automatic subscriptions** - Changes được push ngay lập tức  
3. **Reduce server load** - Không cần maintain REST endpoints
4. **Simplified architecture** - Ít components cần maintain
5. **Built-in authentication** - Supabase handle auth automatically

---

## 🔧 **Priority System Implementation**

### **1. Healthcare Event Publisher Enhanced:**

```python
class HealthcareEventPublisher:
    # Confidence thresholds cho severity mapping
    SEVERITY_THRESHOLDS = {
        'fall': {'critical': 0.80, 'high': 0.60, 'medium': 0.40},
        'seizure': {'critical': 0.75, 'high': 0.50, 'medium': 0.30}
    }
    
    def _should_create_alert(self, confidence: float, event_type: str, user_id: str):
        """Priority-based alert creation logic"""
        # Map confidence → severity → priority level
        severity = self._map_confidence_to_severity(confidence, event_type)
        new_priority = self._calculate_priority_level(severity, 'active')
        
        # Check existing highest priority alert
        highest_alert = self._get_highest_priority_alert(user_id)
        
        # Only create if higher or equal priority
        return new_priority >= current_max_priority
```

### **2. Smart Alert Creation Logic:**

```python
def publish_fall_detection(self, confidence, bounding_boxes, context, ...):
    # Always create event_detection (audit trail)
    event_data = {...}
    event_id = self.postgresql_service.publish_event_detection(event_data)
    
    # Priority check for alert creation
    should_create_alert, severity = self._should_create_alert(confidence, 'fall', user_id)
    
    if should_create_alert:
        # Create alert only if priority check passed
        alert_data = {...}
        self.postgresql_service.publish_alert(alert_data)
    
    # Mobile notification logic
    should_notify = (
        should_create_alert or  # Alert was created
        confidence >= NOTIFICATION_THRESHOLDS['fall']  # High confidence
    )
    
    if should_notify:
        send_mobile_notification(response)
```

### **3. Mobile Response Format Enhanced:**

```python
response = {
    "imageUrl": "https://...",
    "status": "danger",           # Mobile status: normal/warning/danger
    "action": "🚨 Fall detected...",
    "time": "2025-08-17T10:30:00Z",
    "alert_created": True,        # NEW: Was alert created?
    "severity": "critical",       # NEW: Database severity  
    "priority_level": 5          # NEW: Priority for sorting
}
```

---

## 📱 **Mobile App Integration Guide**

### **1. Subscribe to Real-time Events:**

```javascript
class HealthcareRealtimeService {
  setupAlertSubscription() {
    this.alertChannel = supabase
      .channel('healthcare_alerts')
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'alerts',
        filter: `user_id=eq.${this.userId}`
      }, (payload) => {
        const alert = payload.new;
        this.handleNewAlert(alert);
        
        // Show push notification based on severity
        if (alert.severity === 'critical') {
          this.showCriticalNotification(alert);
        }
      })
      .subscribe();
  }
}
```

### **2. Query Alerts với Priority Order:**

```javascript
async getActiveAlerts() {
  const { data: alerts } = await supabase
    .from('alerts')
    .select(`
      *,
      event_detections(confidence_score, detection_data)
    `)
    .eq('user_id', this.userId)
    .eq('status', 'active')
    .order('created_at', { ascending: false });
    
  // Sort by priority level trên client
  return alerts.sort((a, b) => {
    const priorityA = this.mapSeverityToPriority(a.severity);
    const priorityB = this.mapSeverityToPriority(b.severity);
    return priorityB - priorityA;  // Descending order
  });
}
```

### **3. Acknowledge/Resolve Actions:**

```javascript
async acknowledgeAlert(alertId) {
  const { error } = await supabase
    .from('alerts')
    .update({
      status: 'acknowledged',
      acknowledged_by: this.userId,
      acknowledged_at: new Date().toISOString()
    })
    .eq('alert_id', alertId);
    
  if (!error) {
    // This triggers realtime update to all subscribers
    this.refreshAlertList();
  }
}

async resolveAlert(alertId, notes) {
  const { error } = await supabase
    .from('alerts')
    .update({
      status: 'resolved',
      resolved_at: new Date().toISOString(),
      resolution_notes: notes
    })
    .eq('alert_id', alertId);
}
```

---

## 🎯 **Priority Mapping Logic**

### **Confidence → Severity → Priority:**

```
Fall Detection:
├── ≥80%: critical (Priority 5) → "danger" status
├── ≥60%: high (Priority 4) → "danger" status  
├── ≥40%: medium (Priority 3) → "warning" status
└── <40%: low (Priority 2) → "normal" status

Seizure Detection:
├── ≥75%: critical (Priority 5) → "danger" status
├── ≥50%: high (Priority 4) → "danger" status
├── ≥30%: medium (Priority 3) → "warning" status
└── <30%: low (Priority 2) → "normal" status
```

### **Alert Status Transitions:**

```
active → acknowledged: Priority - 2 levels
acknowledged → resolved: Priority = 0 (removed from active list)
```

### **Alert Creation Rules:**

```python
# Only create alert if:
new_priority >= current_highest_priority

# Exception: Always notify if confidence very high
if confidence >= NOTIFICATION_THRESHOLDS[event_type]:
    send_mobile_notification(even_if_no_alert_created)
```

---

## 📊 **Database Schema Leverage**

### **Existing Enums Used:**
```sql
-- severity_enum: low, medium, high, critical  
-- alert_status_enum: active, acknowledged, resolved
-- event_status_enum: detected, verified, dismissed
```

### **Priority Query Examples:**
```sql
-- Get alerts ordered by priority
SELECT *,
  CASE severity
    WHEN 'critical' THEN 5
    WHEN 'high' THEN 4  
    WHEN 'medium' THEN 3
    WHEN 'low' THEN 2
    ELSE 1
  END as priority_level
FROM alerts 
WHERE user_id = $1 AND status = 'active'
ORDER BY priority_level DESC;

-- Count alerts by severity
SELECT severity, COUNT(*) 
FROM alerts 
WHERE user_id = $1 AND status = 'active'
GROUP BY severity;
```

---

## 🚀 **Benefits Achieved**

### **🎯 Healthcare Focus:**
- ✅ **Critical events ưu tiên** không bị lấn át
- ✅ **Persistent dangerous state** cho đến khi được xử lý  
- ✅ **Smart notification** dựa trên medical severity
- ✅ **No spam** với low-priority events

### **🔧 Technical Excellence:**
- ✅ **Zero additional API endpoints** needed
- ✅ **Leverage existing database schema** 
- ✅ **Real-time by default** với Supabase
- ✅ **Simplified mobile architecture**
- ✅ **Automatic audit trail** với event_detections

### **📱 Mobile Experience:**
- ✅ **Priority-ordered alert list** automatic
- ✅ **Real-time updates** without polling
- ✅ **Direct database operations** for actions
- ✅ **Built-in authentication** và permissions

---

## 🔍 **Testing Scenarios**

### **Priority Logic Tests:**
1. **Low Fall (45%)** → No alert created, no notification
2. **Medium Fall (65%)** → Create alert (high priority), send notification  
3. **Critical Seizure (80%)** → Create alert (critical priority), send notification
4. **Low Seizure (35%)** after critical → Skip alert, no notification
5. **Critical Fall (85%)** after critical seizure → Create alert (same priority level)

### **Mobile Integration Tests:**
1. **Real-time alert reception** qua Supabase channels
2. **Priority ordering** trong alert list
3. **Acknowledge action** updates database và triggers realtime
4. **Resolve action** removes từ active list
5. **Push notification** chỉ với critical/high severity

---

## 🎯 **Conclusion**

**API endpoints KHÔNG cần thiết** vì:
- Supabase Realtime đã handle tất cả real-time needs
- Mobile app có thể query database trực tiếp với priority ordering
- Actions (acknowledge/resolve) thực hiện trực tiếp qua Supabase client
- Giảm complexity và server maintenance overhead

**Priority system implemented** ở publishing level trong `HealthcareEventPublisher`, đảm bảo chỉ events quan trọng mới tạo alerts và gửi notifications, hoàn toàn đáp ứng yêu cầu **"phân tầng event theo mức độ nguy hiểm"** của bạn.
