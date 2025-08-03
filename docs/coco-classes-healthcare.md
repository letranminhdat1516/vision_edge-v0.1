# COCO Classes for Healthcare Monitoring

## 📋 **COCO Dataset Overview**

COCO (Common Objects in Context) có **80 classes** objects. Cho healthcare monitoring (detect té ngã/co giật), chúng ta cần focus vào những classes liên quan.

## 🏥 **Healthcare-Relevant COCO Classes**

### **🔴 Critical (Must Have)**

| Class ID | Class Name | Healthcare Use | Implementation Priority |
|----------|------------|----------------|------------------------|
| 0 | `person` | **Bệnh nhân chính** - Đối tượng monitor | ⭐⭐⭐⭐⭐ |
| 56 | `bed` | Giường bệnh - Context té ngã | ⭐⭐⭐⭐ |
| 56 | `chair` | Ghế - Có thể té từ ghế | ⭐⭐⭐⭐ |

### **🟠 Important (High Priority)**

| Class ID | Class Name | Healthcare Use | Implementation Priority |
|----------|------------|----------------|------------------------|
| 57 | `sofa` | Ghế sofa trong phòng bệnh | ⭐⭐⭐ |
| 61 | `toilet` | Nhà vệ sinh - High fall risk area | ⭐⭐⭐ |
| 67 | `cell phone` | Điện thoại - Tool gọi cấp cứu | ⭐⭐ |

### **🟡 Useful (Medium Priority)**

| Class ID | Class Name | Healthcare Use | Implementation Priority |
|----------|------------|----------------|------------------------|
| 39 | `bottle` | Chai thuốc - Có thể rơi khi co giật | ⭐⭐ |
| 41 | `cup` | Cốc nước - Indicator movement | ⭐⭐ |
| 62 | `tvmonitor` | TV trong phòng bệnh | ⭐ |
| 63 | `laptop` | Thiết bị y tế/monitoring | ⭐ |

### **🟢 Context Objects (Low Priority)**

| Class ID | Class Name | Healthcare Use | Implementation Priority |
|----------|------------|----------------|------------------------|
| 64 | `mouse` | Computer peripherals | ⭐ |
| 65 | `remote` | TV remote | ⭐ |
| 66 | `keyboard` | Computer keyboard | ⭐ |
| 73 | `book` | Reading material | ⭐ |
| 74 | `clock` | Time reference | ⭐ |

## ❌ **Irrelevant for Healthcare (Ignore)**

### **Transportation (không cần)**
- `bicycle`, `car`, `motorbike`, `bus`, `train`, `truck`, `boat`, `aeroplane`

### **Outdoor Objects (không cần)**  
- `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`

### **Animals (không có trong bệnh viện)**
- `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`

### **Personal Items (không quan trọng cho medical monitoring)**
- `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`

### **Sports Items (không liên quan)**
- `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`

### **Food Items (noise cho medical monitoring)**
- `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`

### **Kitchen Items (không trong phòng bệnh)**
- `wine glass`, `fork`, `knife`, `spoon`, `bowl`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`

### **Personal Care (không cần detect)**
- `scissors`, `teddy bear`, `hair drier`, `toothbrush`, `vase`

## 🎯 **Healthcare Detection Strategy**

### **Primary Detection Pipeline:**
```python
# YOLO Detection với filtered classes
HEALTHCARE_CLASSES = [
    'person',    # ID: 0 - CRITICAL
    'bed',       # ID: 56 - HIGH  
    'chair',     # ID: 56 - HIGH
    'sofa',      # ID: 57 - MEDIUM
    'toilet',    # ID: 61 - MEDIUM
    'bottle',    # ID: 39 - LOW
    'cup',       # ID: 41 - LOW
    'cell phone' # ID: 67 - LOW
]

def filter_healthcare_objects(yolo_detections):
    """Filter chỉ giữ objects liên quan healthcare"""
    filtered = []
    for detection in yolo_detections:
        if detection.class_name in HEALTHCARE_CLASSES:
            filtered.append(detection)
    return filtered
```

### **Context-Based Risk Assessment:**
```python
def calculate_fall_risk(detected_objects, room_type):
    """Tính risk score dựa trên objects và room context"""
    risk_score = 0.0
    
    if 'person' not in detected_objects:
        return 0.0  # Không có người = không có risk
    
    base_risk = 0.3
    
    # Object-based risk
    if 'bed' in detected_objects:
        risk_score += 0.2  # Fall từ giường
    if 'chair' in detected_objects:
        risk_score += 0.15  # Fall từ ghế
    if 'toilet' in detected_objects:
        risk_score += 0.25  # Bathroom slip risk
    if 'sofa' in detected_objects:
        risk_score += 0.1   # Fall từ sofa
        
    # Room-based risk multiplier
    room_multipliers = {
        'bathroom': 1.5,     # High slip risk
        'patient_room': 1.0, # Normal
        'hallway': 0.8,      # Lower risk
        'nurse_station': 0.5 # Supervised area
    }
    
    multiplier = room_multipliers.get(room_type, 1.0)
    final_risk = (base_risk + risk_score) * multiplier
    
    return min(final_risk, 1.0)
```

### **Object Tracking for Seizure Detection:**
```python
def track_objects_for_seizure(frame_objects_history):
    """Track object movement patterns để detect seizure"""
    
    # Track sudden disappearance của objects
    previous_frame = frame_objects_history[-2] if len(frame_objects_history) > 1 else []
    current_frame = frame_objects_history[-1]
    
    disappeared_objects = []
    for obj in previous_frame:
        if obj not in current_frame and obj in ['bottle', 'cup', 'cell phone']:
            disappeared_objects.append(obj)
    
    # Nếu nhiều objects đột ngột biến mất = có thể co giật
    if len(disappeared_objects) >= 2:
        return {
            'seizure_risk': 0.7,
            'evidence': f"Objects dropped: {disappeared_objects}",
            'confidence': 0.8
        }
    
    return {'seizure_risk': 0.0}
```

## 📊 **Implementation Priority**

### **Phase 1: Core Objects (Week 1-2)**
```python
PHASE_1_OBJECTS = ['person', 'bed', 'chair']
# Focus: Basic person detection + furniture context
```

### **Phase 2: Risk Objects (Week 3-4)**  
```python
PHASE_2_OBJECTS = ['sofa', 'toilet']
# Focus: High-risk area detection
```

### **Phase 3: Context Objects (Week 5-6)**
```python
PHASE_3_OBJECTS = ['bottle', 'cup', 'cell phone']  
# Focus: Seizure indicators + emergency tools
```

### **Phase 4: Full Healthcare (Week 7-8)**
```python
ALL_HEALTHCARE_OBJECTS = PHASE_1 + PHASE_2 + PHASE_3
# Focus: Complete healthcare monitoring system
```

## 🔧 **Configuration Template**

```python
# Healthcare YOLO Configuration
HEALTHCARE_YOLO_CONFIG = {
    'model_path': 'models/yolo/yolov8s.pt',
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    
    'target_classes': {
        # Class ID: (class_name, priority, healthcare_relevance)
        0: ('person', 'critical', 'primary_target'),
        56: ('bed', 'high', 'fall_context'), 
        56: ('chair', 'high', 'fall_context'),
        57: ('sofa', 'medium', 'fall_context'),
        61: ('toilet', 'medium', 'slip_risk'),
        39: ('bottle', 'low', 'seizure_indicator'),
        41: ('cup', 'low', 'seizure_indicator'),
        67: ('cell phone', 'low', 'emergency_tool')
    },
    
    'ignore_classes': [
        # Transportation
        'bicycle', 'car', 'motorbike', 'bus', 'train', 'truck',
        # Animals  
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        # Food
        'banana', 'apple', 'pizza', 'sandwich', 'orange',
        # Sports
        'frisbee', 'sports ball', 'baseball bat', 'tennis racket'
    ]
}
```

---

## 📝 **Summary**

Từ **80 COCO classes**, chúng ta chỉ cần focus vào **~8 classes** cho healthcare monitoring:

1. **`person`** - Đối tượng chính cần monitor
2. **`bed`, `chair`, `sofa`** - Furniture context cho fall detection  
3. **`toilet`** - High-risk area
4. **`bottle`, `cup`** - Seizure indicators
5. **`cell phone`** - Emergency communication

**Approach này giúp:**
- ✅ **Reduce noise** từ irrelevant objects
- ✅ **Focus processing** vào healthcare-relevant detection
- ✅ **Improve accuracy** bằng context-aware analysis
- ✅ **Optimize performance** bằng class filtering
