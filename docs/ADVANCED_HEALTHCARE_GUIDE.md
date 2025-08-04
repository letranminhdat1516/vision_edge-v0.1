# Advanced Healthcare Monitor - Dual Detection System

## 🏥 Tổng quan
Hệ thống giám sát y tế tiên tiến với khả năng phát hiện đồng thời:
- **Fall Detection** (Phát hiện té ngã) - sử dụng YOLO + pose analysis
- **Seizure Detection** (Phát hiện co giật) - sử dụng VSViG + temporal analysis
- **Real-time Statistics** (Thống kê thời gian thực)
- **Keypoint Visualization** (Hiển thị điểm khớp) - có thể bật/tắt

## 🚀 Tính năng chính

### Dual Detection System
- **Fall Detection**: Phát hiện té ngã với độ chính xác cao
- **Seizure Detection**: Phát hiện co giật bằng AI VSViG model
- **Emergency Alerts**: Cảnh báo khẩn cấp với các mức độ khác nhau
- **Priority System**: Ưu tiên cảnh báo theo mức độ nghiêm trọng

### Real-time Monitoring
- **Live Camera Feed**: Kết nối camera IMOU qua RTSP
- **Motion Detection**: Phát hiện chuyển động để tối ưu xử lý
- **Keyframe Optimization**: Chỉ xử lý frame có người
- **Performance Optimization**: Tối ưu hiệu suất với adaptive processing

### Visualization Features
- **Keypoint Display**: Hiển thị 17 điểm khớp của cơ thể
- **Skeleton Connections**: Vẽ khung xương kết nối các điểm khớp
- **Alert Overlays**: Hiển thị cảnh báo trực tiếp trên video
- **Statistics Panel**: Bảng thống kê thời gian thực

### Alert System
- **Normal**: Trạng thái bình thường (xanh lá)
- **Warning**: Cảnh báo nghi ngờ co giật (vàng)
- **High**: Phát hiện té ngã (cam)
- **Critical**: Phát hiện co giật (đỏ)

## 🎮 Điều khiển

### Keyboard Controls
- **'k'**: Bật/tắt hiển thị keypoints
- **'s'**: Bật/tắt hiển thị statistics
- **'q'**: Thoát chương trình

### Configuration Options
Khi khởi động, bạn có thể cấu hình:
1. **Show keypoints**: Hiển thị điểm khớp trên video
2. **Show statistics**: Hiển thị bảng thống kê

## 📊 Statistics Dashboard

### General Stats
- **Runtime**: Thời gian chạy
- **FPS**: Tốc độ xử lý frame
- **Efficiency**: Hiệu suất xử lý (% frame bỏ qua)
- **Total Frames**: Tổng số frame
- **Processed**: Số frame đã xử lý
- **Persons**: Số lượng người phát hiện

### Fall Detection Stats
- **Falls Detected**: Số lần té ngã phát hiện
- **Average Confidence**: Độ tin cậy trung bình
- **Last Fall Time**: Thời gian té ngã cuối cùng

### Seizure Detection Stats
- **Seizures Detected**: Số lần co giật phát hiện
- **Warnings**: Số cảnh báo nghi ngờ
- **Pose Failures**: Số lần trích xuất pose thất bại
- **Buffer Status**: Trạng thái buffer temporal

### Emergency Stats
- **Critical Alerts**: Số cảnh báo nghiêm trọng
- **Total Alerts**: Tổng số cảnh báo
- **Current Status**: Trạng thái hiện tại

### Performance Stats
- **Fall Detection Time**: Thời gian xử lý fall detection
- **Seizure Detection Time**: Thời gian xử lý seizure detection
- **Total Time**: Tổng thời gian xử lý

## 🔧 Cấu hình hệ thống

### Camera Configuration
```python
camera_config = {
    'url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
    'buffer_size': 1,
    'fps': 15,
    'resolution': (640, 480),
    'auto_reconnect': True
}
```

### Detection Thresholds
- **Fall Detection**: confidence_threshold=0.7
- **Seizure Detection**: confidence_threshold=0.7
- **YOLO Person Detection**: confidence=0.4
- **Motion Detection**: threshold=120

### Temporal Analysis
- **Seizure Buffer**: 30 frames (2 giây ở 15 FPS)
- **Alert Cooldown**: 2 giây cho fall detection
- **Warning Threshold**: 0.4 cho seizure detection

## 🚨 Alert Levels

### Normal (Xanh lá)
- Không có phát hiện bất thường
- Hệ thống hoạt động bình thường

### Warning (Vàng)
- Nghi ngờ có dấu hiệu co giật
- Confidence 0.4-0.7
- Cần theo dõi thêm

### High (Cam)
- Phát hiện té ngã
- Cần can thiệp ngay lập tức

### Critical (Đỏ)
- Phát hiện co giật
- Tình trạng khẩn cấp
- Cần hỗ trợ y tế ngay

## 📈 Keypoint Visualization

### 17 Body Keypoints
```
0: Nose            9: Left Wrist
1: Left Eye       10: Right Wrist
2: Right Eye      11: Left Hip
3: Left Ear       12: Right Hip
4: Right Ear      13: Left Knee
5: Left Shoulder  14: Right Knee
6: Right Shoulder 15: Left Ankle
7: Left Elbow     16: Right Ankle
8: Right Elbow
```

### Skeleton Connections
- **Arms**: Shoulder → Elbow → Wrist
- **Legs**: Hip → Knee → Ankle
- **Torso**: Shoulder ↔ Shoulder, Hip ↔ Hip

### Color Coding
- **Green**: High confidence keypoints (>0.7)
- **Yellow**: Medium confidence keypoints (0.3-0.7)
- **Blue**: Skeleton connections

## 🔍 Technical Details

### Dual Detection Pipeline
1. **Frame Acquisition**: RTSP camera feed
2. **Motion Detection**: Phát hiện chuyển động
3. **Person Detection**: YOLO object detection
4. **Keyframe Selection**: Chọn frame có người
5. **Parallel Processing**:
   - Fall Detection: Pose analysis + rule-based
   - Seizure Detection: VSViG temporal analysis
6. **Alert Fusion**: Kết hợp kết quả và ưu tiên
7. **Visualization**: Hiển thị kết quả và statistics

### Performance Optimization
- **Adaptive Processing**: Chỉ xử lý frame có motion/person
- **Buffer Management**: Temporal buffer cho seizure detection
- **Multi-threading**: Camera capture riêng biệt
- **Memory Management**: Automatic cleanup và optimization

## 🛠️ Troubleshooting

### Camera Connection Issues
1. Kiểm tra IP và credentials của camera IMOU
2. Đảm bảo camera hỗ trợ RTSP
3. Kiểm tra network connectivity

### Seizure Detection Issues
1. VSViG model cần GPU để tối ưu hiệu suất
2. Pose extraction có thể thất bại với lighting kém
3. Temporal buffer cần 30 frames để hoạt động

### Performance Issues
1. Giảm resolution camera nếu cần
2. Tăng motion_threshold để giảm xử lý
3. Tắt keypoint visualization nếu không cần

## 📝 Logging

### Log Files
- Location: `logs/advanced_healthcare_monitor_YYYYMMDD_HHMMSS.log`
- Format: Timestamp - Component - Level - Message
- Levels: INFO, WARNING, ERROR, CRITICAL

### Log Content
- System initialization
- Detection events (fall/seizure)
- Performance metrics
- Error diagnostics
- Alert notifications

## 🎯 Use Cases

### Home Healthcare
- Giám sát người cao tuổi
- Phát hiện té ngã tại nhà
- Theo dõi tình trạng sức khỏe

### Medical Facilities
- Giám sát bệnh nhân
- Phát hiện co giật epilepsy
- Hỗ trợ nhân viên y tế

### Rehabilitation Centers
- Theo dõi quá trình phục hồi
- Phát hiện sớm các vấn đề
- Đánh giá tiến triển

## 🔮 Future Enhancements

### Planned Features
- [ ] Multi-camera support
- [ ] Cloud storage integration
- [ ] Mobile app notifications
- [ ] Advanced analytics dashboard
- [ ] AI model updates
- [ ] Custom alert rules

### Integration Options
- [ ] Hospital management systems
- [ ] Emergency response systems
- [ ] Family notification systems
- [ ] Wearable device integration

---

**Developed by**: Vision Edge Healthcare Team  
**Version**: 0.1  
**Last Updated**: December 2024
