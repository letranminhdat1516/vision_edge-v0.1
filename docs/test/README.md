# Healthcare System - Video Testing Suite

## 📋 Tổng quan

Hệ thống test tự động sử dụng video files (.mp4) để kiểm tra chức năng phát hiện té ngã và co giật.
Sử dụng **toàn bộ logic từ main.py** (trừ camera stream).

## 🎯 Chức năng

- ✅ Tự động quét tất cả video .mp4 trong folder `resource/`
- ✅ Test từng video và đánh số case (1, 2, 3, ...)
- ✅ Lưu statistics, keypoint images, alert images
- ✅ Generate Vietnamese captions
- ✅ Export Excel report với 3 sheets: Summary, Events, Statistics
- ✅ Hiển thị progress realtime

## 📁 Cấu trúc thư mục

```
examples/test/
├── resource/                    # ĐẶT VIDEO VÀO ĐÂY
│   ├── video_1.mp4
│   ├── video_2.mp4
│   └── video_n.mp4
├── test_results/               # KẾT QUẢ TEST
│   ├── reports/                # Excel reports (auto-generated)
│   │   └── test_report_YYYYMMDD_HHMMSS.xlsx
│   ├── alerts/                 # Alert images (khi phát hiện sự cố)
│   │   ├── case_1/
│   │   ├── case_2/
│   │   └── case_n/
│   ├── keypoints/              # Keypoint images (vẽ skeleton)
│   │   ├── case_1/
│   │   │   ├── frame_000030.jpg
│   │   │   ├── frame_000060.jpg
│   │   │   └── ...
│   │   └── case_n/
│   └── statistics/             # Statistics files
├── video_camera_service.py     # Video camera service
├── test_video_runner.py        # Main test runner
└── README.md                   # This file
```

## 🚀 Cách sử dụng

### Bước 1: Chuẩn bị video

```bash
# Copy video files vào resource folder
cp your_videos/*.mp4 examples/test/resource/

# Hoặc tạo folder và thêm video
mkdir -p examples/test/resource
# Đặt video_1.mp4, video_2.mp4, ... vào đây
```

### Bước 2: Chạy test

```bash
# Từ root folder của project
cd d:\FPT\Capstone\vision_edge-v0.1

# Chạy test
python examples/test/test_video_runner.py
```

### Bước 3: Xem kết quả

```bash
# Mở Excel report
start examples/test/test_results/reports/test_report_*.xlsx

# Xem alert images
explorer examples/test/test_results/alerts

# Xem keypoint images
explorer examples/test/test_results/keypoints
```

## 📊 Excel Report Format

### Sheet 1: Summary

| Case | Video Name | Status    | Processing Time | Total Frames | FPS  | Events | Falls | Seizures | Keypoint Images | Intelligent Actions |
| ---- | ---------- | --------- | --------------- | ------------ | ---- | ------ | ----- | -------- | --------------- | ------------------- |
| 1    | video_1    | completed | 45.2s           | 1350         | 29.8 | 3      | 2     | 1        | 45              | Yes                 |
| 2    | video_2    | completed | 32.1s           | 960          | 29.9 | 1      | 1     | 0        | 32              | Yes                 |

### Sheet 2: Detected Events

| Case | Video Name | Frame | Event Type        | Confidence | Alert Level | Timestamp | Action           | Caption (VN)         | Image Path        |
| ---- | ---------- | ----- | ----------------- | ---------- | ----------- | --------- | ---------------- | -------------------- | ----------------- |
| 1    | video_1    | 245   | fall              | 0.85       | critical    | 8.2s      | Kiểm tra ngay... | Người già té ngã...  | .../alert_001.jpg |
| 1    | video_1    | 782   | abnormal_behavior | 0.67       | high        | 26.1s     | Quan sát...      | Phát hiện co giật... | .../alert_002.jpg |

### Sheet 3: Statistics

| Case | Video Name | Total Frames | Persons Detected | Fall Events | Seizure Events | Avg FPS | ... |
| ---- | ---------- | ------------ | ---------------- | ----------- | -------------- | ------- | --- |
| 1    | video_1    | 1350         | 1247             | 2           | 1              | 29.8    | ... |

## 🖼️ Output Images

### Alert Images (khi có sự cố)

```
test_results/alerts/case_1/
├── alert_frame_000245_fall.jpg          # Frame phát hiện té ngã
├── alert_frame_000782_seizure.jpg       # Frame phát hiện co giật
└── ...
```

### Keypoint Images (vẽ skeleton)

```
test_results/keypoints/case_1/
├── frame_000030.jpg    # Mỗi 30 frames lưu 1 ảnh
├── frame_000060.jpg    # Có vẽ skeleton keypoints
├── frame_000090.jpg    # Có confidence scores
└── ...                 # + Khi có alert cũng lưu
```

## 📝 Test Output Example

```
================================================================================
🧪 HEALTHCARE SYSTEM VIDEO TEST SUITE
================================================================================
📹 Total Videos: 3
📂 Resource Folder: examples/test/resource
📊 Output Folder: examples/test/test_results
👤 User ID: test_user_001
🤖 Intelligent Actions: ENABLED
================================================================================

====================================================================================================
🎬 CASE #1: Testing Video 'video_1'
📹 Path: examples/test/resource/video_1.mp4
====================================================================================================

🔧 Initializing services...
📹 Video loaded successfully!
   Total frames: 1350
   Video FPS: 30.00
   Duration: 45.00s
🏥 Initializing Healthcare Pipeline...
🤖 Intelligent action pipeline initialized

✅ All systems initialized!
====================================================================================================

🎥 Starting video processing for CASE #1...
====================================================================================================

📊 Progress: 7.4% - Frame 100/1350
📊 Progress: 14.8% - Frame 200/1350

====================================================================================================
🚨 ALERT DETECTED - CASE #1
====================================================================================================
   Frame: 245/1350
   Event Type: fall
   Confidence: 85.23%
   Alert Level: critical
   📝 Action: Kiểm tra người bệnh ngay lập tức. Gọi hỗ trợ y tế nếu cần.
   🇻🇳 Caption: Người già bị té ngã trong phòng tắm, cần hỗ trợ khẩn cấp
   📸 Alert Image: examples/test/test_results/alerts/case_1/alert_001.jpg
====================================================================================================

...

✅ Video processing completed for CASE #1

====================================================================================================
✅ CASE #1 COMPLETED: video_1
====================================================================================================
   Processing Time: 45.23s
   Total Frames: 1350
   FPS: 29.85
   Detected Events: 3
   - Falls: 2
   - Seizures: 1
   Saved Keypoint Images: 45
====================================================================================================

✅ Completed video: video_1.mp4
   Status: completed
   Events detected: 3

====================================================================================================
🎬 CASE #2: Testing Video 'video_2'
...

====================================================================================================
✅ ALL TESTS COMPLETED!
====================================================================================================
📊 Total Cases: 3
📁 Results saved to: examples/test/test_results
📄 Excel Report: examples/test/test_results/reports/test_report_20251110_143022.xlsx
📸 Alert Images: examples/test/test_results/alerts
🎯 Keypoint Images: examples/test/test_results/keypoints
====================================================================================================
```

## 🎯 Test Tips

### Video chuẩn bị:

- Format: .mp4 (recommended)
- Resolution: 720p hoặc cao hơn
- FPS: 30fps (recommended)
- Nội dung: Rõ ràng, có người, có ánh sáng tốt

### Đặt tên video:

- `video_1.mp4`, `video_2.mp4`, ... (auto-sort)
- Hoặc: `fall_test.mp4`, `seizure_test.mp4`, ...
- Hệ thống tự động đánh số case

### Expected results:

- **Fall detection**: Confidence > 60%
- **Seizure detection**: Confidence > 50%
- **Normal activity**: Confidence < 30%

## 🔧 Troubleshooting

### Video không load được?

```bash
# Check video path
ls examples/test/resource/*.mp4

# Check video format
ffprobe video_1.mp4
```

### Không có kết quả?

```bash
# Check logs
cat examples/test/test_results/logs/*.log

# Check permissions
chmod +x examples/test/test_video_runner.py
```

### Import errors?

```bash
# Make sure you're in project root
cd d:\FPT\Capstone\vision_edge-v0.1

# Run from root
python examples/test/test_video_runner.py
```

## 📦 Requirements

- Python 3.8+
- OpenCV
- pandas
- openpyxl
- Tất cả dependencies từ main.py

## 🎓 Technical Details

- Sử dụng `VideoCameraService` thay vì `CameraService`
- Toàn bộ logic xử lý giống hệt `main.py`
- Auto-save keypoint images mỗi 30 frames + khi có alert
- Vietnamese captions từ BLIP model + translation
- Realtime progress tracking
