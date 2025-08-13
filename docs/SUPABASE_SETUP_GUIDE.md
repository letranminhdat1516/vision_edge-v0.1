# HƯỚNG DẪN SETUP SUPABASE REALTIME CHO HEALTHCARE SYSTEM

## Bước 1: Tạo Supabase Project

1. Truy cập https://supabase.com
2. Tạo account và new project
3. Chờ project khởi tạo (khoảng 2 phút)

## Bước 2: Lấy API Keys

1. Vào Settings > API
2. Copy các thông tin sau:
   - Project URL: `https://your-project-ref.supabase.co`
   - anon public key: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
   - service_role key: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

## Bước 3: Tạo Database Tables

Chạy SQL trong Supabase SQL Editor:

```sql
-- Tạo bảng event_detections
CREATE TABLE IF NOT EXISTS public.event_detections (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    camera_id TEXT,
    location TEXT,
    description TEXT,
    bounding_box JSONB,
    snapshot_path TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.event_detections ENABLE ROW LEVEL SECURITY;

-- Create policy cho realtime
CREATE POLICY "Enable realtime for all users" ON public.event_detections
    FOR ALL USING (true);

-- Enable realtime cho table
ALTER PUBLICATION supabase_realtime ADD TABLE public.event_detections;
```

## Bước 4: Cấu hình Environment

Tạo file `.env` trong thư mục root:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-key-here

# Camera Configuration
CAMERA_URL=admin:L2C37340@192.168.8.122:554
```

## Bước 5: Cấu hình HTML Demo

Mở file `examples/mobile_realtime_demo.html` và thay đổi:

```javascript
// Thay đổi dòng này:
const SUPABASE_URL = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';

// Thành:
const SUPABASE_URL = 'https://your-project-ref.supabase.co';
const SUPABASE_ANON_KEY = 'your-actual-anon-key';
```

## Bước 6: Test Realtime

### Method 1: Test với Simple Insert
```bash
python examples/test_supabase_insert.py
```

### Method 2: Test với Full Healthcare System
```bash
python examples/healthcare_realtime_test.py
```

### Method 3: Manual Test với Supabase SQL Editor
```sql
INSERT INTO public.event_detections (
    event_id, event_type, confidence_score, 
    camera_id, location, description
) VALUES (
    'manual_test_' || extract(epoch from now()),
    'fall',
    0.85,
    'test_cam',
    'Test Room',
    'Manual test for mobile realtime'
);
```

## Bước 7: Xem Realtime Notifications

1. Mở `examples/mobile_realtime_demo.html` trong browser
2. Chạy test script hoặc insert manual
3. Sẽ thấy notification realtime xuất hiện ngay lập tức!

## Troubleshooting

### Lỗi "Table doesn't exist"
- Chạy SQL create table trong Supabase SQL Editor

### Lỗi "Realtime not working"  
- Kiểm tra RLS policy
- Kiểm tra publication: `ALTER PUBLICATION supabase_realtime ADD TABLE public.event_detections;`

### Lỗi "Connection failed"
- Kiểm tra SUPABASE_URL và keys trong .env
- Kiểm tra internet connection

### Lỗi "Anon key invalid"
- Copy đúng anon key từ Settings > API
- Không dùng service_role key cho client-side
