"""
Check Specific Fall Events - Phân tích 4 events liên tiếp
"""
import psycopg2
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

def check_specific_fall_events():
    """Kiểm tra chi tiết 4 fall events vừa xảy ra"""
    try:
        conn = psycopg2.connect(
            host="aws-1-ap-southeast-1.pooler.supabase.com",
            port=5432,
            database="postgres",
            user="postgres.undznprwlqjpnxqsgyiv",
            password=os.getenv("DB_PASSWORD", "Phanthihoaidiem7903")
        )
        
        cur = conn.cursor()
        
        print("🔍 PHÂN TÍCH 4 FALL EVENTS LIÊN TIẾP")
        print("=" * 100)
        
        # Lấy 10 fall events gần nhất để phân tích
        cur.execute("""
            SELECT 
                event_id, 
                event_type, 
                event_description, 
                confidence_score, 
                detected_at,
                created_at,
                context_data,
                bounding_boxes,
                status,
                detection_data,
                camera_id
            FROM event_detections 
            WHERE event_type = 'fall'
            ORDER BY detected_at DESC 
            LIMIT 10;
        """)
        
        events = cur.fetchall()
        
        if not events:
            print("❌ Không tìm thấy fall events")
            return
        
        print(f"📊 Tìm thấy {len(events)} fall events gần nhất\n")
        
        # Phân tích từng event
        for i, event in enumerate(events, 1):
            (event_id, event_type, description, confidence, detected_at, 
             created_at, context_data, bounding_boxes, status, detection_data, camera_id) = event
            
            print(f"{'='*100}")
            print(f"🚨 EVENT #{i}: {event_id}")
            print(f"{'='*100}")
            print(f"⏰ Detected at: {detected_at}")
            print(f"📅 Created at:  {created_at}")
            print(f"📊 Confidence:  {confidence:.3f}")
            print(f"🎯 Status:      {status}")
            print(f"📷 Camera:      {camera_id}")
            print(f"📝 Description: {description[:150]}...")
            
            # Parse context_data (JSON)
            if context_data:
                try:
                    context = json.loads(context_data) if isinstance(context_data, str) else context_data
                    print(f"\n📦 CONTEXT DATA:")
                    
                    # Thông tin motion
                    if 'motion_level' in context:
                        print(f"   🏃 Motion level:     {context['motion_level']:.4f}")
                    
                    # Thông tin fall type
                    if 'fall_type' in context:
                        print(f"   🎭 Fall type:        {context['fall_type']}")
                    
                    # Thông tin fall velocity
                    if 'fall_velocity' in context:
                        print(f"   ⚡ Fall velocity:    {context['fall_velocity']:.2f} px/s")
                    
                    # Thông tin fall duration
                    if 'fall_duration' in context:
                        print(f"   ⏱️  Fall duration:    {context['fall_duration']:.3f} s")
                    
                    # Thông tin detection method
                    if 'detection_method' in context:
                        print(f"   🔍 Detection method: {context['detection_method']}")
                    
                    # Vertical movement
                    if 'vertical_movement' in context:
                        print(f"   📏 Vertical move:    {context['vertical_movement']:.1f} px")
                    
                    # Alert level
                    if 'alert_level' in context:
                        print(f"   ⚠️  Alert level:      {context['alert_level']}")
                        
                except Exception as e:
                    print(f"   ⚠️ Error parsing context: {e}")
            
            # Parse bounding_boxes (JSON)
            if bounding_boxes:
                try:
                    boxes = json.loads(bounding_boxes) if isinstance(bounding_boxes, str) else bounding_boxes
                    print(f"\n📦 BOUNDING BOXES: ({len(boxes)} boxes)")
                    for j, box in enumerate(boxes[:2], 1):  # Show first 2 boxes
                        x = box.get('x', 0)
                        y = box.get('y', 0)
                        w = box.get('width', 0)
                        h = box.get('height', 0)
                        aspect = w / h if h > 0 else 0
                        print(f"   Box {j}: x={x}, y={y}, w={w}, h={h}, aspect={aspect:.2f}")
                except Exception as e:
                    print(f"   ⚠️ Error parsing bounding boxes: {e}")
            
            # Parse detection_data (JSON)
            if detection_data:
                try:
                    det_data = json.loads(detection_data) if isinstance(detection_data, str) else detection_data
                    print(f"\n📦 DETECTION DATA:")
                    
                    if 'method' in det_data:
                        print(f"   🔍 Method:           {det_data['method']}")
                    
                    if 'confidence' in det_data:
                        print(f"   📊 Confidence:       {det_data['confidence']:.3f}")
                    
                    if 'has_real_motion' in det_data:
                        print(f"   🏃 Has real motion:  {det_data['has_real_motion']}")
                    
                    if 'is_rapid_fall' in det_data:
                        print(f"   ⚡ Is rapid fall:    {det_data['is_rapid_fall']}")
                        
                except Exception as e:
                    print(f"   ⚠️ Error parsing detection data: {e}")
            
            print()  # Empty line between events
        
        # ===== PHÂN TÍCH TIME GAP GIỮA CÁC EVENTS =====
        print("\n" + "="*100)
        print("⏱️  TIME GAP ANALYSIS - Khoảng cách giữa các events")
        print("="*100)
        
        for i in range(len(events) - 1):
            curr_event = events[i]
            next_event = events[i + 1]
            
            curr_time = curr_event[4]  # detected_at
            next_time = next_event[4]  # detected_at
            
            time_diff = (curr_time - next_time).total_seconds()
            
            print(f"Event #{i+1} → Event #{i+2}: {abs(time_diff):.1f} seconds")
            
            # Cảnh báo nếu quá gần nhau
            if abs(time_diff) < 15:
                print(f"   ⚠️  WARNING: Events quá gần nhau! Có thể là FALSE POSITIVE")
            else:
                print(f"   ✅ OK: Khoảng cách hợp lý")
        
        # ===== PHÂN TÍCH CONFIDENCE & MOTION =====
        print("\n" + "="*100)
        print("📊 CONFIDENCE & MOTION ANALYSIS")
        print("="*100)
        
        for i, event in enumerate(events, 1):
            confidence = event[3]
            context_data = event[6]
            
            motion_level = None
            if context_data:
                try:
                    context = json.loads(context_data) if isinstance(context_data, str) else context_data
                    motion_level = context.get('motion_level')
                except:
                    pass
            
            motion_str = f"{motion_level:.4f}" if motion_level is not None else "N/A"
            
            print(f"Event #{i}: confidence={confidence:.3f}, motion={motion_str}")
            
            # Cảnh báo nếu confidence cao nhưng motion thấp
            if confidence >= 0.85 and motion_level is not None and motion_level < 0.02:
                print(f"   ⚠️  WARNING: HIGH confidence nhưng LOW motion - Có thể là bbox jitter!")
        
        cur.close()
        conn.close()
        
        print("\n" + "="*100)
        print("✅ Phân tích hoàn tất!")
        print("="*100)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_specific_fall_events()
