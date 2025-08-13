#!/usr/bin/env python3
"""
Simple test to query database and show JSON format
"""
import psycopg2
import json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

def test_json_response():
    try:
        # Connect to database
        conn = psycopg2.connect(
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME')
        )
        cursor = conn.cursor()
        
        print('🔍 Testing JSON response format...\n')
        
        # Get latest events
        cursor.execute("""
            SELECT 
                event_id,
                event_type,
                confidence_score,
                detected_at,
                created_at,
                detection_data,
                context_data,
                status
            FROM event_detections 
            ORDER BY detected_at DESC, created_at DESC
            LIMIT 5
        """)
        
        events = cursor.fetchall()
        print(f'📊 Found {len(events)} events')
        
        # Convert to JSON format
        events_list = []
        for event in events:
            # Convert row to dict manually
            event_dict = {
                'event_id': str(event[0]),  # UUID to string
                'event_type': event[1],
                'confidence_score': float(event[2]) if event[2] else 0.0,
                'detected_at': event[3].isoformat() if event[3] else None,
                'created_at': event[4].isoformat() if event[4] else None,
                'detection_data': event[5] if event[5] else {},
                'context_data': event[6] if event[6] else {},
                'status': event[7]
            }
            events_list.append(event_dict)
        
        # Create API response format
        api_response = {
            'success': True,
            'events': events_list,
            'count': len(events_list),
            'timestamp': datetime.now().isoformat()
        }
        
        # Print JSON
        print('📋 JSON Response:')
        print('=' * 50)
        print(json.dumps(api_response, indent=2, ensure_ascii=False))
        print('=' * 50)
        
        # Print sample event for mobile format
        if events_list:
            sample_event = events_list[0]
            
            # Determine status based on confidence and type
            status = 'normal'
            if sample_event['event_type'] == 'fall':
                if sample_event['confidence_score'] >= 0.8:
                    status = 'danger'
                elif sample_event['confidence_score'] >= 0.6:
                    status = 'warning'
            elif sample_event['event_type'] == 'abnormal_behavior':
                if sample_event['confidence_score'] >= 0.7:
                    status = 'danger'
                elif sample_event['confidence_score'] >= 0.5:
                    status = 'warning'
            
            # Mobile format
            mobile_format = {
                'imageUrl': f"https://healthcare-system.com/snapshots/{sample_event['event_id']}.jpg",
                'status': status,
                'action': f"Phát hiện {sample_event['event_type']} với độ tin cậy {int(sample_event['confidence_score'] * 100)}%",
                'time': sample_event['detected_at']
            }
            
            print('\n📱 Mobile Format Example:')
            print('=' * 50)
            print(json.dumps(mobile_format, indent=2, ensure_ascii=False))
            print('=' * 50)
        
        cursor.close()
        conn.close()
        
        print('✅ JSON test completed successfully!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_json_response()
