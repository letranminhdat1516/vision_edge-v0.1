#!/usr/bin/env python3
"""
Simple HTTP Server để serve HTML và API endpoints cho healthcare events
"""

import os
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment
load_dotenv()

class HealthcareAPIHandler(SimpleHTTPRequestHandler):
    """HTTP Handler với API endpoints cho healthcare data"""
    
    def __init__(self, *args, **kwargs):
        # Database connection params
        self.db_params = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == '/api/latest-events':
            self.handle_latest_events(parsed_url.query)
        elif parsed_url.path == '/api/new-events':
            self.handle_new_events(parsed_url.query)
        elif parsed_url.path == '/api/stats':
            self.handle_stats()
        else:
            # Serve static files
            super().do_GET()
    
    def handle_latest_events(self, query_string):
        """Get latest healthcare events from database"""
        try:
            # Parse query parameters
            params = parse_qs(query_string)
            limit = int(params.get('limit', [10])[0])
            
            # Connect to database
            conn = psycopg2.connect(**self.db_params, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            
            # Query latest events
            cursor.execute("""
                SELECT 
                    id, event_type, confidence_score, detected_at, 
                    camera_id, location, metadata, created_at
                FROM event_detections 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            
            events = cursor.fetchall()
            
            # Convert to JSON-serializable format
            events_data = []
            for event in events:
                events_data.append({
                    'id': event['id'],
                    'event_type': event['event_type'],
                    'confidence_score': float(event['confidence_score']),
                    'detected_at': event['detected_at'].isoformat() if event['detected_at'] else None,
                    'camera_id': event['camera_id'],
                    'location': event['location'],
                    'metadata': event['metadata'],
                    'created_at': event['created_at'].isoformat() if event['created_at'] else None
                })
            
            cursor.close()
            conn.close()
            
            # Send JSON response
            self.send_json_response({
                'success': True,
                'events': events_data,
                'count': len(events_data)
            })
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def handle_new_events(self, query_string):
        """Get new events since last check"""
        try:
            params = parse_qs(query_string)
            since_id = params.get('since', [0])[0]
            
            conn = psycopg2.connect(**self.db_params, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            
            # Get events newer than since_id
            cursor.execute("""
                SELECT 
                    id, event_type, confidence_score, detected_at,
                    camera_id, location, metadata, created_at
                FROM event_detections 
                WHERE id > %s
                ORDER BY created_at ASC
            """, (since_id,))
            
            events = cursor.fetchall()
            
            events_data = []
            for event in events:
                events_data.append({
                    'id': event['id'],
                    'event_type': event['event_type'],
                    'confidence_score': float(event['confidence_score']),
                    'detected_at': event['detected_at'].isoformat() if event['detected_at'] else None,
                    'camera_id': event['camera_id'],
                    'location': event['location'],
                    'metadata': event['metadata'],
                    'created_at': event['created_at'].isoformat() if event['created_at'] else None
                })
            
            cursor.close()
            conn.close()
            
            self.send_json_response({
                'success': True,
                'events': events_data,
                'count': len(events_data)
            })
            
        except Exception as e:
            print(f"❌ New events API error: {e}")
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def handle_stats(self):
        """Get healthcare statistics"""
        try:
            conn = psycopg2.connect(**self.db_params, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            
            # Total events
            cursor.execute("SELECT COUNT(*) as total FROM event_detections")
            total = cursor.fetchone()['total']
            
            # Fall events
            cursor.execute("SELECT COUNT(*) as falls FROM event_detections WHERE event_type = 'fall'")
            falls = cursor.fetchone()['falls']
            
            # Seizure events  
            cursor.execute("SELECT COUNT(*) as seizures FROM event_detections WHERE event_type = 'abnormal_behavior'")
            seizures = cursor.fetchone()['seizures']
            
            # Latest event time
            cursor.execute("SELECT MAX(detected_at) as latest FROM event_detections")
            latest = cursor.fetchone()['latest']
            
            cursor.close()
            conn.close()
            
            self.send_json_response({
                'success': True,
                'stats': {
                    'total_events': total,
                    'fall_events': falls,
                    'seizure_events': seizures,
                    'latest_event': latest.isoformat() if latest else None
                }
            })
            
        except Exception as e:
            print(f"❌ Stats API error: {e}")
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        response_data = json.dumps(data, indent=2)
        
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()
        self.wfile.write(response_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {format % args}")

def run_server(port=8000):
    """Run the HTTP server"""
    try:
        print(f"🚀 Starting Healthcare API Server on port {port}")
        print(f"📊 Database: {os.getenv('DB_HOST')}")
        print(f"🌐 Access at: http://localhost:{port}")
        print(f"📱 API endpoints:")
        print(f"   - http://localhost:{port}/api/latest-events?limit=10")
        print(f"   - http://localhost:{port}/api/new-events?since=0")
        print(f"   - http://localhost:{port}/api/stats")
        print("=" * 60)
        
        httpd = HTTPServer(('localhost', port), HealthcareAPIHandler)
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    run_server()
