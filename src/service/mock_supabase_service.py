"""
Mock Supabase Service for Development/Demo
Simulates Supabase operations without actual network calls
"""

import uuid
import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
import logging

logger = logging.getLogger(__name__)

class MockSupabaseService:
    """Mock Supabase service for development and demo purposes"""
    
    def __init__(self):
        self.is_connected = True
        self.mock_data = {
            'event_detections': [],
            'alerts': [],
            'snapshots': []
        }
        self.subscribers = {}
        
        logger.info("Mock Supabase service initialized (Demo Mode)")
    
    def table(self, table_name: str):
        """Mock table interface"""
        return MockTable(table_name, self)
    
    def subscribe_to_events(self, table: str, event_type: str, handler: Callable):
        """Mock subscription to events"""
        subscription_key = f"{table}_{event_type}"
        self.subscribers[subscription_key] = handler
        logger.info(f"Subscribed to mock {table} {event_type} events")
    
    def publish_event(self, table: str, data: Dict[str, Any]):
        """Simulate publishing an event and notifying subscribers"""
        # Store in mock data
        if table not in self.mock_data:
            self.mock_data[table] = []
        
        self.mock_data[table].append(data)
        
        # Notify subscribers
        subscription_key = f"{table}_INSERT"
        if subscription_key in self.subscribers:
            event_data = {
                'event_type': 'INSERT',
                'table': table,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'new_data': data,
                'old_data': {}
            }
            
            # Call handler in separate thread
            threading.Thread(
                target=self.subscribers[subscription_key],
                args=(event_data,),
                daemon=True
            ).start()
        
        logger.info(f"Mock event published to {table}: {data.get('event_type', 'unknown')}")
        return data
    
    def get_recent_events(self, table: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent mock events"""
        events = self.mock_data.get(table, [])
        return events[-limit:] if events else []
    
    def close(self):
        """Mock close operation"""
        logger.info("Mock Supabase service closed")

class MockTable:
    """Mock table interface"""
    
    def __init__(self, table_name: str, service: MockSupabaseService):
        self.table_name = table_name
        self.service = service
    
    def insert(self, data: Dict[str, Any]):
        """Mock insert operation"""
        # Add default fields
        data['created_at'] = datetime.now(timezone.utc).isoformat()
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())
        
        return MockResult([data])
    
    def select(self, columns: str = '*'):
        """Mock select operation"""
        return MockQuery(self.table_name, self.service)

class MockQuery:
    """Mock query builder"""
    
    def __init__(self, table_name: str, service: MockSupabaseService):
        self.table_name = table_name
        self.service = service
    
    def limit(self, count: int):
        """Mock limit"""
        return self
    
    def order(self, column: str, desc: bool = False):
        """Mock order"""
        return self
    
    def gte(self, column: str, value: Any):
        """Mock greater than or equal"""
        return self
    
    def execute(self):
        """Mock execute"""
        data = self.service.mock_data.get(self.table_name, [])
        return MockResult(data)

class MockResult:
    """Mock query result"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

# Global mock service
mock_supabase_service = MockSupabaseService()
