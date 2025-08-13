"""
Supabase Configuration for Healthcare Monitoring System
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SupabaseConfig:
    """Configuration class for Supabase connection and realtime settings"""
    
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL', 'https://your-project.supabase.co')
        self.key = os.getenv('SUPABASE_KEY', 'your-anon-key')  # Changed from SUPABASE_ANON_KEY
        self.service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY', self.key)  # Fallback to anon key
        self.database_url = os.getenv('DATABASE_URL', '')  # Direct PostgreSQL connection
        
        # Realtime configuration
        self.realtime_config = {
            'enabled_tables': [
                'event_detections',
                'alerts',
                'snapshots',
                'notifications',
                'cameras'
            ],
            'broadcast_channel': 'healthcare_monitoring',
            'presence_channel': 'system_status'
        }
    
    def get_client_config(self) -> Dict[str, str]:
        """Get configuration for Supabase client"""
        return {
            'url': self.url,
            'key': self.key
        }
    
    def get_service_config(self) -> Dict[str, str]:
        """Get configuration for Supabase service role client (for server operations)"""
        return {
            'url': self.url,
            'key': self.service_role_key
        }

# Global config instance
supabase_config = SupabaseConfig()
