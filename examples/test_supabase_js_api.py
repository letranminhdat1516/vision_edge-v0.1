#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

print('🔍 Checking Supabase config from .env:')
print(f'SUPABASE_URL: {os.getenv("SUPABASE_URL")}')
print(f'SUPABASE_KEY exists: {bool(os.getenv("SUPABASE_KEY"))}')
print(f'SUPABASE_KEY length: {len(os.getenv("SUPABASE_KEY") or "")}')

# Test connection
try:
    from supabase import create_client
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY') 
    
    if not url or not key:
        print('❌ Missing SUPABASE_URL or SUPABASE_KEY')
        exit(1)
        
    print('\n🔄 Creating Supabase client...')
    supabase = create_client(url, key)
    
    # Test simple query
    print('🧪 Testing query...')
    result = supabase.table('event_detections').select('count').limit(1).execute()
    print('✅ Python Supabase connection successful!')
    
    # Test getting recent events
    print('📋 Getting recent events...')
    recent = supabase.table('event_detections').select('*').order('created_at', desc=True).limit(3).execute()
    print(f'📊 Found {len(recent.data)} recent events')
    for event in recent.data:
        print(f'  - {event.get("event_type")}: {event.get("confidence_score")} at {event.get("detected_at")}')
        
except Exception as e:
    print(f'❌ Python connection failed: {e}')
    import traceback
    traceback.print_exc()
