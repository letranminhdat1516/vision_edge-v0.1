"""
Simple network connectivity test for Supabase
"""

import os
import sys
import requests
from urllib.parse import urlparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.supabase_config import supabase_config

def test_basic_connectivity():
    """Test basic network connectivity"""
    print("🔍 Testing Network Connectivity...")
    print("=" * 50)
    
    # Test DNS resolution
    try:
        import socket
        url_parts = urlparse(supabase_config.url)
        hostname = url_parts.hostname
        
        print(f"📍 Testing DNS resolution for: {hostname}")
        ip = socket.gethostbyname(hostname)
        print(f"✅ DNS resolved: {hostname} → {ip}")
        
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    
    # Test HTTP connectivity
    try:
        print(f"📍 Testing HTTP connection to: {supabase_config.url}")
        
        # Simple GET request to check connectivity
        headers = {
            'apikey': supabase_config.key,
            'Authorization': f'Bearer {supabase_config.key}'
        }
        
        # Test REST API endpoint
        test_url = f"{supabase_config.url}/rest/v1/"
        response = requests.get(test_url, headers=headers, timeout=10)
        
        print(f"✅ HTTP connection successful: Status {response.status_code}")
        return True
        
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout error: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def test_supabase_config():
    """Test Supabase configuration"""
    print("\n🔧 Testing Supabase Configuration...")
    print("=" * 50)
    
    print(f"📍 URL: {supabase_config.url}")
    print(f"📍 Key: {supabase_config.key[:20]}...{supabase_config.key[-5:]}")
    
    # Validate URL format
    if not supabase_config.url.startswith('https://'):
        print("❌ Invalid URL format (must start with https://)")
        return False
    
    if '.supabase.co' not in supabase_config.url:
        print("❌ Invalid Supabase URL format")
        return False
    
    # Validate key format (JWT)
    if not supabase_config.key.startswith('eyJ'):
        print("❌ Invalid API key format (should be JWT)")
        return False
    
    print("✅ Configuration format looks correct")
    return True

def test_specific_endpoints():
    """Test specific Supabase endpoints"""
    print("\n🎯 Testing Specific Endpoints...")
    print("=" * 50)
    
    base_url = supabase_config.url
    headers = {
        'apikey': supabase_config.key,
        'Authorization': f'Bearer {supabase_config.key}',
        'Content-Type': 'application/json'
    }
    
    endpoints = [
        '/rest/v1/',
        '/rest/v1/users?limit=1',
        '/rest/v1/event_detections?limit=1',
        '/rest/v1/alerts?limit=1'
    ]
    
    for endpoint in endpoints:
        try:
            url = base_url + endpoint
            print(f"📍 Testing: {endpoint}")
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                print(f"  ✅ Success: {response.status_code}")
            elif response.status_code == 404:
                print(f"  ⚠️ Table not found: {response.status_code}")
            else:
                print(f"  ⚠️ Response: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main test function"""
    print("🚀 SUPABASE CONNECTIVITY TEST")
    print("=" * 60)
    
    # Test 1: Configuration
    config_ok = test_supabase_config()
    
    # Test 2: Network connectivity
    network_ok = test_basic_connectivity()
    
    # Test 3: Specific endpoints (only if network is ok)
    if network_ok:
        test_specific_endpoints()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if config_ok and network_ok:
        print("✅ All tests passed! Supabase should work correctly.")
        print("💡 Try running the healthcare pipeline now.")
    elif config_ok:
        print("⚠️ Configuration OK but network issues detected.")
        print("💡 Check firewall, proxy, or network connectivity.")
    else:
        print("❌ Configuration issues detected.")
        print("💡 Check your .env file and Supabase credentials.")

if __name__ == "__main__":
    main()
