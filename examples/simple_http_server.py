#!/usr/bin/env python3
"""
Simple HTTP Server for serving alert images and mobile client
"""

import http.server
import socketserver
import os
import threading
from pathlib import Path

class AlertImageServer:
    """Simple HTTP server to serve alert images and mobile client"""
    
    def __init__(self, port=8080, directory="examples"):
        self.port = port
        self.directory = Path(directory).resolve()
        self.server = None
        self.thread = None
        
    def start_server(self):
        """Start the HTTP server in a separate thread"""
        os.chdir(self.directory)
        
        handler = http.server.SimpleHTTPRequestHandler
        
        # Custom handler to serve files with proper CORS headers
        class CORSHandler(handler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                super().end_headers()
        
        self.server = socketserver.TCPServer(("", self.port), CORSHandler)
        
        def run_server():
            print(f"🌐 HTTP Server started on http://localhost:{self.port}")
            print(f"📱 Mobile client: http://localhost:{self.port}/mobile_client.html")
            print(f"🖼️ Alert images: http://localhost:{self.port}/data/saved_frames/alerts/")
            self.server.serve_forever()
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
    def stop_server(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("🛑 HTTP Server stopped")

if __name__ == "__main__":
    server = AlertImageServer()
    try:
        server.start_server()
        input("Press Enter to stop the server...\n")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_server()
