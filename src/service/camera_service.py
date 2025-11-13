from camera.simple_camera import SimpleIMOUCamera

class CameraService:
    def __init__(self, config):
        self.camera = SimpleIMOUCamera(config)
        # 🔥 Store RTSP URL for 5-snapshot capture
        self.rtsp_url = config.get('url', '')
        
    def connect(self):
        return self.camera.connect()
    def get_frame(self):
        return self.camera.get_frame()
    def disconnect(self):
        self.camera.disconnect()
