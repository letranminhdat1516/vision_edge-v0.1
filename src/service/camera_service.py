from camera.simple_camera import SimpleIMOUCamera

class CameraService:
    def __init__(self, config):
        self.camera = SimpleIMOUCamera(config)
    def connect(self):
        return self.camera.connect()
    def get_frame(self):
        return self.camera.get_frame()
    def disconnect(self):
        self.camera.disconnect()
