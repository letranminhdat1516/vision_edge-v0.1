# Vision Edge Healthcare System - Raspberry Pi Deployment Guide

## 📋 Prerequisites

### Hardware Requirements

- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended)
- **MicroSD Card**: 32GB minimum (Class 10 or better)
- **Camera**: USB webcam or Pi Camera Module
- **Audio**: Bluetooth speaker or USB speaker
- **Network**: Ethernet or WiFi connection
- **Power**: Official 5V/3A power supply

### Software Requirements

- **OS**: Raspberry Pi OS (64-bit) Bullseye or later
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+

## 🚀 Installation Steps

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y git curl
```

### 2. Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER

# Enable Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Logout and login again for group changes to take effect
```

### 3. Install Docker Compose

```bash
# Install Docker Compose plugin
sudo apt-get install docker-compose-plugin -y

# Verify installation
docker compose version
```

### 4. Clone or Transfer Project

**Option A: From Git**

```bash
cd ~
git clone <your-repo-url> vision_edge
cd vision_edge
```

**Option B: Transfer via SCP**

```bash
# On your computer:
scp -r vision_edge-v0.1/ pi@<pi-ip-address>:~/vision_edge/
```

### 5. Configure Environment

```bash
cd ~/vision_edge

# Create .env file
nano .env
```

**Required environment variables:**

```bash
# Database (Supabase)
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=your_supabase_host
DB_NAME=postgres
DB_PORT=5432

# User ID
DEFAULT_USER_ID=your_user_id

# Optional: Audio settings
AUDIO_VOLUME=0.8
```

### 6. Deploy Application

```bash
# Make deploy script executable
chmod +x deploy_pi.sh

# Run deployment
./deploy_pi.sh
```

## 📊 Managing the System

### Start Services

```bash
docker compose up -d
```

### Stop Services

```bash
docker compose down
```

### View Logs

```bash
# All logs
docker compose logs -f

# Specific service
docker compose logs -f vision_edge
```

### Restart Service

```bash
docker compose restart vision_edge
```

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 🔧 Troubleshooting

### Container Won't Start

**Check logs:**

```bash
docker compose logs vision_edge
```

**Common issues:**

- Camera not accessible: Check `/dev/video0` permissions
- Audio not working: Verify bluetooth speaker connection
- Memory issues: Reduce resolution or increase swap

### Camera Access Issues

```bash
# Check camera device
ls -l /dev/video*

# Add user to video group
sudo usermod -aG video $USER

# Test camera
v4l2-ctl --list-devices
```

### Audio Issues

```bash
# Check audio devices
aplay -l

# Test bluetooth
bluetoothctl
> scan on
> pair <device-mac>
> connect <device-mac>
```

### Memory Optimization

```bash
# Increase swap size
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Restart container with memory limits
docker compose restart
```

## 🎯 Performance Optimization

### 1. Reduce Video Resolution

Edit camera config in database or code:

```python
resolution = (1280, 720)  # Instead of 1920x1080
```

### 2. Lower FPS

```python
fps = 15  # Instead of 30
```

### 3. Use Smaller Models

- Use `yolov8n.pt` instead of `yolov8s.pt`
- Disable BLIP captioning if not needed

### 4. Disable GUI (Headless Mode)

Comment out `cv2.imshow()` calls in `main.py`

## 🔒 Security Recommendations

1. **Change default passwords**
2. **Enable firewall:**
   ```bash
   sudo ufw enable
   sudo ufw allow 8000/tcp
   ```
3. **Use HTTPS** for API endpoints
4. **Secure SSH:**
   ```bash
   sudo nano /etc/ssh/sshd_config
   # Set PasswordAuthentication no
   # Use SSH keys instead
   ```

## 📱 Accessing the System

### API Endpoint

```
http://<pi-ip-address>:8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

### API Documentation

```
http://<pi-ip-address>:8000/docs
```

## 🔄 Auto-Start on Boot

Service is configured with `restart: unless-stopped` in docker-compose.yml

To ensure Docker starts on boot:

```bash
sudo systemctl enable docker
```

## 📊 Monitoring

### Check Resource Usage

```bash
# Container stats
docker stats vision_edge_healthcare

# System resources
htop

# Disk usage
df -h
```

### Temperature Monitoring

```bash
# Check CPU temperature
vcgencmd measure_temp

# Install monitoring tool
sudo apt-get install rpi-monitor
```

## 🆘 Support

If issues persist:

1. Check logs: `docker compose logs -f`
2. Verify network connectivity to Supabase
3. Test camera: `raspistill -o test.jpg` (Pi Camera)
4. Check system resources: `free -h` and `df -h`

## 📚 Additional Resources

- [Docker on Raspberry Pi](https://docs.docker.com/engine/install/debian/)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [OpenCV on Pi](https://qengineering.eu/install-opencv-on-raspberry-pi-4.html)
