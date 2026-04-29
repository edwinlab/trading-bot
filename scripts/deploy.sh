#!/bin/bash
# =============================================================================
# Deploy AI Trading Bot to Tencent Cloud Lighthouse (Ubuntu/Debian)
# =============================================================================
# Run this script on your VPS:
#   curl -fsSL https://raw.githubusercontent.com/YOUR_REPO/deploy.sh | bash
#   OR
#   scp deploy.sh user@your-vps-ip:~ && ssh user@your-vps-ip './deploy.sh'
# =============================================================================

set -euo pipefail

APP_DIR="/opt/tradding-bot"
APP_USER="botuser"
PYTHON_VERSION="3.11"

echo "============================================="
echo "  AI Trading Bot — VPS Deployment"
echo "  Target: Tencent Cloud Lighthouse 2C"
echo "============================================="

# --- Step 1: System updates ---
echo "[1/7] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# --- Step 2: Install Python and dependencies ---
echo "[2/7] Installing Python ${PYTHON_VERSION} and build tools..."
sudo apt-get install -y -qq \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    wget \
    curl \
    htop

# --- Step 3: Create app user ---
echo "[3/7] Creating application user..."
if ! id "${APP_USER}" &>/dev/null; then
    sudo useradd -r -m -s /bin/bash "${APP_USER}"
fi

# --- Step 4: Setup application directory ---
echo "[4/7] Setting up application directory..."
sudo mkdir -p "${APP_DIR}"
sudo chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"

# Copy project files (assumes this script is run from the project root)
if [ -f "user_data/config.json" ]; then
    sudo cp -r user_data/ "${APP_DIR}/"
    sudo cp -r scripts/ "${APP_DIR}/"
    sudo cp requirements.txt "${APP_DIR}/"
    sudo chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
fi

# --- Step 5: Create virtualenv and install Freqtrade ---
echo "[5/7] Installing Freqtrade + FreqAI in virtualenv..."
sudo -u "${APP_USER}" bash -c "
    cd ${APP_DIR}
    python${PYTHON_VERSION} -m venv venv
    source venv/bin/activate
    pip install --upgrade pip wheel setuptools
    pip install freqtrade[freqai]
    pip install -r requirements.txt
"

# --- Step 6: Install systemd service ---
echo "[6/7] Installing systemd service..."
sudo cp systemd/trading-bot.service /etc/systemd/system/trading-bot.service 2>/dev/null || \
    echo "   ⚠️  No systemd service file found. Copy it manually."
sudo systemctl daemon-reload
sudo systemctl enable trading-bot

# --- Step 7: Setup .env ---
echo "[7/7] Setting up environment..."
if [ ! -f "${APP_DIR}/.env" ]; then
    sudo cp .env.example "${APP_DIR}/.env" 2>/dev/null || \
        echo "   ⚠️  No .env.example found. Create ${APP_DIR}/.env manually."
    sudo chown "${APP_USER}:${APP_USER}" "${APP_DIR}/.env"
    sudo chmod 600 "${APP_DIR}/.env"
    echo ""
    echo "⚠️  IMPORTANT: Edit ${APP_DIR}/.env with your API keys!"
    echo "   sudo nano ${APP_DIR}/.env"
fi

echo ""
echo "============================================="
echo "  ✅ Deployment Complete!"
echo "============================================="
echo ""
echo "  Next steps:"
echo "  1. Edit API keys:     sudo nano ${APP_DIR}/.env"
echo "  2. Edit config:       sudo nano ${APP_DIR}/user_data/config.json"
echo "     - Set exchange key/secret (or use .env injection)"
echo "     - Set telegram token/chat_id"
echo "  3. Start bot:         sudo systemctl start trading-bot"
echo "  4. View logs:         sudo journalctl -u trading-bot -f"
echo "  5. Check status:      sudo systemctl status trading-bot"
echo ""
echo "  FreqUI dashboard:     http://YOUR_VPS_IP:8080"
echo "  (Make sure port 8080 is open in Tencent Cloud firewall)"
echo ""
echo "  ⚠️  Start with dry_run=true in config.json!"
echo "============================================="
