"""
app.py
======
Production entry-point using eventlet for async SocketIO support.
Run this file on the Raspberry Pi instead of main.py directly.

Usage
-----
  python app.py
"""

import eventlet
eventlet.monkey_patch()

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

os.environ["FLASK_APP"] = "app.py"

# Import after monkey-patch
from main import app, socketio, sensor_loop   # noqa: E402
import threading

if __name__ == "__main__":
    # Launch background sensor + fusion loop
    bg = threading.Thread(target=sensor_loop, daemon=True, name="sensor-loop")
    bg.start()

    # Serve the Flask-SocketIO dashboard
    print("BlindGuard Dashboard running at http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
