import eventlet
eventlet.monkey_patch()

from main import app, socketio, sensor_loop
import threading
import os

# Ensure the Flask environment is aware of where we are
os.environ['FLASK_APP'] = 'app.py'

def run_background_logic():
    """Start the background sensor and processing loop."""
    print("Starting background sensor loop...")
    sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
    sensor_thread.start()

if __name__ == '__main__':
    # Start the hardware/sensor background thread
    run_background_logic()
    
    # Run the Flask-SocketIO server
    # Port 5000 as requested
    print("Starting BlindGuard Dashboard at http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)

