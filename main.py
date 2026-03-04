import cv2
import serial
import time
import numpy as np
from gpiozero import LED, OutputDevice
from mpu6050 import mpu6050
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. HARDWARE SETUP
# ==========================================

# GPIO Setup (Using the pins from our diagram)
right_led = LED(22)
# Using OutputDevice for the transistor controlling the motor
right_motor = OutputDevice(25) 

# MPU6050 Setup (I2C)
# The default I2C address is usually 0x68
sensor_mpu = mpu6050(0x68)
MAX_LEAN_ANGLE = 25.0 # Degrees. If leaning more than this, ignore radar.

# mmWave Radar Setup (UART4 - /dev/ttyAMA2 or similar depending on Pi config)
# Adjust baud rate based on your specific mmWave sensor datasheet
right_radar = serial.Serial('/dev/ttyAMA2', baudrate=115200, timeout=0.1)

# USB Camera Setup
# 0 is usually the first USB camera. Try 1 or 2 if you have multiple.
right_cam = cv2.VideoCapture(0)
right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Keep resolution low for speed
right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ==========================================
# 2. MACHINE LEARNING SETUP (TFLite)
# ==========================================

# Load your custom or pre-trained YOLO TFLite model
# (You will need to place your yolov8n.tflite file in the same directory)
interpreter = Interpreter(model_path="yolov8n.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Target classes we care about (COCO dataset indices: 2=car, 3=motorcycle, 5=bus, 7=truck)
THREAT_CLASSES = [2, 3, 5, 7] 

def detect_vehicle(frame):
    """Runs a single frame through the TFLite model and returns True if a vehicle is detected."""
    # Resize and normalize the image for the TFLite model (usually 640x640 or 320x320)
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get results (this parsing depends slightly on your specific YOLO TFLite export format)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Simplified check: loop through detections to see if any match our threat classes with high confidence
    for box in boxes:
        # Assuming format: [x, y, w, h, confidence, class_id]
        if len(box) >= 6:
            confidence = box[4]
            class_id = int(box[5])
            if confidence > 0.5 and class_id in THREAT_CLASSES:
                return True # Threat found!
    return False

# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    print("System Starting... Warming up sensors.")
    time.sleep(2)
    
    try:
        while True:
            # Step 1: Check Lean Angle
            # If leaning hard into a turn, skip radar checks to avoid false positives from the ground
            accel_data = sensor_mpu.get_accel_data()
            # Calculate simple roll angle (tilt left/right)
            roll = np.degrees(np.arctan2(accel_data['y'], accel_data['z']))
            
            if abs(roll) > MAX_LEAN_ANGLE:
                print(f"Leaning ({roll:.1f} deg). Suppressing warnings.")
                right_led.off()
                right_motor.off()
                time.sleep(0.1)
                continue

            # Step 2: Check mmWave Radar
            radar_triggered = False
            if right_radar.in_waiting > 0:
                radar_data = right_radar.readline().decode('utf-8').strip()
                # Your specific mmWave sensor will have a specific data format.
                # Here we assume it sends a keyword like "DETECT" or distance data.
                if "DETECT" in radar_data: 
                    radar_triggered = True
                    print("Radar triggered!")

            # Step 3: Verify with Camera if Radar is triggered
            if radar_triggered:
                # Flush the camera buffer to get the absolute newest frame
                for _ in range(5): right_cam.grab() 
                ret, frame = right_cam.read()
                
                if ret:
                    # Run the ML model
                    is_vehicle = detect_vehicle(frame)
                    
                    if is_vehicle:
                        print("WARNING: Vehicle in Right Blindspot!")
                        right_led.on()
                        right_motor.on()
                    else:
                        print("Radar false alarm (no vehicle seen).")
                        right_led.off()
                        right_motor.off()
            else:
                # Clear warnings if nothing is detected
                right_led.off()
                right_motor.off()

            # Small delay to prevent CPU maxing out
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Shutting down system.")
    finally:
        right_cam.release()
        right_led.off()
        right_motor.off()
        right_radar.close()

if __name__ == '__main__':
    main()