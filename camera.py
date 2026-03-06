import cv2
def find_cameras(limit=10):
    """
    Scans for available camera indices.
    Returns a list of working camera indices.
    """
    available_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera found at index {i}")
            cap.release()
    return available_cameras

class Camera:
    def __init__(self, index=0, width=320, height=240):
        self.index = index
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            print(f"Error: Could not open camera at index {self.index}")

    def get_frame(self):
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    print("Scanning for cameras...")
    cameras = find_cameras()
    if not cameras:
        print("No cameras detected.")
    else:
        print(f"Available camera indices: {cameras}")
