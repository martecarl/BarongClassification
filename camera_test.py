import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible.")
else:
    ret, frame = cap.read()
    if ret:
        print("Camera is working!")
    else:
        print("Failed to capture a frame.")
cap.release()
