import numpy as np
import cv2

# Initialize the webcam feed
capture = cv2.VideoCapture(0)

# Main Logic
while True:

    # Start reading the webcam feed frame by frame
    ret, frame = capture.read()
    if not ret:
        break

    # Convert BGR image to HSV image
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set the lower and upper HSV ranges for skin color detection
    lower_range = np.array([0, 10, 60], dtype = "uint8")
    upper_range = np.array([20, 150, 255], dtype="uint8")

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsv_frame, lower_range, upper_range)

    # Visualize the real part of the target color
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # show the skin in the image along with the mask
    cv2.imshow("images", np.hstack([frame, res]))

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()