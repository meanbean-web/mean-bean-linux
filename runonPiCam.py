import dlib
import cv2
import time
import numpy
from picamera import PiCamera
from picamera.array import PiRGBArray

# LOAD TRAINED DETECTOR
file_name = 'Hand_Detector.svm'
detector = dlib.simple_object_detector(file_name)

# DETECTION SETTINGS
scale_factor = 2.0 #downscaling size, for faster detection; set to 1.0 if no detection
size, center_x = 0, 0 #initial size & center x point of the hand = 0

#initialize variables for calculating FPS
fps = 0
image_counter = 0
start_time = time.time()

# DECLARE PICAMERA
camera = PiCamera ()
camera.resolution = (840, 480)
camera.vflip = True
camera.framerate = 32
rawCapture = PiRGBArray(camera, size= (840, 480))
time.sleep(0.5) #allow picamera to warm up

# CAPTURE CONTINUOUS STREAM

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array #store each incoming image as a numpy array
    
   # cv2.imshow("Frame", image)

   # while (True):

        # Read frame by frame
        # cv2.imshow ("Frame", image)

        # Laterally flip the frame
    image = cv2.flip(image, 1)

        # Calculate the Average FPS
    image_counter += 1
    fps = (image_counter / (time.time() - start_time))

        # Create a clean copy of the frame
    copy = image.copy()

        # Downsize the frame.
    new_width = int(image.shape[1] / scale_factor)
    new_height = int(image.shape[0] / scale_factor)
    resized_image = cv2.resize(copy, (new_width, new_height))

        # Detect with detector
    detections = detector(resized_image)

        # Loop for each detection.
    for detection in (detections):
            # Since we downscaled the image we will need to resacle the coordinates according to the original image.
            x1 = int(detection.left() * scale_factor)
            y1 = int(detection.top() * scale_factor)
            x2 = int(detection.right() * scale_factor)
            y2 = int(detection.bottom() * scale_factor)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, 'Hand Detected', (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

            # Calculate size of the hand.
            size = int((x2 - x1) * (y2 - y1))

            # Extract the center of the hand on x-axis.
            center_x = x2 - x1 // 2

        # Display FPS and size of hand
    cv2.putText(image, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

        # This information is useful for when you'll be building hand gesture applications
    cv2.putText(image, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.putText(image, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))

        # Display the image
    cv2.imshow('frame', image)

    #cv2.imshow("Frame", image) #display each new incoming image
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0) #clear stream for next frame

    if key == ord("q"):
        break


image.release()
cv2.destroyAllWindows()
