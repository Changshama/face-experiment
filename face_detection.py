import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = np.copy(colored_img)
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    if len(faces) < 1:
      return None
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img_copy

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def face_detect_img_set(directory, face_detector):
  detected_faces = 0
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    test2 = cv2.imread(os.path.join(directory, filename))
    
    faces_detected_img = detect_faces(face_detector, test2)
    if faces_detected_img is None:
      continue
    detected_faces += 1
    print(filename)
    # #conver image to RGB and show image
    # plt.imshow(convertToRGB(faces_detected_img))
  return detected_faces, len(os.listdir(directory))

# file folder containing the video frames
directory = '/content/face-experiment/images/video_frames/2'
haar_face_cascade = cv2.CascadeClassifier('/content/face-experiment/data/haarcascade_frontalface_alt.xml')
lbp_face_cascade = cv2.CascadeClassifier('/content/face-experiment/data/lbpcascade_frontalface.xml')


t1 = time.time()
detected_faces, total_faces = face_detect_img_set(directory, haar_face_cascade)
t2 = time.time()

print(f'haar_face_cascade detected faces: {detected_faces} \n {t2-t1} spent in detecting {total_faces} frames')

t1 = time.time()
detected_faces, total_faces = face_detect_img_set(directory, lbp_face_cascade)
t2 = time.time()

print(f'lbp_face_cascade detected faces: {detected_faces} \n {t2-t1} spent in detecting {total_faces} frames')