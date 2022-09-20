import cv2
import face_recognition

img = cv2.imread('images/messi-768x512.jpeg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread('images/messi-test.png')
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

img3 = cv2.imread('images/messi-sideface.png')
rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img_encoding3 = face_recognition.face_encodings(rgb_img3)[0]

img4 = cv2.imread('images/messi-neg.png')
rgb_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
img_encoding4 = face_recognition.face_encodings(rgb_img4)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
score = 1- face_recognition.face_distance([img_encoding], img_encoding2)
print(f"Messi or not: {result}\n Score: {score}")

result = face_recognition.compare_faces([img_encoding], img_encoding3)
score = 1- face_recognition.face_distance([img_encoding], img_encoding3)
print(f"Messi or not: {result}\n Score: {score}")

result = face_recognition.compare_faces([img_encoding], img_encoding4)
score = 1- face_recognition.face_distance([img_encoding], img_encoding4)
print(f"Messi or not: {result}\n Score: {score}")