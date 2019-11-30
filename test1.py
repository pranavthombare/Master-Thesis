import face_recognition
import cv2

# load an image as an arrary
image = face_recognition.load_image_file("sample_2.jpg")

# detect faces from input image.
face_locations = face_recognition.face_locations(image, model="hog")

for (top,right,bottom,left), landmarks in zip(face_locations,face_landmarks):
    cv2.rectangle(image,(left,bottom),(right,top),(255,0,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
