import cv2 as cv

#Get Image
img = cv.imread('Faces/group2.jpg')
cv.imshow('Group', img)
#Convert Image to GrayScale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
#Read XML
haar_cascade = cv.CascadeClassifier('haar_face.xml')
#Detect Face
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
#Print Number of faces found
print(f'Number of faces found = {len(faces_rect)}')
#Draw rectangle
for(x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y),(x+w, y+h),(0,255,0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)