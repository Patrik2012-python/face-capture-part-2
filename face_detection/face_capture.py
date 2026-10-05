import cv2
import sys
import os
import time
#stores the ml model
ml_model='face_detection/haarcascade_frontalface_default.xml'
#stores the data sets folder
data_sets='face_detection/data_sets'
#this will be the face that will be unlocked for face id
sub_folder='patrik'
#exact path of the image
path=os.path.join(data_sets,sub_folder)
#width
width=130
#height
height=100
#applying machine learning model to facecapture
facecapture=cv2.CascadeClassifier(ml_model)
#opens the webcam
webcam=cv2.VideoCapture(0)
#if it doesnt open, error will be printed.
if not webcam.isOpened():
    print('cannot open webcam.')
    sys.exit()

#keep count of images.
count=1
while count <30:
    # checks if webcam is existing and reads the image and returns the actual image that will be stored in the im variable
    (_, im)= webcam.read()
    #convert image to greyscale
    grey=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #detects the face in the grey color shades
    faces=facecapture.detectMultiScale(grey,1.3,4)

    for (x,y,w,h) in faces:
        #create a rectangle around the face, x related to w, y related to h, light blue color,thickness of rectangle
        cv2.rectangle(im,(x,y),(x+w,y+h),( 255,247,5),10)
        #measure the x and y coordinates and the w and h in greyscale
        face= grey[y:y+h, x:x+w]
        #resizing image as for the width and height of the face
        face_resize= cv2.resize(face,(width,height))
        #saving the face resized image and taking value of count to give the file name on its own
        cv2.imwrite('% s/% s.png'%(path,count),face_resize)
        count=count+1
    cv2.imshow('facecapture',im)
    cv2.waitKey(10)
webcam.release()
cv2.destroyAllWindows()


