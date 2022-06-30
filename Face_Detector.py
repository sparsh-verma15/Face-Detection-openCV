import cv2;
from random import randrange

#add training data
trained_face_data = cv2.CascadeClassifier('C:\Sparsh Verma\Projects\Python AI\Face Detector\haarcascade_frontalface_default.xml')

#import image
#img = cv2.imread(r'C:\Sparsh Verma\Projects\Python AI\Face Detector\RDJ.jpg')

webcam = cv2.VideoCapture(0)

while True:
    #Read the current frame
    successFul_frame_read, frame = webcam.read()

    #convert to gray
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x,y,w,h) in face_coordinates:
        # if (w<150):
        #     continue
        # if (h<150):
        #     continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        

    cv2.imshow('Webcam, Press Esc to Quit',frame)
    key = cv2.waitKey(1)

    if(key == 27):
        break

webcam.release()



print("Code Executed!!")