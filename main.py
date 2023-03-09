
'''
    EMOTION DETECTION IN REAL TIME

    Notes:

        I am using the FER 2013 Dataset for this project.

        The known imbalance problem was solved by making sure there is equal dataset for each emotion.

        Images where the face is covered with hands or other things will be hard to detect. This is known as the occlusion problem.

        There are some images with a contrast variation problem where they may be too dark or too light for the computer to detect.

        Images with faces wearing glasses will also be a difficulty for the computer.

        To draw a rectangle around a detected face, we use haar (an algorithm) which is part of matplotlib. It is only used to detect a face, not emotions.

'''

import cv2                                          # pip install opencv
from matplotlib import pyplot as plt                # pip install matplotlib
from deepface import DeepFace                       # pip install deepface
import time

path = 'C://Users//Aimee//Documents//C - 2nd Year Uni//PERSONAL CODING PROJECTS//Python Projects//Project 3 - Emotion Detection In Real Time//recordings//'

def recognise():

    # detecting all faces (1 or more) in given image.
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # initiating the video camera
    video = cv2.VideoCapture(0)
    video.set(3,640)    # defined resolution, parameters = width,height
    width = video.get(3)
    height = video.get(4)
    print("Video resolution is set to {} x {}".format(width, height))
    print("Help-- \nPress z to exit.")

    while video.isOpened():

        # takes a single image from the video and reads it
        ret,frame = video.read()

        try:
            prediction = DeepFace.analyze(frame,actions=['emotion'])[0]

            # we convert the image into gray so we can then draw a rectangle.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)

            # drawing a rectangle around the face detected.
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                              2)  # (0,255,0) is BGR so the rectangle will be shown in green. The 2 after this BGR is the width of the line.

            # putting the emotion as text above the face detected.
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,
                        prediction['dominant_emotion'], # the text i want to show on image
                        (50,50),    # this is row and column
                        font, 3,
                        (0,0,255),  # this is BGR so it will be red colour
                        2,
                        cv2.LINE_4);

            # displaying the window.
            cv2.imshow("Aimee Silver", frame)   # creates a frame/window and opens it.
        except ValueError:
            print("Cannot detect face at the moment.")

        # closes window if z is pressed.
        if cv2.waitKey(2) & 0xFF  == ord("z"):
            print("Window closed.")
            break

    video.release()
    cv2.destroyAllWindows()

recognise()