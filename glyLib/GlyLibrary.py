import json
import cv2 as cv
from numpy import append
from .DetectedArea import DetectedArea
# from .DetectedArea import DetectedFace
from .ImageManager import ImageManager
from .Point import Point
from .HelperFunctions import *
import os
import sys




def findAndDrawFacesVanilla(frame):
    """
    Create a grayscaled version of the image, find the faces, and draw them on the original image
    All using vanilla facial detection that came with opencv
    """
    package_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(package_directory, "classifier/haarcascade_frontalface_default.xml")
    
    # print("DEBUG", file=sys.stderr)
    # print(path, file=sys.stderr)
    haar_cascasde_face = cv.CascadeClassifier(path)



    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haar_cascasde_face.detectMultiScale(
        grayFrame, 1.1, 10, minSize=(120, 120))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    return frame

      

def findAndDrawFacesVanilla_coordinates(frame):
    """
    Create a grayscaled version of the image, find the faces, and draw them on the original image
    All using vanilla facial detection that came with opencv
    """
    package_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(package_directory, "classifier/haarcascade_frontalface_default.xml")
    # print("DEBUG", file=sys.stderr)
    # print(path, file=sys.stderr)
    # return None
    haar_cascasde_face = cv.CascadeClassifier(path)



    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haar_cascasde_face.detectMultiScale(
        grayFrame, 1.1, 10, minSize=(120, 120))
    # Convert to a list that is json serializable
    if len(faces) == 0:
        return None
    faces = faces.tolist()
    return faces
        

      

      

  


  