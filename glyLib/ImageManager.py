
from email.quoprimime import unquote
import cv2 as cv
from .Point import Point
from .DetectedArea import DetectedArea
from .DetectedArea import DetectedFace
from .HelperFunctions import *
import numpy as np
import os


""" 
    At this point in development, what I need the ImageManager to do is store the image, perform eyes detection on the image in 0, 90, 180, and 270 degree.
    
    Then ImageManager collects all "detected eyes" called DetectedArea objs, which still has the x, y cordinate corresponding to the rotated angle of the image when
    they were found. It then proceed to convert the x, y value to the cordinate in the original non-rotated image.
    
    After that, it checks for overlapping DetectedAreas and merge them into one big DetectedArea obj.

    Once that's done, we get an array of all the location of the eyes and their sizes. We can then connect every possible pair of approximately-same-size-eyes to see 
    if the distance between them are "reasonable". If a pair of eyes has a reasonable distance inbetweeen and the size of the eyes don't differ too much, 
    we can call it a potential face.

    Once we get an array of potential faces, we rotate the image by an angle decided by the position of the two eyes (to increase precision), crop out the face (to increase performance), 
    and run facial detection on that.

    At the end, ImageManger should have an array list of all detected faces.

"""

package_directory = os.path.dirname(os.path.abspath(__file__))
eye_path = os.path.join(package_directory, "classifier/haarcascade_eye.xml")
face_path = os.path.join(package_directory, "classifier/haarcascade_frontalface_default.xml")

class ImageManager:
    """
    A manager that stores the actual image and can do image processing function on it. This object will be used to take care of facial detection.
    """
    
    haarcascade_eye = cv.CascadeClassifier(eye_path)
    haarcascade_face = cv.CascadeClassifier(face_path)
    # haarcascade_face = cv.CascadeClassifier("classifier/lbpcascaade_frontalface_improved.xml")
    # haarcascade_nose = cv.CascadeClassifier("classifier/haarcascade_nose.xml")
    HARDCODED_similarSizeScale = 0.7
    HARDCODED_pairOfEyesDistanceRange = (1.5, 3.5)
    # eye and face min and max dimensions in a 500pixel x ? pixel images
    HARDCODED_eyeMinDimensions = (40, 40)
    HARDCODED_eyeMaxDimensions = (120, 120)
    HARDCODED_faceMinDimensions = (80, 80)
    HARDCODED_faceMaxDimensions = (500, 500)

    def __init__(self, img):
        """
        Constructing an ImageManger object
            :param img: the image/frame that we will run facial detection on
        """
        # Blank canvas that we are going to use to store the rotated image
        self.image = img
        self.grayImage = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.imageCenter = Point(img.shape[1]/2, img.shape[0]/2)
    
    

    
    def HELPER_runHaarDetectionCounterClockwiseAngle(self, detector, minDimensions, maxDimensions, angle, scaleFactor, minNeighbors):
        """
        Run the given haarDetection on the image rotated by the given angle and generate 1 array that 
        contain detected objects found.         
            :param detector: the haarcascade object that is going to scan the image
            :param angle: the angle by which the image is rotated
            :param scaleFactor: scaleFactor parameter for detectMultiScale function
            :param minNeighbors: minNeighbors parameter for detectMultiScale function
            :return a 2-tuple with the first element being an array containing all the raw coordinates of the detected objects in the rotated image,
            and the second element the Point object containing the coordinates of the center of the rotated image.
        """

        angle = angle % 360

        # finding objects in then given angle degree rotated image
       
        rotatedGrayImage = rotateCounterClockwise(self.grayImage, angle)
            
        # Collecting raw detected objects received from detectMultiScale
        rawObjs = detector.detectMultiScale(rotatedGrayImage, scaleFactor = scaleFactor, minNeighbors = minNeighbors, minSize = minDimensions, maxSize = maxDimensions)
            
        detectedAreas = []
        rotatedCenter = Point(rotatedGrayImage.shape[1]/2, rotatedGrayImage.shape[0]/2)

        for obj in rawObjs:
            # Convert raw information into DetectedArea obj
            area = DetectedArea((obj[0], obj[1]), (obj[2], obj[3]))

            # Put the translated DetectedArea into translatedObj array
            detectedAreas.append(area)

        return (detectedAreas, rotatedCenter)    


    def HELPER_rotateDetectedAreaClockwise(self, rawPositions, origin, angle):
        """
        Rotate detectedAreas in rawPositions around the given origin
            :param rawPositions: the array that contains detectedAreas with raw values taken from detectMultiScale
            :param origin: the origin which the detectedAreas are rotating around
            :param angle: the angle by which the image was rotated when detectMultiScale ran
            :return an array containing detectedAreas with translated coordinates in the non-rotated image
        """
        angle = angle % 360
        if angle == 0:
            return rawPositions

        # Translate the coordinates to the coordinates in the non-rotated image
        for area in rawPositions:
            area.rotateAreaClockwise(origin, angle)

        return rawPositions


    def HELPER_projectDetectedArea(self, rawPositions, rotatedCenter):
        """
        Project the raw positions such that the new positions relative to self.imageCenter is the same
        as relative positions of the old Coordinates to rotatedCenter.
            :param rawPositions: the array that contains detectedAreas
            :return an array containing detectedAreas with projected coordinates
        """
        # Translate the coordinates to the coordinates in the non-rotated image
        for area in rawPositions:
            area.projectArea(rotatedCenter, self.imageCenter)

        return rawPositions


    def HELPER_standardizeCounterClockwiseDetectedArea(self, detectedAreas, rotatedCenter, angle):
        """
        Translate the coordinates of counter clockwise rotated detectedAreas to the original image coordinates
            :param detectedAreas: a list of detectedArea objects
            :param rotatedCenter: the rotated image's center's point object
            :param angle: the angle by which the image was rotated counter clockwise by
            :return a list of translated detectedAreas for the nonrotated image
        """

        AllObjects = []    

        self.HELPER_rotateDetectedAreaClockwise(detectedAreas, rotatedCenter, angle)
        self.HELPER_projectDetectedArea(detectedAreas, rotatedCenter)
        AllObjects.append(detectedAreas)

        return AllObjects
        

    def HELPER_mergeDetectedObjs(self, list):
        """
        Given a list of any number of arrays of (detected objects, rotatedCenter), scan through all of them and if find two duplicates (similar detected objects with similar 
        sizes and positions), merge them and put all the unique detected object in an array. 
            :param list: a list of arrays
            :return an array that contains all the unique detected objects.
        """
        array0 = list[0]
        others = []
        for i in range(1,len(list)):
            others.append(list[i])


        # checking duplicates in the first array
        i = 0
        while i < len(array0) - 1:
            j = i + 1
            while j < len(array0):
                if array0[i].similarSize(array0[j], self.HARDCODED_similarSizeScale) and array0[i].overlap(array0[j]):
                        array0[i].merge(array0[j])
                        array0.pop(j)
                        j = j - 1
                j = j + 1
            i = i + 1

        # check for duplicates in the other arrays. Merge them if they are, append them if they arent
        for array in others:
            for area in array:
                matched = False
                # Scan through the array to find matches
                for area0 in array0:
                    if area0.similarSize(area, self.HARDCODED_similarSizeScale) and area0.overlap(area):
                        area0.merge(area)
                        matched = True
                        break
                
                # if didnt find any matches then append it onto array0
                if matched == False:
                    array0.append(area)    
        
        return array0


    def HELPER_findEyesCounterClockwiseAngle(self, angle, scaleFactor, minNeighbors):
        """
        Find the (non-paired) eyes in the counter clockwise rotated image. Merge the duplicates and return them in an array
            :param angle: the angle by which the image is rotated by counter clockwise
            :param scaleFactor: detectMultiscale parameter
            :param minNeighbors: detectMultiscale parameter
            :return an array of detected eyes in the counter clockwise rotated image
        """
        (detectedEyes, rotatedCenter) = self.HELPER_runHaarDetectionCounterClockwiseAngle(self.haarcascade_eye, self.HARDCODED_eyeMinDimensions, self.HARDCODED_eyeMaxDimensions, angle, scaleFactor, minNeighbors)
        eyes = self.HELPER_mergeDetectedObjs(self.HELPER_standardizeCounterClockwiseDetectedArea(detectedEyes, rotatedCenter, angle))
           
        return eyes



    def findPairsOfEyesCounterClockwiseAngle(self, angle, scaleFactor, minNeighbors):
        """
        Find the pairs of eyes in the counter clockwise rotated image. Return them in an array containing 2-tuple with the 
        first element being the left-most eye and the second element the right-most eye.
            :param angle: the angle by which the image is rotated by counter clockwise
            :param scaleFactor: detectMultiscale parameter
            :param minNeighbors: detectMultiscale parameter
            :return an array of pairs of eyes 
        """
        
        eyes = self.HELPER_findEyesCounterClockwiseAngle(angle, scaleFactor, minNeighbors)

        pairOfEyes = []
        
        for i in range(len(eyes) - 1):
            for j in range(i,len(eyes)):
                if eyes[i].similarSize(eyes[j], self.HARDCODED_similarSizeScale):
                    dist = eyes[i].center.distTo(eyes[j].center)
                    averageRadius = (eyes[1].radius + eyes[j].radius)/2
                    if dist < self.HARDCODED_pairOfEyesDistanceRange[1] * averageRadius and dist > self.HARDCODED_pairOfEyesDistanceRange[0] * averageRadius:
                        # Let the left most eye be the first eye. This is for calculating relative angle of the face in findFacesUsingPairOfEyes method
                        if eyes[i].center.x < eyes[j].center.x:
                            pairOfEyes.append((eyes[i], eyes[j]))
                        else:
                            pairOfEyes.append((eyes[j], eyes[i]))

        return pairOfEyes


    def findPairsOfEyesCounterClockwiseMultipleAngles(self, angles, scaleFactor, minNeighbors):
        """
        Find the pairs of eyes in the counter clockwise rotated images. Return them in an array containing 2-tuple with the 
        first element being the left-most eye and the second element the right-most eye.
            :param angles: the angles by which the image is rotated by counter clockwise
            :param scaleFactor: detectMultiscale parameter
            :param minNeighbors: detectMultiscale parameter
            :return an array of pairs of eyes 
        """
        eyes = []
        for angle in angles:
            eyes.append(self.HELPER_findEyesCounterClockwiseAngle(angle, scaleFactor, minNeighbors))

        eyes = self.HELPER_mergeDetectedObjs(eyes)

        pairOfEyes = []
        
        for i in range(len(eyes) - 1):
            for j in range(i,len(eyes)):
                if eyes[i].similarSize(eyes[j], self.HARDCODED_similarSizeScale):
                    if eyes[i].appropriateDistanceTo(eyes[j], self.HARDCODED_pairOfEyesDistanceRange[0], self.HARDCODED_pairOfEyesDistanceRange[1]):
                        # Let the left most eye be the first eye. This is for calculating relative angle of the face in findFacesUsingPairOfEyes method
                        if eyes[i].center.x < eyes[j].center.x:
                            pairOfEyes.append((eyes[i], eyes[j]))
                        else:
                            pairOfEyes.append((eyes[j], eyes[i]))

        return pairOfEyes

 
    def DEBUG_findFacesUsingPairOfEyes(self, pairOfEyes, scaleFactor, minNeighbors):
        """
        Using given pairs of eyes, for each pair find the angle the face is leaning, crop the area the face could be out and run haarDetection on that area.
        Return an array of all detectedAreas encapsulating faces.
            :param pairOfEyes: 2-tuple (left eye, right eye)
            :param scaleFactor: parameter for detectMultiScale
            :param minNeighbors: parameter for detectedMultiScale
            return array of all detected faces
        """

        # For right now, let face width be 6 average radius, and height be 10 average radius with 5 average radiuses from 2 eyes center to the top 
        # 5 average radiuses from 2 eyes center to the chin

        debugArrayFaces = []
        debugPotentialFaceNumber = 1
        for pair in pairOfEyes:
            
            leftEye = pair[0]
            rightEye = pair[1]
            eyeAverageRadius = (leftEye.radius + rightEye.radius)/2
            # halfFaceDimensions store the distance from faceOrigin to the left border, to the upper border, and to the lower border
            # faceMinRadius is the min radius of the rectangle encapsulating the detected Face
            halfFaceDimensions = (eyeAverageRadius * 4, eyeAverageRadius * 5, eyeAverageRadius * 5)
            faceMinRadius = eyeAverageRadius * 3

            # relative angle is the relative angle of the right eye to the left eye, but I limit the ranges from 270 -> 0 -> 90 degree because faces usually aren't up side down, 
            # still this function takes care of that case also (scan it upside down when find no face)
            relativeCounterClockwiseAngle = (leftEye.center.relativeCounterClockwiseAngle(rightEye.center) + 90) % 180 - 90
            originalImageCenter = Point(self.grayImage.shape[1]/2, self.grayImage.shape[0]/2)
            faceOrigin = Point((leftEye.center.x + rightEye.center.x)/2,(leftEye.center.y + rightEye.center.y)/2)
           
            # Rotate the face origin point and the image such that the face is straightened up
            rotatedGrayImage = rotateClockwise(self.grayImage, relativeCounterClockwiseAngle)
            rotatedImageCenter = Point(rotatedGrayImage.shape[1]/2, rotatedGrayImage.shape[0]/2)
            rotatedFaceOrigin = faceOrigin.rotatePointClockwise(originalImageCenter, relativeCounterClockwiseAngle)
            rotatedFaceOrigin = rotatedFaceOrigin.projectPoint(originalImageCenter, rotatedImageCenter)
            
            
            
            # DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- 
            debugOriginalImage = rotateClockwise(self.image, 0)
            leftEye.draw(debugOriginalImage, (255,0,255), 2)
            rightEye.draw(debugOriginalImage, (0,255,255), 2)
            cv.circle(debugOriginalImage, faceOrigin.exportCoordinates(), 0, (0, 255, 0), 20)
            debugRotatedOriginalImage = rotateClockwise(debugOriginalImage, relativeCounterClockwiseAngle)
            cv.circle(debugRotatedOriginalImage, rotatedFaceOrigin.exportCoordinates(), 0, (255, 255, 255), 12)
            # DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- DEBUG step --- 
            


            # Crop the potential face out
            cropRangeMinY = int(max(0, rotatedFaceOrigin.y - halfFaceDimensions[1]))
            cropRangeMaxY = int(min(rotatedGrayImage.shape[0], rotatedFaceOrigin.y + halfFaceDimensions[2]))

            cropRangeMinX = int(max(0, rotatedFaceOrigin.x - halfFaceDimensions[0]))
            cropRangeMaxX = int(min(rotatedGrayImage.shape[1], rotatedFaceOrigin.x + halfFaceDimensions[0]))

            try:
                rotatedCrop = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX), dtype='uint8')
                rotatedCrop[0: rotatedCrop.shape[0], 0: rotatedCrop.shape[1]] = \
                    rotatedGrayImage[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]
             
            except:
                rotatedCrop = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX, 3), dtype ='uint8')
                rotatedCrop[0: rotatedCrop.shape[0], 0: rotatedCrop.shape[1]] = \
                    rotatedGrayImage[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]
   
            try:
                debugRotatedCrop = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX), dtype ='uint8')
                debugRotatedCrop[0: debugRotatedCrop.shape[0], 0: debugRotatedCrop.shape[1]] = debugRotatedOriginalImage[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]
            except:
                debugRotatedCrop = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX, 3), dtype ='uint8')
                debugRotatedCrop[0: debugRotatedCrop.shape[0], 0: debugRotatedCrop.shape[1]] = debugRotatedOriginalImage[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]

        
            rotatedFaceCenter = Point((cropRangeMinX + cropRangeMaxX)/2, (cropRangeMinY + cropRangeMaxY)/2)
            croppedCenter = Point(rotatedCrop.shape[1]/2, rotatedCrop.shape[0]/2)


            # find the face in the cropped Area
            detectedFaces = self.haarcascade_face.detectMultiScale(rotatedCrop, scaleFactor, minNeighbors, minSize = self.HARDCODED_faceMinDimensions, maxSize = self.HARDCODED_faceMaxDimensions)


            # DEBUGING AREA
            debugRotatedCrop = cv.circle(debugRotatedCrop, croppedCenter.exportCoordinates(), 0, (255, 255, 255), 6)

            # DEBUGING AREA


            boolUpSideDown = False

            if len(detectedFaces) == 0:
                # Scan the image upside down in case of upside down faces
                rotatedCrop = rotateClockwise(rotatedCrop, 180)
                detectedFaces = self.haarcascade_face.detectMultiScale(rotatedCrop, scaleFactor, minNeighbors, minSize = self.HARDCODED_faceMinDimensions, maxSize = self.HARDCODED_faceMaxDimensions)
                if len(detectedFaces) == 0:
                    continue
                boolUpSideDown = True


            # merge smaller faces to the biggest face
            biggestFace = detectedFaces[0]
            for i in range(1, len(detectedFaces)):
                # Doesn't have to be too complicated, if one dimension is larger the other is 99% of the time larger as well
                if detectedFaces[i][2] > biggestFace[2]:
                    biggestFace = detectedFaces[i]


            # if face's radius is too small then its not a face (face right now is a 4-tuple (x, y, w, h))
            if biggestFace[2] ** 2 + biggestFace[3] ** 2 < (faceMinRadius * 2) ** 2:

                if boolUpSideDown:
                    continue

                # Scan the image upside down in case of upside down faces
                rotatedCrop = rotateClockwise(rotatedCrop, 180)
                detectedFaces = self.haarcascade_face.detectMultiScale(rotatedCrop, scaleFactor, minNeighbors, minSize = self.HARDCODED_faceMinDimensions, maxSize = self.HARDCODED_faceMaxDimensions)
                if len(detectedFaces) == 0:
                    continue
                boolUpSideDown = True

                
                # merge smaller faces to the biggest face
                biggestFace = detectedFaces[0]
                for i in range(1, len(detectedFaces)):
                    # if one dimension is larger the other is 99% of the time larger as well
                    if detectedFaces[i][2] > biggestFace[2]:
                        biggestFace = detectedFaces[i]

                if biggestFace[2] ** 2 + biggestFace[3] ** 2 < (faceMinRadius * 2) ** 2:
                    continue
            
            # if the face found was upside down, then face angle = eye's relative angle + 180
            if boolUpSideDown:
                counterClockwiseFaceAngle = relativeCounterClockwiseAngle + 180
            # if not then = eye's relative angle
            else:
                counterClockwiseFaceAngle = relativeCounterClockwiseAngle

            biggestFace = DetectedFace((biggestFace[0],biggestFace[1]), (biggestFace[2], biggestFace[3]), counterClockwiseFaceAngle)

            
            if boolUpSideDown:
                biggestFace.rotateAreaClockwise(croppedCenter, 180)

            # Convert biggestFace coordinates from being in the cropped image to the original image
            biggestFace.draw(debugRotatedCrop, (0, 0, 255), 2)
            cv.imshow("Potential face " + str(debugPotentialFaceNumber), resizeMinTo(debugRotatedCrop, 250))

            debugArrayFaces.append((biggestFace.copy(), rotatedFaceCenter, croppedCenter, rotatedImageCenter, relativeCounterClockwiseAngle, originalImageCenter)) 
                
            
            debugPotentialFaceNumber = debugPotentialFaceNumber + 1


        return debugArrayFaces


    def findFacesUsingPairOfEyes(self, pairOfEyes, scaleFactor, minNeighbors):
        """
        Using given pairs of eyes, for each pair of similar-size eyes, find the angle the face is leaning, crop the area the face could be out and run haarDetection on that area.
        Return an array of all detectedAreas encapsulating faces.
            :param pairOfEyes: 2-tuple (left eye, right eye)
            :param scaleFactor: parameter for detectMultiScale
            :param minNeighbors: parameter for detectedMultiScale
            return array of all detected faces
        """

        faces = [] 

        # For right now, let face width be 6 average radius, and height be 10 average radius with 5 average radiuses from 2 eyes center to the top 
        # 5 average radiuses from 2 eyes center to the chin. I'm going to make the height ratio 3 top:5 bottom once i added mouth/nose detection for orientation
        
        for pair in pairOfEyes:
            
            leftEye = pair[0]
            rightEye = pair[1]
            eyeAverageRadius = (leftEye.radius + rightEye.radius)/2
            # halfFaceDimensions store the distance from faceOrigin to the left border, to the upper border, and to the lower border
            # faceMinRadius is the min radius of the rectangle encapsulating the detected Face
            halfFaceDimensions = (eyeAverageRadius * 4, eyeAverageRadius * 5, eyeAverageRadius * 5)
            faceMinRadius = eyeAverageRadius * 3

            # relative angle is the relative angle of the right eye to the left eye, but I limit the ranges from 270 -> 0 -> 90 degree because faces usually aren't up side down, 
            # still this function takes care of that case also (scan it upside down when find no face)
            relativeCounterClockwiseAngle = (leftEye.center.relativeCounterClockwiseAngle(rightEye.center) + 90) % 180 - 90
            originalImageCenter = Point(self.grayImage.shape[1]/2, self.grayImage.shape[0]/2)
            faceOrigin = Point((leftEye.center.x + rightEye.center.x)/2,(leftEye.center.y + rightEye.center.y)/2)
           
            # Rotate the face origin point and the image such that the face is straightened up
            rotatedGrayImage = rotateClockwise(self.grayImage, relativeCounterClockwiseAngle)
            rotatedImageCenter = Point(rotatedGrayImage.shape[1]/2, rotatedGrayImage.shape[0]/2)
            rotatedFaceOrigin = faceOrigin.rotatePointClockwise(originalImageCenter, relativeCounterClockwiseAngle).projectPoint(originalImageCenter, rotatedImageCenter)

            # Crop the potential face out
            cropRangeMinY = int(max(0, rotatedFaceOrigin.y - halfFaceDimensions[1]))
            cropRangeMaxY = int(min(rotatedGrayImage.shape[0], rotatedFaceOrigin.y + halfFaceDimensions[2]))

            cropRangeMinX = int(max(0, rotatedFaceOrigin.x - halfFaceDimensions[0]))
            cropRangeMaxX = int(min(rotatedGrayImage.shape[1], rotatedFaceOrigin.x + halfFaceDimensions[0]))

            try:
                rotatedCrop = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX), dtype='uint8')
                rotatedCrop[0: rotatedCrop.shape[0], 0: rotatedCrop.shape[1]] = \
                    rotatedGrayImage[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]
             
            except:
                rotatedCrop = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX, 3), dtype ='uint8')
                rotatedCrop[0: rotatedCrop.shape[0], 0: rotatedCrop.shape[1]] = \
                    rotatedGrayImage[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]
        
            rotatedFaceCenter = Point((cropRangeMinX + cropRangeMaxX)/2, (cropRangeMinY + cropRangeMaxY)/2)
            croppedCenter = Point(rotatedCrop.shape[1]/2, rotatedCrop.shape[0]/2)


            # find the face in the cropped Area
            detectedFaces = self.haarcascade_face.detectMultiScale(rotatedCrop, scaleFactor, minNeighbors, minSize = self.HARDCODED_faceMinDimensions, maxSize = self.HARDCODED_faceMaxDimensions)
            boolUpSideDown = False


            # if found no faces right side up
            if len(detectedFaces) == 0:
                # Scan the image upside down
                rotatedCrop = rotateClockwise(rotatedCrop, 180)
                detectedFaces = self.haarcascade_face.detectMultiScale(rotatedCrop, scaleFactor, minNeighbors, minSize = self.HARDCODED_faceMinDimensions, maxSize = self.HARDCODED_faceMaxDimensions)
                if len(detectedFaces) == 0:
                    continue
                boolUpSideDown = True


            # merge smaller faces to the biggest face
            biggestFace = detectedFaces[0]
            for i in range(1, len(detectedFaces)):
                # Doesn't have to be too complicated, if one dimension is larger the other is 99% of the time larger as well
                if detectedFaces[i][2] > biggestFace[2]:
                    biggestFace = detectedFaces[i]


            # if face's radius is too small then its not a face (face right now is a 4-tuple (x, y, w, h))
            if biggestFace[2] ** 2 + biggestFace[3] ** 2 < (faceMinRadius * 2) ** 2:

                if boolUpSideDown:
                    continue

                # Scan the image upside down in case of upside down faces
                rotatedCrop = rotateClockwise(rotatedCrop, 180)
                detectedFaces = self.haarcascade_face.detectMultiScale(rotatedCrop, scaleFactor, minNeighbors, minSize = self.HARDCODED_faceMinDimensions, maxSize = self.HARDCODED_faceMaxDimensions)
                if len(detectedFaces) == 0:
                    continue
                boolUpSideDown = True

                
                # merge smaller faces to the biggest face
                biggestFace = detectedFaces[0]
                for i in range(1, len(detectedFaces)):
                    # if one dimension is larger the other is 99% of the time larger as well
                    if detectedFaces[i][2] > biggestFace[2]:
                        biggestFace = detectedFaces[i]

                # if face's radius is too small then its not a face (face right now is a 4-tuple (x, y, w, h))
                if biggestFace[2] ** 2 + biggestFace[3] ** 2 < (faceMinRadius * 2) ** 2:
                    continue



            biggestFace = DetectedFace((biggestFace[0],biggestFace[1]), (biggestFace[2], biggestFace[3]))
            
            
            


            if boolUpSideDown:
                biggestFace.rotateAreaClockwise(croppedCenter, 180)

            # Convert biggestFace coordinates from being in the cropped image to the original image
            biggestFace.projectArea(croppedCenter, rotatedFaceCenter)
            biggestFace.rotateAreaCounterClockwise(rotatedImageCenter, relativeCounterClockwiseAngle)
            biggestFace.projectArea(rotatedImageCenter, originalImageCenter)


            
            # if the face found was upside down, then face angle = eye's relative angle + 180, the eyes are swapped
            if boolUpSideDown:
                counterClockwiseFaceAngle = (relativeCounterClockwiseAngle + 180) % 360
                biggestFace.counterClockwiseAngle = counterClockwiseFaceAngle
                biggestFace.leftEye = rightEye
                biggestFace.rightEye = leftEye
            # if not then = eye's relative angle, the eyes are in the same order
            else:
                counterClockwiseFaceAngle = relativeCounterClockwiseAngle
                biggestFace.counterClockwiseAngle = counterClockwiseFaceAngle
                biggestFace.leftEye = leftEye
                biggestFace.rightEye = rightEye
            

            faces.append(biggestFace)

        return faces


    def findFacesCounterClockwiseAngle(self, angle, scaleFactor, minNeighbors):
        """
        Find faces in the image and return them as detectedArea objects in an array
            :param angle: counter clockwise angle by which the image is rotated
            :param scaleFactor: paramter for detectMultiScale
            :param minNeighbors: parameter for detectMultiScale
            :return an array of faces as detectedFace objects
        """
        return self.findFacesUsingPairOfEyes(self.findPairsOfEyesCounterClockwiseAngle(angle, scaleFactor, minNeighbors), scaleFactor, minNeighbors)


    def findFacesCounterClockwiseMultipleAngles(self, angles, scaleFactor, minNeighbors):
        """
        Find faces in the image and return them as detectedArea objects in an array
            :param angles: counter clockwise angles by which the image is rotated
            :param scaleFactor: paramter for detectMultiScale
            :param minNeighbors: parameter for detectMultiScale
            :return an array of faces as detectedFace objects
        """
        return self.findFacesUsingPairOfEyes(self.findPairsOfEyesCounterClockwiseMultipleAngles(angles, scaleFactor, minNeighbors), scaleFactor, minNeighbors)

    

    def findFacesCounterClockwiseMultipleAngles_Alternative(self, angles, scaleFactor, minNeighbors):
        """
        Find faces in the image and return them as detectedArea objects in an array
        This method, unlike the other one, just going to scan for faces in multiple angles instead of eyes. Then merge them together. Scanning for eyes is only for finding the faces' rotations
            :param angles: counter clockwise angles by which the image is rotated
            :param scaleFactor: paramter for detectMultiScale
            :param minNeighbors: parameter for detectMultiScale
            :return an array of faces as detectedFace objects
        """     

        
        detectedFaces = []

        for angle in angles:
            (detectedAreas, rotatedCenter) = self.HELPER_runHaarDetectionCounterClockwiseAngle(self.haarcascade_face, self.HARDCODED_faceMinDimensions, self.HARDCODED_faceMaxDimensions, angle, scaleFactor, minNeighbors)
            
            print(detectedAreas)
            for area in detectedAreas:
                face = DetectedFace((area.upperLeft.x, area.upperLeft.y), area.dimensions, angle)
                face.rotateAreaClockwise(rotatedCenter, angle)

                unique = True
                for detectedFace in detectedFaces:
                    if face.similarSize(detectedFace) and face.overlap(detectedFace):
                        unique = False
                        break
                
                if unique:
                    detectedFaces.append(face)
        
        return detectedFaces

                











