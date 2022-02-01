from distutils.log import debug
from threading import Thread, Lock
from turtle import left
from .FrameThread import FrameThread
from .ImageManager import ImageManager
from .HelperFunctions import *
from .Point import Point
from .DetectedArea import DetectedArea, DetectedFace
import time
import queue
import json
import copy

HARDCODED_eyeMinDimensions = (10, 10)
HARDCODED_eyeMaxDimensions = (120, 120)
HARDCODED_faceMinDimensions = (60, 60)
HARDCODED_faceMaxDimensions = (300, 300)


class ImprovedEnhancedServerVideoManager:
    def __init__(self, frameRate, socket, clientID, coordinates):
        self.coordinates = coordinates

        # Used for the emitting thread
        self.playing = True
        self.frameRate = frameRate

        self.frameThreads = [None, None, None, None, None,
                             None, None, None, None, None,
                             None, None, None, None, None,
                             None, None, None, None, None,
                             None, None, None, None, None,
                             None, None, None, None, None,
                             None, None, None, None, None,
                             None, None, None, None, None]
        self.freeThreadIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        self.threadCount = 40
        self.returnedFrameID = 0

        self.socket = socket
        self.clientID = clientID

        # If no new frames for 3 seconds, stop the background thread. * 6 is because background interval = 1/(6*frameRate) when no thread running
        self.maxTimeOut = frameRate * 3 * 6
        self.timeOutCount = 0
        self.emittingThread = None
        self.threadLock = Lock()

        # Variables for the advanced facial dectection
        self.lastUpdatedFrameID = 0

        self.HARDCODED_pairOfEyesDistanceRange = (1.5, 3.5)
        self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax = 0.8
        self.HARDCODED_updateFaceLocationSearchMultiplier = 1.6
        self.HARDCODED_similarSizeScale = 0.5
        self.HARDCODED_faceNotFoundCountLimit = 10

        # currentDetectedFacesManager is an array that store the tuples of most recent position of the detected faces and an int indicating to whether should the face be displayed on the frame (0 means yes, > 0 means no)
        self.currentDetectedFacesManager = []

        # outputQueue is used for storing threaded functions output
        self.outputQueue = queue.Queue()

        # this is a thread
        self.macroFaceDetectionThread = None
        self.lastMacroFrameID = 0

    def start(self):
        self.playing = True
        self.emittingThread = Thread(target=emitter, args=[self])
        self.emittingThread.start()
        self.lastMacroFrameID = 0

    def stop(self):
        if self.playing:
            self.playing = False
            self.macroFaceDetectionThread = None

    def processNextFrame(self, frame, frameID, endPoint):
        if not len(self.freeThreadIDs):
            print('*****************************************************************************************\n*  ERROR: in VideoManager file: No more free threads. Sir can I have sum more threads?  *\n*****************************************************************************************\n')
            return

        # If the macroFaceDetectionThread wasnt initialized (this is the first ever frame)
        if self.macroFaceDetectionThread == None:
            self.lastMacroFrameID = 0
            # Run the next macro facial detection (my implementation) in a seperate thread to avoid interfering with updating new frames
            args = {
                'frameMngr': ImageManager(frame),
                'angles': (45, 0, -45),
                'scaleFactor': 1.1,
                'minNeighbours': 6
            }
            self.macroFaceDetectionThread = FrameThread(
                None, frameID, MacroFacialDetection_findFacesCounterClockwiseMultipleAngles, args, None)
            self.macroFaceDetectionThread.start()

        elif not self.macroFaceDetectionThread.is_alive():
            # getting the most recent thoroughly detected faces' positions
            delayDetectedFaces = self.macroFaceDetectionThread.output

            self.lastMacroFrameID = self.macroFaceDetectionThread.frameID
            #################################################################################################################################################################################################

            # Merging the macro facial detection into the micro facial detection list
            # Check if there's overlapping faces, keep the one in the currentDetectedFacesManager because its the newest one, and remove the one in hte delayDetectedFaces.
            # Every other delayFace that doens't overlap with any other face in the currentDetectedFacesManager, we add onto the currentDetectedFacesManager
            tempNewFaces = []
            if delayDetectedFaces != None and len(delayDetectedFaces) != 0:
                if self.currentDetectedFacesManager == None or len(self.currentDetectedFacesManager) == 0:
                    self.currentDetectedFacesManager = []
                    for newFace in delayDetectedFaces:
                        self.currentDetectedFacesManager.append(
                            [newFace, self.lastMacroFrameID])
                else:
                    for delayFace in delayDetectedFaces:
                        alreadyFound = False

                        for currFaceManageer in self.currentDetectedFacesManager:
                            currFace = currFaceManageer[0]

                            if delayFace.similarSize(currFace, self.HARDCODED_similarSizeScale) and delayFace.overlap(currFace):
                                alreadyFound = True
                                break

                        if not alreadyFound:
                            tempNewFaces.append(delayFace)

                    for newFace in tempNewFaces:
                        self.currentDetectedFacesManager.append(
                            [newFace, self.lastMacroFrameID])

            # Run the next macro facial detection (my implementation) in a seperate thread to avoid interfering with updating new frames
            args = {
                'frameMngr': ImageManager(frame),
                'angles': (45, 0, -45),
                'scaleFactor': 1.1,
                'minNeighbours': 6
            }
            self.macroFaceDetectionThread = FrameThread(
                None, frameID, MacroFacialDetection_findFacesCounterClockwiseMultipleAngles, args, None)
            self.macroFaceDetectionThread.start()

        threadID = self.freeThreadIDs.pop(0)

        args = {
            'iesvm': self,
            'frame': frame,
            'frameID': frameID,
            'copy_currentDetectedFacesManager': copy.deepcopy(self.currentDetectedFacesManager),
        }
        self.frameThreads[threadID] = FrameThread(
            threadID, frameID, MicroFacialDetection_findFacesBasedOnCurrentDetectedFacesList, args, endPoint)
        self.frameThreads[threadID].start()


def MacroFacialDetection_findFacesCounterClockwiseMultipleAngles(args):
    """
    Help multithreading a function by being a dummy function which only purpose is to call the threadedly-desired? function and save its output into the
    argument queue instead of returning it.
        :param function: the function that needs to be multithreaded
        :param arguments: a list of function's arguments []
        :param outputQueue: the queue that stores the function output
    """
    return args.get('frameMngr').findFacesCounterClockwiseMultipleAngles(
        args.get('angles'), args.get('scaleFactor'), args.get('minNeighbours'))


def MicroFacialDetection_findFacesBasedOnCurrentDetectedFacesList(args):

    iesvm = args.get('iesvm')
    frame = args.get('frame')
    frameID = args.get('frameID')
    copy_currentDetectedFacesManager = args.get(
        'copy_currentDetectedFacesManager')

    # updating the detected faces in the last frame by searching around their area
    if copy_currentDetectedFacesManager != None and len(copy_currentDetectedFacesManager) != 0:

        bool_loop = True
        index_currentDetectedFacesManager = 0
        if index_currentDetectedFacesManager >= len(copy_currentDetectedFacesManager):
            bool_loop = False
        while bool_loop:
            face = copy_currentDetectedFacesManager[index_currentDetectedFacesManager][0]

            dimensions = face.dimensions
            counterClockwiseAngle = face.counterClockwiseAngle % 360
            faceCenter = Point(face.center.x, face.center.y)

            # rotate clockwise because the face's angle is the counter clockwise angle of the right eye to the left eye
            frameOrigin = Point(
                frame.shape[1]/2, frame.shape[0]/2)
            rotatedFrame = rotateClockwise(
                frame, counterClockwiseAngle)
            rotatedFrameOrigin = Point(
                rotatedFrame.shape[1]/2, rotatedFrame.shape[0]/2)

            # rotate clockwise the faceCenter and project it to the new image
            rotatedFaceCenter = faceCenter.rotatePointClockwise(
                frameOrigin, counterClockwiseAngle)
            rotatedFaceCenter = rotatedFaceCenter.projectPoint(
                frameOrigin, rotatedFrameOrigin)

            # Calculate the region of the potential face
            searchingDimensions = (dimensions[0] * iesvm.HARDCODED_updateFaceLocationSearchMultiplier,
                                   dimensions[1] * iesvm.HARDCODED_updateFaceLocationSearchMultiplier)
            cropRangeMinY = int(
                max(0, rotatedFaceCenter.y - searchingDimensions[1] / 2))
            cropRangeMaxY = int(
                min(rotatedFrame.shape[0], rotatedFaceCenter.y + searchingDimensions[1] / 2))

            cropRangeMinX = int(
                max(0, rotatedFaceCenter.x - searchingDimensions[0] / 2))
            cropRangeMaxX = int(
                min(rotatedFrame.shape[1], rotatedFaceCenter.x + searchingDimensions[0] / 2))

            # Cropping out the region of the potential face
            try:
                croppedPotentialFace = np.zeros(
                    (cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX), dtype='uint8')
                croppedPotentialFace[0: croppedPotentialFace.shape[0], 0: croppedPotentialFace.shape[1]] = \
                    rotatedFrame[cropRangeMinY:cropRangeMaxY,
                                 cropRangeMinX:cropRangeMaxX]

            except:
                croppedPotentialFace = np.zeros(
                    (cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX, 3), dtype='uint8')
                croppedPotentialFace[0: croppedPotentialFace.shape[0], 0: croppedPotentialFace.shape[1]] = \
                    rotatedFrame[cropRangeMinY:cropRangeMaxY,
                                 cropRangeMinX:cropRangeMaxX]

            croppedPotentialFaceCenter = Point(
                croppedPotentialFace.shape[1]/2, croppedPotentialFace.shape[0]/2)

            # gray scale the image
            grayCroppedPotentialFace = cv.cvtColor(
                croppedPotentialFace, cv.COLOR_BGR2GRAY)

            # run basic haarcascade facial detection on it
            # CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ?
            detectedFaces = haarcascade_FacialDetection(
                grayCroppedPotentialFace, 1.1, 6, HARDCODED_faceMinDimensions, HARDCODED_faceMaxDimensions)

            # if found nothing then move to the next face's last frame location
            if len(detectedFaces) == 0:

                # If no faces found after HARDCODED_faceNotFoundCountLimit number of frame, then remove it off the list of potential faces
                faceLastFoundOnFrameID = copy_currentDetectedFacesManager[
                    index_currentDetectedFacesManager][1]
                if frameID - faceLastFoundOnFrameID >= iesvm.HARDCODED_faceNotFoundCountLimit:

                    copy_currentDetectedFacesManager.pop(
                        index_currentDetectedFacesManager)
                    # if there's still more currentFaceManager to iterate
                    if index_currentDetectedFacesManager < len(copy_currentDetectedFacesManager):
                        continue

                    # if there's no more currentFaceManager to iterate then break the loop
                    bool_loop = False
                    continue

                # if still under the faceNotFoundLimit then continue iterating
                index_currentDetectedFacesManager = index_currentDetectedFacesManager + 1
                if index_currentDetectedFacesManager >= len(copy_currentDetectedFacesManager):
                    bool_loop = False

                continue

            # if found anything, then convert the rectangle coordinates received from detectMultiScale function to a detectedFace obj.
            biggestFace = detectedFaces[0]
            for i in range(1, len(detectedFaces)):
                # comparing the dimensions (width only, cause if width is > then height is also >) among all the faces found.
                if biggestFace[2] < detectedFaces[i][2]:
                    biggestFace = detectedFaces[i]

            # Creating DetectedFace obj representing the biggest face found in the potential region above
            biggestFace = DetectedFace(
                (biggestFace[0], biggestFace[1]), (biggestFace[2], biggestFace[3]))

            # Finding biggestFace's angle
            # CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ?

            # find the eyes
            tempEyesArray = haarcascade_EyeDetection(
                grayCroppedPotentialFace, 1.1, 6, HARDCODED_eyeMinDimensions, HARDCODED_eyeMaxDimensions)
            detectedEyes = []

            if face.leftEye != None:
                referenceEye = face.leftEye
            elif face.rightEye != None:
                referenceEye = face.rightEye
            else:
                faceUpperLeftPoint = biggestFace.upperLeft
                # if the face has no eyes found, then create a standard reference eye
                # HARDCODED ___ Eye's width is usually at most 3/8 of face's width. Eye's height is usually at most 1/3 of face's height
                referenceEye = DetectedArea((faceUpperLeftPoint.x + biggestFace.dimensions[0]/8, faceUpperLeftPoint.y + biggestFace.dimensions[1]/6), (
                    biggestFace.dimensions[0] * 3 / 8, biggestFace.dimensions[1]/3))

            for (x, y, w, h) in tempEyesArray:
                eye = DetectedArea((x, y), (w, h))
                # If the shape and size of the eye found in the image is approximately the same as the reference eye
                if eye.similarSize(referenceEye, iesvm.HARDCODED_similarSizeScale):
                    detectedEyes.append(eye)

            # find the pairs of eyes
            if len(detectedEyes) > 1:
                # This might be slightlyyyyy overkill but usually there's like at most 2-3 eyes so it wont be too computationally heavy.
                for i in range(len(detectedEyes)):
                    j = i + 1
                    for j in range(len(detectedEyes)):
                        if detectedEyes[i].appropriateDistanceTo(detectedEyes[j], iesvm.HARDCODED_pairOfEyesDistanceRange[0], iesvm.HARDCODED_pairOfEyesDistanceRange[1]):
                            leftEye = detectedEyes[i]
                            rightEye = detectedEyes[j]
                            if detectedEyes[i].center.x > detectedEyes[j].center.x:
                                leftEye = detectedEyes[j]
                                rightEye = detectedEyes[i]
                            # if the distance of LeftEye's center to face's upperLeft Point ~ that of RightEye's center to face's upperRight Point
                            # and distance of LeftEye's center to face's lowerLeft Point ~ that of RightEye's center to face's lowerRight Point
                            # then this is probably the right pair of eyes
                            upperLeftDist = leftEye.center.distTo(
                                biggestFace.upperLeft)
                            upperRightDist = rightEye.center.distTo(
                                biggestFace.upperRight)
                            lowerLeftDist = leftEye.center.distTo(
                                biggestFace.lowerLeft)
                            lowerRightDist = rightEye.center.distTo(
                                biggestFace.lowerRight)
                            criterias = 0
                            if upperLeftDist < upperRightDist and upperLeftDist > upperRightDist * iesvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                criterias = criterias + 1
                            if upperRightDist <= upperLeftDist and upperRightDist > upperLeftDist * iesvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                criterias = criterias + 1
                            if lowerLeftDist < lowerRightDist and lowerLeftDist > lowerRightDist * iesvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                criterias = criterias + 1
                            if lowerRightDist <= lowerLeftDist and lowerRightDist > lowerLeftDist * iesvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                criterias = criterias + 1
                            if criterias == 2:
                                biggestFace.leftEye = leftEye
                                biggestFace.rightEye = rightEye
                                break

            # find the face's angle using the pair of eyes
            if biggestFace.leftEye != None and biggestFace.rightEye != None:
                # calculate teh relative angle created by the two eyes
                relativeCounterClockwiseAngle = biggestFace.leftEye.center.relativeCounterClockwiseAngle(
                    biggestFace.rightEye.center)
                # the face's current angle is the already rotated angle pre-haarcascade detection + the relative angle
                biggestFace.counterClockwiseAngle = counterClockwiseAngle + \
                    relativeCounterClockwiseAngle
                biggestFace.rotateAreaCounterClockwise(
                    biggestFace.center, relativeCounterClockwiseAngle)
            else:
                # if there's not a pair of eyes then the face's angle = the previous face that was detected in the same area's angle
                biggestFace.counterClockwiseAngle = counterClockwiseAngle

            # Returning the face's coordinates to the original frame
            biggestFace.projectArea(
                croppedPotentialFaceCenter, rotatedFaceCenter)
            biggestFace.rotateAreaCounterClockwise(
                rotatedFrameOrigin, biggestFace.counterClockwiseAngle)
            biggestFace.projectArea(
                rotatedFrameOrigin, frameOrigin)

            if biggestFace.leftEye != None:
                biggestFace.leftEye.projectArea(
                    croppedPotentialFaceCenter, rotatedFaceCenter)
                biggestFace.leftEye.rotateAreaCounterClockwise(
                    rotatedFrameOrigin, biggestFace.counterClockwiseAngle)
                biggestFace.leftEye.projectArea(
                    rotatedFrameOrigin, frameOrigin)

            if biggestFace.rightEye != None:
                biggestFace.rightEye.projectArea(
                    croppedPotentialFaceCenter, rotatedFaceCenter)
                biggestFace.rightEye.rotateAreaCounterClockwise(
                    rotatedFrameOrigin, biggestFace.counterClockwiseAngle)
                biggestFace.rightEye.projectArea(
                    rotatedFrameOrigin, frameOrigin)

            # update the face's location and reset last face found ID to frameID
            copy_currentDetectedFacesManager[index_currentDetectedFacesManager][0] = biggestFace
            copy_currentDetectedFacesManager[index_currentDetectedFacesManager][1] = frameID

            # continue iterating the currentFaceManager
            index_currentDetectedFacesManager = index_currentDetectedFacesManager + 1
            if index_currentDetectedFacesManager >= len(copy_currentDetectedFacesManager):
                bool_loop = False

        # Return the updated currentDetectedFacesManager
        return copy_currentDetectedFacesManager

    # If currentDetectedFaceManager has nothing in it then theres nothing to scan/update
    return []


def emitter(videoManager):

    while videoManager.playing:
        if len(videoManager.freeThreadIDs) < videoManager.threadCount:

            i = 0
            frameThreads = videoManager.frameThreads
            freeThreadIDs = videoManager.freeThreadIDs

            latestFrameIDIndex = -1
            latestFrameID = -1

            videoManager.threadLock.acquire()

            while i < len(frameThreads):

                # Find the thread with the latest frameID and has finished running and output is not None (this is a workaround for a bug, output value is not yet initialized)

                if frameThreads[i] != None and not frameThreads[i].is_alive() and frameThreads[i].output != None:
                    if frameThreads[i].frameID > latestFrameID:
                        latestFrameIDIndex = i
                        latestFrameID = frameThreads[i].frameID

                i += 1

            videoManager.threadLock.release()

            if latestFrameID > -1:
                latestThread = frameThreads[latestFrameIDIndex]
                # if the latest frame that just got rendered by frameThread is newer than the last frame that we sent back to the client
                # then send that latest frame to the client
                if latestFrameID > videoManager.returnedFrameID:

                    videoManager.returnedFrameID = latestFrameID

                    newestCurrentDetectedFacesManager = []

                    # Pick whether to use the currentDetectedFacesManager from the micro thread or the last currentDetectedFacesManager
                    if latestFrameID > videoManager.lastMacroFrameID:
                        newestCurrentDetectedFacesManager = latestThread.output
                    else:
                        videoManager.threadLock.acquire()
                        newestCurrentDetectedFacesManager = videoManager.currentDetectedFacesManager
                        videoManager.threadLock.release()

                    # Sending back upperLeft, upperRight, lowerRight, and lowerLeft of the faces, leftEyes, and rightEyes
                    returnedFacesInfo = []
                    if newestCurrentDetectedFacesManager == None:
                        returnedFacesInfo = None
                    else:
                        for faceManager in newestCurrentDetectedFacesManager:
                            detectedFaceObj = faceManager[0]
                            upperLeft = (detectedFaceObj.upperLeft.x,
                                         detectedFaceObj.upperLeft.y)
                            upperRight = (detectedFaceObj.upperRight.x,
                                          detectedFaceObj.upperRight.y)
                            lowerRight = (detectedFaceObj.lowerRight.x,
                                          detectedFaceObj.lowerRight.y)
                            lowerLeft = (detectedFaceObj.lowerLeft.x,
                                         detectedFaceObj.lowerLeft.y)
                            faceInfo = (upperLeft, upperRight,
                                        lowerRight, lowerLeft)

                            leftEye = detectedFaceObj.leftEye
                            if leftEye != None:
                                upperLeft = (leftEye.upperLeft.x,
                                             leftEye.upperLeft.y)
                                upperRight = (leftEye.upperRight.x,
                                              leftEye.upperRight.y)
                                lowerRight = (leftEye.lowerRight.x,
                                              leftEye.lowerRight.y)
                                lowerLeft = (leftEye.lowerLeft.x,
                                             leftEye.lowerLeft.y)
                                leftEyeInfo = (
                                    upperLeft, upperRight, lowerRight, lowerLeft)
                            else:
                                leftEyeInfo = None

                            rightEye = detectedFaceObj.rightEye
                            if rightEye != None:
                                upperLeft = (rightEye.upperLeft.x,
                                             rightEye.upperLeft.y)
                                upperRight = (rightEye.upperRight.x,
                                              rightEye.upperRight.y)
                                lowerRight = (rightEye.lowerRight.x,
                                              rightEye.lowerRight.y)
                                lowerLeft = (rightEye.lowerLeft.x,
                                             rightEye.lowerLeft.y)
                                rightEyeInfo = (
                                    upperLeft, upperRight, lowerRight, lowerLeft)
                            else:
                                rightEyeInfo = None

                            # returnedFacesInfo.append([faceInfo, leftEyeInfo, rightEyeInfo, detectedFaceObj.counterClockwiseAngle])
                            angle = detectedFaceObj.counterClockwiseAngle % 360
                            returnedFacesInfo.append((np.array(faceInfo).tolist(), np.array(
                                leftEyeInfo).tolist(), np.array(rightEyeInfo).tolist(), angle))

                        if len(returnedFacesInfo) == 0:
                            returnedFacesInfo = None
                        else:
                            returnedFacesInfo = np.array(returnedFacesInfo)
                            returnedFacesInfo = returnedFacesInfo.tolist()

                    data = {'facesInfo': returnedFacesInfo,
                            'frameID': latestThread.frameID}
                    videoManager.socket.emit(
                        latestThread.endPoint, data, to=videoManager.clientID)

                    if latestFrameID > videoManager.lastMacroFrameID:
                        # Update the currentDetectedFacesManager list
                        videoManager.threadLock.acquire()
                        videoManager.currentDetectedFacesManager = latestThread.output
                        videoManager.threadLock.release()

                for i in range(len(frameThreads)):
                    if frameThreads[i] != None and frameThreads[i].frameID <= latestFrameID:
                        frameThreads[i] = None
                        freeThreadIDs.append(i)

            time.sleep(1/(3*videoManager.frameRate))
        else:
            time.sleep(1/(6 * videoManager.frameRate))
