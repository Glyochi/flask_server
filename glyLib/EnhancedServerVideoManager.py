from threading import Thread, Lock
from .FrameThread import FrameThread
from .ImageManager import ImageManager
from .HelperFunctions import *
from .Point import Point
from .DetectedArea import DetectedArea, DetectedFace
import time
import queue
import json


class EnhancedServerVideoManager:
    def __init__(self, frameRate, socket, clientID, coordinates):
        self.coordinates = coordinates

        # Used for the emitting thread
        self.playing = True
        self.frameRate = frameRate

        self.frameThreads = [None, None, None, None,
                             None, None, None, None, None, None]
        self.freeThreadIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.threadCount = 10
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
        self.HARDCODED_updateFaceLocationSearchMultiplier = 1.4
        self.HARDCODED_similarSizeScale = 0.7
        self.HARDCODED_faceNotFoundCountLimit = 20

        # currentDetectedFacesManager is an array that store the tuples of most recent position of the detected faces and an int indicating to whether should the face be displayed on the frame (0 means yes, > 0 means no)
        self.currentDetectedFacesManager = []

        # outputQueue is used for storing threaded functions output
        self.outputQueue = queue.Queue()

        # this is a thread
        self.macroFaceDetectionThread = None

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
        # Its pretty impossible to have all 5 threads taken up
        if not len(self.freeThreadIDs):
            print('*****************************************************************************************\n*  ERROR: in VideoManager file: No more free threads. Sir can I have sum more threads?  *\n*****************************************************************************************\n')
            return

        # If the macroFaceDetectionThread wasnt initialized (this is the first ever frame)
        if self.macroFaceDetectionThread == None:
            # Run the next macro facial detection (my implementation) in a seperate thread to avoid interfering with updating new frames
            args = {
                'frameMngr': ImageManager(frame),
                'angles': (45, 0, -45),
                'scaleFactor': 1.1,
                'minNeighbours': 3
            }
            self.macroFaceDetectionThread = FrameThread(
                None, frameID, MacroFacialDetection_findFacesCounterClockwiseMultipleAngles, args, None)
            self.macroFaceDetectionThread.start()

        elif not self.macroFaceDetectionThread.is_alive():
            # print("LAST MACRO ", self.macroFaceDetectionThread.frameID)
            # print("THIS MACRO ", frameID)
            # print("SKIPPED FRAMES ", frameID - self.macroFaceDetectionThread.frameID)

            delayDetectedFaces = self.macroFaceDetectionThread.output

            # Merging the macro facial detection into the micro facial detection list
            tempNewFaces = []
            if delayDetectedFaces != None and len(delayDetectedFaces) != 0:

                if self.currentDetectedFacesManager == None:
                    self.currentDetectedFacesManager = []
                    # Adding all the new faces (werent detected before into the list of all current faces)
                    for newFace in delayDetectedFaces:
                        self.currentDetectedFacesManager.append([newFace, 0])
                else:
                    # Go thru all the faces and check to see if them have already been detected previously
                    for delayFace in delayDetectedFaces:
                        alreadyFound = False

                        for i in range(len(self.currentDetectedFacesManager)):
                            currFace = self.currentDetectedFacesManager[i][0]

                            if delayFace.similarSize(currFace, self.HARDCODED_similarSizeScale) and delayFace.overlap(currFace):
                                alreadyFound = True
                                break

                        if not alreadyFound:
                            tempNewFaces.append(delayFace)

                    # Adding all the new faces (werent detected before into the list of all current faces)
                    for newFace in tempNewFaces:
                        self.currentDetectedFacesManager.append([newFace, 0])

            # Run the next macro facial detection (my implementation) in a seperate thread to avoid interfering with updating new frames
            args = {
                'frameMngr': ImageManager(frame),
                'angles': (45, 0, -45),
                'scaleFactor': 1.1,
                'minNeighbours': 10
            }
            self.macroFaceDetectionThread = FrameThread(
                None, frameID, MacroFacialDetection_findFacesCounterClockwiseMultipleAngles, args, None)
            self.macroFaceDetectionThread.start()

        threadID = self.freeThreadIDs.pop(0)

        args = {
            'esvm': self,
            'frame': frame,
            'frameID': frameID
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
    return args.get('frameMngr').findFacesCounterClockwiseMultipleAngles_Alternative(
        args.get('angles'), args.get('scaleFactor'), args.get('minNeighbours'))


def MicroFacialDetection_findFacesBasedOnCurrentDetectedFacesList(args):


    esvm = args.get('esvm')

    frame = args.get('frame')
    frameID = args.get('frameID')

    # updating the detected faces in the last frame by searching around their area
    if esvm.currentDetectedFacesManager != None and len(esvm.currentDetectedFacesManager) != 0:

        esvm.threadLock.acquire()
        tempCurrentDetectedFacesManager = esvm.currentDetectedFacesManager.copy()
        esvm.threadLock.release()

        print("DEBUG")
        print(tempCurrentDetectedFacesManager)

        bool_loop = True
        index_currentDetectedFacesManager = 0
        if index_currentDetectedFacesManager >= len(tempCurrentDetectedFacesManager):
            bool_loop = False
        while bool_loop:
            face = tempCurrentDetectedFacesManager[index_currentDetectedFacesManager][0]

            dimensions = face.dimensions
            counterClockwiseAngle = face.counterClockwiseAngle
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
            searchingDimensions = (dimensions[0] * esvm.HARDCODED_updateFaceLocationSearchMultiplier,
                                   dimensions[1] * esvm.HARDCODED_updateFaceLocationSearchMultiplier)
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
                grayCroppedPotentialFace, 1.1, 10)

            # if found nothing then move to the next face's last frame location
            if len(detectedFaces) == 0:

                # increment the faceNotFoundCount
                faceLastFoundOnFrameID = tempCurrentDetectedFacesManager[
                    index_currentDetectedFacesManager][1]
                if frameID - faceLastFoundOnFrameID >= esvm.HARDCODED_faceNotFoundCountLimit:

                    tempCurrentDetectedFacesManager.pop(
                        index_currentDetectedFacesManager)
                    # if there's still more currentFaceManager to iterate
                    if index_currentDetectedFacesManager < len(tempCurrentDetectedFacesManager):
                        continue

                    # if there's no more currentFaceManager to iterate then break the loop
                    bool_loop = False
                    continue

                # if still under the faceNotFoundLimit then continue iterating 
                # index_currentDetectedFacesManager += 1
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
                grayCroppedPotentialFace, 1.1, 10)
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
                if eye.similarSize(referenceEye, esvm.HARDCODED_similarSizeScale):
                    detectedEyes.append(eye)

            # find the pairs of eyes
            if len(detectedEyes) > 1:
                # This might be slightlyyyyy overkill but usually there's like at most 2-3 eyes so it wont be too computationally heavy.
                for i in range(len(detectedEyes)):
                    j = i + 1
                    for j in range(len(detectedEyes)):
                        if detectedEyes[i].appropriateDistanceTo(detectedEyes[j], esvm.HARDCODED_pairOfEyesDistanceRange[0], esvm.HARDCODED_pairOfEyesDistanceRange[1]):
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
                            if upperLeftDist < upperRightDist and upperLeftDist > upperRightDist * esvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                criterias = criterias + 1
                            if upperRightDist <= upperLeftDist and upperRightDist > upperLeftDist * esvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                criterias = criterias + 1
                            if lowerLeftDist < lowerRightDist and lowerLeftDist > lowerRightDist * esvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                criterias = criterias + 1
                            if lowerRightDist <= lowerLeftDist and lowerRightDist > lowerLeftDist * esvm.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
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

            # update the face's location and reset last face found ID to frameID
            tempCurrentDetectedFacesManager[index_currentDetectedFacesManager][0] = biggestFace
            tempCurrentDetectedFacesManager[index_currentDetectedFacesManager][1] = frameID

            # continue iterating the currentFaceManager
            index_currentDetectedFacesManager = index_currentDetectedFacesManager + 1
            if index_currentDetectedFacesManager >= len(tempCurrentDetectedFacesManager):
                bool_loop = False

        # Return the updated currentDetectedFacesManager
        return tempCurrentDetectedFacesManager

    # If currentDetectedFaceManager has nothing in it then theres nothing to scan/update
    return None


def emitter(videoManager):

    while videoManager.playing:
        if len(videoManager.freeThreadIDs) < videoManager.threadCount:
            # Reset time out count
            # videoManager.timeOutCount = 0

            i = 0
            frameThreads = videoManager.frameThreads
            freeThreadIDs = videoManager.freeThreadIDs
            finishedThreadIDs = []

            videoManager.threadLock.acquire()

            while i < len(frameThreads):

                # If thread t has finished running

                if frameThreads[i] != None and not frameThreads[i].is_alive():
                    # add the finished thread id onto the freeThreadIDs attribute of the videoManager
                    freeThreadIDs.append(i)

                    # add the finished thread id onto the finishedThreadArray thats gonna be used later
                    finishedThreadIDs.append(i)

                i += 1

            videoManager.threadLock.release()

            if len(finishedThreadIDs) >= 1:
                # If more than one thread finished, then only send the latest one's output to the client

                latestFrameIDIndex = 0
                latestFrameID = frameThreads[finishedThreadIDs[latestFrameIDIndex]].frameID

                # Finding the newest frame that just got rendered by frameThread
                i = 1
                while i < len(finishedThreadIDs):

                    if(frameThreads[finishedThreadIDs[i]].frameID > latestFrameID):
                        latestFrameIDIndex = i
                        latestFrameID = frameThreads[finishedThreadIDs[i]].frameID

                    i += 1

                # if the latest frame that just got rendered by frameThread is newer than the last frame that we sent back to the client
                # then send that latest frame to the client
                if latestFrameID > videoManager.returnedFrameID:
                    latestThread = frameThreads[finishedThreadIDs[latestFrameIDIndex]]
                    videoManager.returnedFrameID = latestFrameID

                    # Update the currentDetectedFacesManager list
                    videoManager.threadLock.acquire()
                    videoManager.currentDetectedFacesManager = latestThread.output
                    videoManager.threadLock.release()

                    # Only sending back upper left point coordinates, width and height, and counter clock-wise angle of the face
                    returnedFacesInfo = []
                    if videoManager.currentDetectedFacesManager == None:
                        returnedFacesInfo = None
                    else:
                        for faceManager in videoManager.currentDetectedFacesManager:
                            detectedFaceObj = faceManager[0]
                            faceInfo = []
                            upperLeft = [detectedFaceObj.upperLeft.x, detectedFaceObj.upperLeft.y]
                            upperRight = [detectedFaceObj.upperRight.x, detectedFaceObj.upperRight.y]
                            lowerRight = [detectedFaceObj.lowerRight.x, detectedFaceObj.lowerRight.y]
                            lowerLeft = [detectedFaceObj.lowerLeft.x, detectedFaceObj.lowerLeft.y]
                            faceInfo.append(upperLeft)
                            faceInfo.append(upperRight)
                            faceInfo.append(lowerRight)
                            faceInfo.append(lowerLeft)

                            # leftEyeInfo = []
                            # leftEye = detectedFaceObj.leftEye
                            # if leftEye == None:
                            #     leftEyeInfo = None
                            # else:
                            #     upperLeft = [leftEye.upperLeft.x, leftEye.upperLeft.y]
                            #     upperRight = [leftEye.upperRight.x, leftEye.upperRight.y]
                            #     lowerRight = [leftEye.lowerRight.x, leftEye.lowerRight.y]
                            #     lowerLeft = [leftEye.lowerLeft.x, leftEye.lowerLeft.y]
                            #     leftEyeInfo.append(upperLeft)
                            #     leftEyeInfo.append(upperRight)
                            #     leftEyeInfo.append(lowerRight)
                            #     leftEyeInfo.append(lowerLeft)
                            
                            # rightEyeInfo = []
                            # rightEye = detectedFaceObj.rightEye
                            # if rightEye == None:
                            #     rightEyeInfo = None
                            # else:
                            #     upperLeft = [rightEye.upperLeft.x, rightEye.upperLeft.y]
                            #     upperRight = [rightEye.upperRight.x, rightEye.upperRight.y]
                            #     lowerRight = [rightEye.lowerRight.x, rightEye.lowerRight.y]
                            #     lowerLeft = [rightEye.lowerLeft.x, rightEye.lowerLeft.y]
                            #     rightEyeInfo.append(upperLeft)
                            #     rightEyeInfo.append(upperRight)
                            #     rightEyeInfo.append(lowerRight)
                            #     rightEyeInfo.append(lowerLeft)
                            

                            # returnedFacesInfo.append([faceInfo, leftEyeInfo, rightEyeInfo, detectedFaceObj.counterClockwiseAngle])
                            angle = detectedFaceObj.counterClockwiseAngle % 360
                            returnedFacesInfo.append([faceInfo, angle])
                            print("ANGLE ", angle)

                        if len(returnedFacesInfo) == 0:
                            returnedFacesInfo = None
                        else:
                            returnedFacesInfo = np.array(returnedFacesInfo)
                            returnedFacesInfo = returnedFacesInfo.tolist()
                            # returnedFacesInfo = json.dumps(returnedFacesInfo)

                    data = {'facesInfo': returnedFacesInfo,
                            'frameID': latestThread.frameID}
                    videoManager.socket.emit(
                        latestThread.endPoint, data, to=videoManager.clientID)

                # if the latest frame is behind the last frame that was sent to the client, then do nothing

                # Removing the finished threads out of the running threads array
                for i in finishedThreadIDs:
                    frameThreads[i] = None

            time.sleep(1/(3*videoManager.frameRate))
        else:
            time.sleep(1/(6 * videoManager.frameRate))
            # videoManager.timeOutCount += 1
            # if videoManager.timeOutCount == videoManager.maxTimeOut:
            #     videoManager.stop()
