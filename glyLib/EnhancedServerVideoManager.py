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


            #################################################################################################################################################################################################


            # Replacing the currentDetectedFacesManager with faces found in macro facial detection 
            if delayDetectedFaces != None and len(delayDetectedFaces) != 0:
                self.currentDetectedFacesManager = []
                for delayFace in delayDetectedFaces:
                    self.currentDetectedFacesManager.append([delayFace, 0])



            returnedFacesInfo = []
            for faceManager in self.currentDetectedFacesManager:
                detectedFaceObj = faceManager[0]
                upperLeft = (detectedFaceObj.upperLeft.x, detectedFaceObj.upperLeft.y)
                upperRight = (detectedFaceObj.upperRight.x, detectedFaceObj.upperRight.y)
                lowerRight = (detectedFaceObj.lowerRight.x, detectedFaceObj.lowerRight.y)
                lowerLeft = (detectedFaceObj.lowerLeft.x, detectedFaceObj.lowerLeft.y)
                faceInfo = (upperLeft, upperRight, lowerRight, lowerLeft)

                leftEye = detectedFaceObj.leftEye
                if leftEye != None:
                    upperLeft = (leftEye.upperLeft.x, leftEye.upperLeft.y)
                    upperRight = (leftEye.upperRight.x, leftEye.upperRight.y)
                    lowerRight = (leftEye.lowerRight.x, leftEye.lowerRight.y)
                    lowerLeft = (leftEye.lowerLeft.x, leftEye.lowerLeft.y)
                    leftEyeInfo = (upperLeft, upperRight, lowerRight, lowerLeft)
                else:
                    leftEyeInfo = None

                rightEye = detectedFaceObj.rightEye
                if rightEye != None:
                    upperLeft = (rightEye.upperLeft.x, rightEye.upperLeft.y)
                    upperRight = (rightEye.upperRight.x, rightEye.upperRight.y)
                    lowerRight = (rightEye.lowerRight.x, rightEye.lowerRight.y)
                    lowerLeft = (rightEye.lowerLeft.x, rightEye.lowerLeft.y)
                    rightEyeInfo = (upperLeft, upperRight, lowerRight, lowerLeft)
                else:
                    rightEyeInfo = None

                

                # returnedFacesInfo.append([faceInfo, leftEyeInfo, rightEyeInfo, detectedFaceObj.counterClockwiseAngle])
                angle = detectedFaceObj.counterClockwiseAngle % 360
                returnedFacesInfo.append((np.array(faceInfo).tolist(), np.array(leftEyeInfo).tolist(), np.array(rightEyeInfo).tolist(), angle))

            if len(returnedFacesInfo) == 0:
                returnedFacesInfo = None
            else:
                returnedFacesInfo = np.array(returnedFacesInfo)
                returnedFacesInfo = returnedFacesInfo.tolist()
                # returnedFacesInfo = json.dumps(returnedFacesInfo)
            
            data = {'facesInfo': returnedFacesInfo,
                    'frameID': self.macroFaceDetectionThread.frameID}
            self.socket.emit(
                endPoint, data, to=self.clientID)





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
