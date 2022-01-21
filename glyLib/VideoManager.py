import cv2 as cv
from numpy import append
from .DetectedArea import DetectedArea
from .DetectedArea import DetectedFace
from .ImageManager import ImageManager
from .Point import Point
from .HelperFunctions import *
import time
import threading
import queue

"""
The idea is that VideoManagers take care of both displaying the original frames when its time and do detections on the given frames.
VideoManager splits the two tasks into two processes/threads?. One thread handles drawing frames and detectedFaces saved in VideoManager.faces array.
The other thread handles running face detection on every other frame/third frame/fourth frame/ however it sees fit-th frame and update the faces location in VideoManager.faces array.

This works by the assumption that for a standard 30fps videos, a normal human being wont be able to move to fast, making the exact location of their face in this frame can be used to 
approximate the upcoming locations in the next few frames. So 
"""


class VideoManager:
    """
    A manager that stores multiple frames (images) and manage how to display them and perform facial detection on those frames.
    """

    HARDCODED_pairOfEyesDistanceRange = (1.5, 3.5)
    HARDCODED_distanceVariation_EyeToCorners_MinOfMax = 0.8
    HARDCODED_updateFaceLocationSearchMultiplier = 1.4
    HARDCODED_similarSizeScale = 0.5
    HARDCODED_faceNotFoundCountLimit = 3

    def __init__(self, videoFps=0, dir=""):
        """
        Construct a VideoManager object
            :param videoFps: the fps of the input video
            :param dir: the directory to the video
        """
        self.videoFps = videoFps
        self.video = cv.VideoCapture(dir)
        self.faces = []

    def displayNonInterferedMethod(self):
        """
        SHOW THE NORMAL JUST GRAB NEWFRAME WHENEVER IT COMES INSTEAD OF LIMITING FRAMERATE THING
        """
        haar_cascasde_face = cv.CascadeClassifier(
            "classifier/haarcascade_frontalface_default.xml")

        frameTimeInterval = 1/self.videoFps
        # if frame is read correctly/ new frame is available, then ret is True.

        prev_frame = time.time()
        while True:
            # Capture frame-by-frame
            ret, frame = self.video.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            start = time.time()
            frame = resizeMinTo500(frame)
            print(f"DEBUG resize time {time.time() - start}")

            # Display the resulting frame
            next_frame = time.time()
            frameTime = next_frame - prev_frame
            fps = 1/frameTime
            videoSpeed = frameTimeInterval * 100 / frameTime

            cv.putText(frame, f"DEBUG {fps:0.2f} FPS", (12, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv.putText(frame, f"DEBUG FrameTime {frameTime*1000:0.2f}ms",
                       (12, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv.putText(frame, f"DEBUG video speed {videoSpeed:0.2f}%", (
                12, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            prev_frame = next_frame

            cv.imshow('60fps maybe?', frame)

            if cv.waitKey(1) == ord('q'):
                break
            # When everything done, release the capture
        self.video.release()
        cv.destroyAllWindows()

    def displayStandardMethod(self):
        """
        SHOW THEM THE STANDARD METHOD PERFORMANCE
        """
        haar_cascasde_face = cv.CascadeClassifier(
            "classifier/haarcascade_frontalface_default.xml")

        # frameTimeInterval, prev_frame, and next_frame are used to determine the frame rate of the video, how often do we check for new frames
        frameTimeInterval = 1/self.videoFps
        prev_frame = time.time()
        next_frame = prev_frame + frameTimeInterval

        # actual_prev_frame and actual_next_frame are for determining the actual fps/frame time when displayed. The deviation from the actual fps/frametime values is because
        # of rendering and image processings applied to the image
        actual_prev_frame = prev_frame
        actual_next_frame = 0

        bool_DisplayingVideo = True

        # startTime, endTime, frameCount are used for debugging (checking average fps)
        debug_startTime = time.time()
        debug_frameCount = 0

        while bool_DisplayingVideo:

            # if its next time for the next frame to be drawn
            if time.time() >= next_frame:

                # Capture frame-by-frame
                ret, frame = self.video.read()

                # if frame is read correctly/ new frame is available, then ret is True.
                if not ret:
                    print("Can't receive frame. Exiting ...")
                    bool_DisplayingVideo = False
                    continue

                # set new frame target (next_frame) after check if new frame is available instead of before. This is to make sure when live capturing
                # the newest image will be shown as soon as they are ready. If this was to be put before checking if new frame is available, newest
                # frame have to wait till the next frameTimeInterval to be shown, which increases latency
                prev_frame = next_frame
                next_frame = prev_frame + frameTimeInterval

                # Our operations on the frame come here
                debug_frameCount = debug_frameCount + 1

                frame = resizeMinTo500(frame)

                # startTime = time.time()
                grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # print(f"grayscale runtime {time.time() - startTime:0.6f} seconds")

                # startTime = time.time()
                faces = haar_cascasde_face.detectMultiScale(
                    grayFrame, 2, 6, minSize=(30, 30))
                # print(f"Haarcascade Runtime {time.time() - startTime:0.6f} seconds")

                for (x, y, w, h) in faces:
                    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

                # Calculating fps, timeFrame, video speed
                actual_next_frame = time.time()
                frameTime = actual_next_frame - actual_prev_frame

                actual_prev_frame = actual_next_frame
                fps = 1/frameTime
                videoSpeed = frameTimeInterval * 100 / frameTime

                # Display the resulting frame
                cv.putText(frame, f"DEBUG {fps:0.2f} FPS", (12, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(frame, f"DEBUG FrameTime {frameTime*1000:0.2f}ms",
                           (12, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(frame, f"DEBUG video speed {videoSpeed:0.2f}%", (
                    12, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                # print(f"DEBUG interval {frameTimeInterval*1000:0.2f} frameTime {frameTime*1000:0.2f}")
                cv.imshow('video standard method', frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    bool_DisplayingVideo = False
                    continue

        print(f"DEBUG Video length {time.time() - debug_startTime:0.2f}")
        print(
            f"DEBUG Average fps {debug_frameCount / (time.time() - debug_startTime):0.2f} ")

    def displayHowBadUnoptimizedProcessIs(self):
        """
        SHOW THEM HOW BAD IT IS TO JUST RUN DETECTION ON EVERY FRAMES
        """

        # frameTimeInterval, prev_frame, and next_frame are used to determine the frame rate of the video, how often do we check for new frames
        frameTimeInterval = 1/self.videoFps
        prev_frame = time.time()
        next_frame = prev_frame + frameTimeInterval

        # actual_prev_frame and actual_next_frame are for determining the actual fps/frame time when displayed. The deviation from the actual fps/frametime values is because
        # of rendering and image processings applied to the image
        actual_prev_frame = prev_frame
        actual_next_frame = 0

        bool_DisplayingVideo = True

        # startTime, endTime, frameCount are used for debugging (checking average fps)
        debug_startTime = time.time()
        debug_frameCount = 0

        while bool_DisplayingVideo:

            # if its next time for the next frame to be drawn
            if time.time() >= next_frame:

                # Capture frame-by-frame
                ret, frame = self.video.read()

                # if frame is read correctly/ new frame is available, then ret is True.
                if not ret:
                    print("Can't receive frame. Exiting ...")
                    bool_DisplayingVideo = False
                    continue

                # set new frame target (next_frame) after check if new frame is available instead of before. This is to make sure when live capturing
                # the newest image will be shown as soon as they are ready. If this was to be put before checking if new frame is available, newest
                # frame have to wait till the next frameTimeInterval to be shown, which increases latency
                prev_frame = next_frame
                next_frame = prev_frame + frameTimeInterval

                # Our operations on the frame come here
                debug_frameCount = debug_frameCount + 1

                frame = resizeMinTo500(frame)
                frameMgnr = ImageManager(frame)
                # startTime = time.time()
                faces = frameMgnr.findFacesCounterClockwiseMultipleAngles(
                    (0, 45), 1.2, 10)
                # print(f"Runtime {time.time() - startTime:0.6f} seconds")
                for face in faces:
                    face.draw(frame, (255, 255, 255), 2)

                # Calculating fps, timeFrame, video speed
                actual_next_frame = time.time()
                frameTime = actual_next_frame - actual_prev_frame

                actual_prev_frame = actual_next_frame
                fps = 1/frameTime
                videoSpeed = frameTimeInterval * 100 / frameTime

                # Display the resulting frame
                cv.putText(frame, f"DEBUG {fps:0.2f} FPS", (12, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(frame, f"DEBUG FrameTime {frameTime*1000:0.2f}ms",
                           (12, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(frame, f"DEBUG video speed {videoSpeed:0.2f}%", (
                    12, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.imshow('video unoptimized', frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    bool_DisplayingVideo = False
                    continue

        print(f"DEBUG Video length {time.time() - debug_startTime:0.2f}")
        print(
            f"DEBUG Average fps {debug_frameCount / (time.time() - debug_startTime):0.2f} ")

    def displayOptimizedMethod(self):
        """
        SHOW THEM THE "SOMEWHAT" BEST OF YOU
        This method uses the facial detection pipeline written in the ImageManager class (look for eyes, find the pairs, crop out potential faces to run haarcascade facial detection on that area)
        By nature, this method is much more computationally heavier than just running the haarcascade facial detection directly on the image. So instead of running it on every frame, this method only 
        run once its previous call has finished running. This will result in choppy face detection framerate (8 -> 15 fps). 
        """
        # faces is an array that store the most recent position of the detected faces
        faces = []
        # outputQueue is used for storing threaded functions output
        outputQueue = queue.Queue()
        # this is a thread
        faceDetectionThread = threading.Thread()
        faceDetectionThread.start()

        # frameTimeInterval, prev_frame, and next_frame are used to determine the frame rate of the video, how often do we check for new frames
        frameTimeInterval = 1/self.videoFps
        prev_frame = time.time()
        next_frame = prev_frame + frameTimeInterval

        # actual_prev_frame and actual_next_frame are for determining the actual fps/frame time when displayed. The deviation from the actual fps/frametime values is because
        # of rendering and image processings applied to the image
        actual_prev_frame = prev_frame
        actual_next_frame = 0

        bool_DisplayingVideo = True

        # startTime, endTime, frameCount are used for debugging (checking average fps)
        debug_startTime = time.time()
        debug_frameCount = 0
        debug_frameSkip = 0

        while bool_DisplayingVideo:

            # if its next time for the next frame to be drawn
            if time.time() >= next_frame:

                # Capture frame-by-frame
                ret, frame = self.video.read()

                # if frame is read correctly/ new frame is available, then ret is True.
                if not ret:
                    print("Can't receive frame. Exiting ...")
                    bool_DisplayingVideo = False
                    continue

                # resizing the frame to fit my tiny monitor, and to increase haarcascade detection performance too ye
                frame = resizeMinTo500(frame)

                # set new display time target (next_frame) after check if new frame is available instead of before. This is to make sure when live capturing
                # the newest image will be shown as soon as they are ready (variable fps maybe). If this was to be put before checking if new frame is available, newest
                # frame have to wait till the next frameTimeInterval to be shown if it misses the frame display time target, which increases latency (MAYBE :thonking:?)
                prev_frame = next_frame
                next_frame = prev_frame + frameTimeInterval

                # Our operations on the frame come here
                debug_frameCount = debug_frameCount + 1

                # variable face detection scanning fps
                if not faceDetectionThread.is_alive():
                    print(f"DEBUG frameSkip {debug_frameSkip + 1} frames")
                    debug_frameSkip = 0
                    # getting the most recent faces' positions
                    if not outputQueue.empty():
                        faces = outputQueue.get()

                    frameMgnr = ImageManager(frame)
                    args = [(45, 0, -45), 1.2, 10]
                    faceDetectionThread = threading.Thread(
                        target=self.HELPER_findFacesCounterClockwiseMultipleAngles_to_outputQueue, args=[frameMgnr, args, outputQueue])
                    faceDetectionThread.start()
                else:
                    debug_frameSkip = debug_frameSkip + 1

                for face in faces:
                    face.draw(frame, (255, 255, 255), 2)

                # Calculating fps, timeFrame, video speed
                actual_next_frame = time.time()
                frameTime = actual_next_frame - actual_prev_frame

                actual_prev_frame = actual_next_frame
                fps = 1/frameTime
                videoSpeed = frameTimeInterval * 100 / frameTime

                # Display the resulting frame
                cv.putText(frame, f"DEBUG {fps:0.2f} FPS", (12, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(frame, f"DEBUG FrameTime {frameTime*1000:0.2f}ms",
                           (12, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(frame, f"DEBUG video speed {videoSpeed:0.2f}%", (
                    12, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.imshow('video optimized', frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    bool_DisplayingVideo = False
                    continue

        print(f"DEBUG Video length {time.time() - debug_startTime:0.2f}")
        print(
            f"DEBUG Average fps {debug_frameCount / (time.time() - debug_startTime):0.2f} ")

    def displayEvenMoreOptimizedMethod(self):
        """
        SHOW THEM THE BEST OF YOU
        This method uses the facial detection pipeline (macroFacialDetection) written in the ImageManager class (look for eyes, find the pairs, crop out potential faces to run haarcascade facial detection on that area)
        By nature, this method is much more computationally heavier than just running the haarcascade facial detection directly on the image. So instead of running it on every frame, this method only 
        run once its previous call has finished running. This will result in choppy face detection framerate (8 -> 15 fps). 

        To improve the choppy framerate, I use past locations and run basic haarcascade facial detection (microFacialDetection) on those areas only. Then run eye detection in the new detected faces to update the angles of the faces.
        """
        # currentDetectedFacesManager is an array that store the tuples of most recent position of the detected faces and an int indicating to whether should the face be displayed on the frame (0 means yes, > 0 means no)
        currentDetectedFacesManager = []
        # while shadowDetectedFaces store the faces detected by macro facial detection, which lags 2-6 frames behind
        shadowDetectedFaces = []
        # outputQueue is used for storing threaded functions output
        outputQueue = queue.Queue()
        # this is a thread
        faceDetectionThread = threading.Thread()
        faceDetectionThread.start()

        # frameTimeInterval, prev_frame, and next_frame are used to determine the frame rate of the video, how often do we check for new frames
        frameTimeInterval = 1/self.videoFps
        prev_frame = time.time()
        next_frame = prev_frame + frameTimeInterval

        # actual_prev_frame and actual_next_frame are for determining the actual fps/frame time when displayed. The deviation from the actual fps/frametime values is because
        # of rendering and image processings applied to the image
        actual_prev_frame = prev_frame
        actual_next_frame = 0

        bool_DisplayingVideo = True

        # startTime, endTime, frameCount are used for debugging (checking average fps)
        debug_startTime = time.time()
        debug_frameCount = 0
        debug_frameSkip = 0

        while bool_DisplayingVideo:

            # print(f"DEBUG currentDetectedFacesManager length {len(currentDetectedFacesManager)}")
            # print(f"DEBUG shadowDetectedFaces length {len(shadowDetectedFaces)}")

            # if its next time for the next frame to be drawn
            if time.time() >= next_frame:

                # Capture frame-by-frame
                ret, frame = self.video.read()

                # if frame is read correctly/ new frame is available, then ret is True.
                if not ret:
                    print("Can't receive frame. Exiting ...")
                    bool_DisplayingVideo = False
                    continue
                
                startTime = time.time()
                # resizing the frame to fit my tiny monitor, and to increase haarcascade detection performance too ye
                min500frame = resizeMinTo500(frame)
                resizeTime = time.time() - startTime

                # set new display time target (next_frame) after check if new frame is available instead of before. This is to make sure when live capturing
                # the newest image will be shown as soon as they are ready (variable fps maybe). If this was to be put before checking if new frame is available, newest
                # frame have to wait till the next frameTimeInterval to be shown if it misses the frame display time target, which increases latency (MAYBE :thonking:?)
                prev_frame = next_frame
                next_frame = prev_frame + frameTimeInterval

                # Our operations on the frame come here
                debug_frameCount = debug_frameCount + 1

                # variable face detection scanning fps
                if not faceDetectionThread.is_alive():
                    # --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- #

                    # If the previous thoroughly facial detection (my implementation) has finished running
                    # print(f"DEBUG frameSkip {debug_frameSkip + 1} frames")
                    debug_frameSkip = 0

                    # delayDetectedFaces is the array that store detectedArea objs returned by previously called findFacesCounterClockwiseMultipleAngles function.
                    # These objs dont represent the current faces, but faces found in 2-6 frames previously when the function was called
                    delayDetectedFaces = []
                    # getting the most recent thoroughly detected faces' positions
                    if not outputQueue.empty():
                        delayDetectedFaces = outputQueue.get()

                        # print(f"DEBUG delayDetectedFaces length {len(delayDetectedFaces)}")

                    # Merging the macro facial detection into the micro facial detection list
                    if len(shadowDetectedFaces) == 0:
                        shadowDetectedFaces = delayDetectedFaces
                        for shadowFace in shadowDetectedFaces:
                            currentDetectedFacesManager.append(
                                [shadowFace.copy(), 0])
                    elif len(delayDetectedFaces) != 0:

                        # check if there's a new face detected by macro facial detection
                        for shadowFace in shadowDetectedFaces:

                            bool_innerLoop = False
                            i = 0
                            if i < len(delayDetectedFaces):
                                bool_innerLoop = True

                            while bool_innerLoop:
                                delayFace = delayDetectedFaces[i]
                                # if the face returned by macro facial detection has already exists then pop it off
                                if shadowFace.similarSize(delayFace, self.HARDCODED_similarSizeScale) and shadowFace.overlap(delayFace):
                                    delayDetectedFaces.pop(i)
                                    i = i - 1
                                i = i + 1
                                if i >= len(delayDetectedFaces):
                                    bool_innerLoop = False

                        # if there are faces that haven't been detected then add them to the currentDetectedFacesManager list
                        if len(delayDetectedFaces) != 0:
                            for newFace in delayDetectedFaces:
                                currentDetectedFacesManager.append(
                                    [newFace, 0])
                    else:
                        # ?????????????????????????????????????????????????????????????
                        shadowDetectedFaces = [] 

                    # Run the next macro facial detection (my implementation) in a seperate thread to avoid interfering with updating new frames
                    frameMgnr = ImageManager(min500frame)
                    args = [(45, 0, -45), 1.1, 10]
                    faceDetectionThread = threading.Thread(
                        target=self.HELPER_findFacesCounterClockwiseMultipleAngles_to_outputQueue, args=[frameMgnr, args, outputQueue])
                    faceDetectionThread.start()

                    # update the shadowDetectedFace list for when the thread finished running
                    for currentFaceManager in currentDetectedFacesManager:
                        currentFace = currentFaceManager[0]
                        shadowDetectedFaces.append(currentFace.copy())

                else:

                    debug_frameSkip = debug_frameSkip + 1

                    # --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- #

                    # updating the detected faces in the last frame by searching around their area
                    if len(currentDetectedFacesManager) != 0:
                        bool_loop = True
                        index_currentDetectedFacesManager = 0
                        if index_currentDetectedFacesManager >= len(currentDetectedFacesManager):
                            bool_loop = False
                        while bool_loop:
                            face = currentDetectedFacesManager[index_currentDetectedFacesManager][0]

                            dimensions = face.dimensions
                            counterClockwiseAngle = face.counterClockwiseAngle
                            faceCenter = Point(face.center.x, face.center.y)

                            # rotate clockwise because the face's angle is the counter clockwise angle of the right eye to the left eye
                            frameOrigin = Point(
                                min500frame.shape[1]/2, min500frame.shape[0]/2)
                            rotatedFrame = rotateClockwise(
                                min500frame, counterClockwiseAngle)
                            rotatedFrameOrigin = Point(
                                rotatedFrame.shape[1]/2, rotatedFrame.shape[0]/2)

                            # rotate clockwise the faceCenter and project it to the new image
                            rotatedFaceCenter = faceCenter.rotatePointClockwise(
                                frameOrigin, counterClockwiseAngle)
                            rotatedFaceCenter = rotatedFaceCenter.projectPoint(
                                frameOrigin, rotatedFrameOrigin)

                            # Calculate the region of the potential face
                            searchingDimensions = (dimensions[0] * self.HARDCODED_updateFaceLocationSearchMultiplier,
                                                   dimensions[1] * self.HARDCODED_updateFaceLocationSearchMultiplier)
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
                                grayCroppedPotentialFace, 1.3, 6)

                            # if found nothing then move to the next face's last frame location
                            if len(detectedFaces) == 0:

                                # increment the faceNotFoundCount
                                faceNotFoundCount = currentDetectedFacesManager[
                                    index_currentDetectedFacesManager][1]
                                faceNotFoundCount = faceNotFoundCount + 1
                                if faceNotFoundCount >= self.HARDCODED_faceNotFoundCountLimit:

                                    currentDetectedFacesManager.pop(
                                        index_currentDetectedFacesManager)
                                    # if there's still more currentFaceManager to iterate
                                    if index_currentDetectedFacesManager < len(currentDetectedFacesManager):
                                        continue

                                    # if there's no more currentFaceManager to iterate then break the loop
                                    bool_loop = False
                                    continue

                                # update faceNotFoundCount
                                currentDetectedFacesManager[index_currentDetectedFacesManager][1] = faceNotFoundCount

                                # if still under the faceNotFoundLimit then continue iterating
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
                                grayCroppedPotentialFace, 1.3, 10)
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
                                if eye.similarSize(referenceEye, self.HARDCODED_similarSizeScale):
                                    detectedEyes.append(eye)

                            # find the pairs of eyes
                            if len(detectedEyes) > 1:
                                # This might be slightlyyyyy overkill but usually there's like at most 2-3 eyes so it wont be too computationally heavy.
                                for i in range(len(detectedEyes)):
                                    j = i + 1
                                    for j in range(len(detectedEyes)):
                                        if detectedEyes[i].appropriateDistanceTo(detectedEyes[j], self.HARDCODED_pairOfEyesDistanceRange[0], self.HARDCODED_pairOfEyesDistanceRange[1]):
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
                                            if upperLeftDist < upperRightDist and upperLeftDist > upperRightDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                                criterias = criterias + 1
                                            if upperRightDist <= upperLeftDist and upperRightDist > upperLeftDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                                criterias = criterias + 1
                                            if lowerLeftDist < lowerRightDist and lowerLeftDist > lowerRightDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
                                                criterias = criterias + 1
                                            if lowerRightDist <= lowerLeftDist and lowerRightDist > lowerLeftDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
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

                            # update the face's location and reset faceNotFoundCount value to 0
                            currentDetectedFacesManager[index_currentDetectedFacesManager][0] = biggestFace
                            currentDetectedFacesManager[index_currentDetectedFacesManager][1] = 0

                            # continue iterating the currentFaceManager
                            index_currentDetectedFacesManager = index_currentDetectedFacesManager + 1
                            if index_currentDetectedFacesManager >= len(currentDetectedFacesManager):
                                bool_loop = False

                for faceManager in currentDetectedFacesManager:
                    face = faceManager[0]
                    face.draw(min500frame, (255, 255, 255), 2)

                # Calculating fps, timeFrame, video speed
                actual_next_frame = time.time()
                frameTime = actual_next_frame - actual_prev_frame

                actual_prev_frame = actual_next_frame
                fps = 1/frameTime
                videoSpeed = frameTimeInterval * 100 / frameTime

                # Display the resulting frame
                cv.putText(min500frame, f"DEBUG {fps:0.2f} FPS", (
                    12, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(min500frame, f"DEBUG FrameTime {frameTime*1000:0.2f}ms",
                           (12, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(min500frame, f"DEBUG video speed {videoSpeed:0.2f}%", (
                    12, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.imshow('video EVEN MORE optimized', min500frame)


                totalTime = time.time() - startTime

                # print("REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n")
                # print(resizeTime*100/totalTime)
                # print("REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n")

                if cv.waitKey(1) & 0xFF == ord('q'):
                    bool_DisplayingVideo = False
                    continue

        print(f"DEBUG Video length {time.time() - debug_startTime:0.2f}")
        print(
            f"DEBUG Average fps {debug_frameCount / (time.time() - debug_startTime):0.2f} ")

    # def correctSpeed_displayEvenMoreOptimizedMethod(self):
    #     """
    #     SHOW THEM THE BEST OF YOU
    #     This method uses the facial detection pipeline (macroFacialDetection) written in the ImageManager class (look for eyes, find the pairs, crop out potential faces to run haarcascade facial detection on that area)
    #     By nature, this method is much more computationally heavier than just running the haarcascade facial detection directly on the image. So instead of running it on every frame, this method only
    #     run once its previous call has finished running. This will result in choppy face detection framerate (8 -> 15 fps).

    #     To improve the choppy framerate, I use past locations and run basic haarcascade facial detection (microFacialDetection) on those areas only. Then run eye detection in the new detected faces to update the angles of the faces.
    #     """
    #     # currentDetectedFacesManager is an array that store the tuples of most recent position of the detected faces and an int indicating to whether should the face be displayed on the frame (0 means yes, > 0 means no)
    #     currentDetectedFacesManager = []
    #     # while shadowDetectedFaces store the faces detected by macro facial detection, which lags 2-6 frames behind
    #     shadowDetectedFaces = []
    #     # outputQueue is used for storing threaded functions output
    #     outputQueue = queue.Queue()
    #     # this is a thread
    #     faceDetectionThread = threading.Thread()
    #     faceDetectionThread.start()

    #     # frameTimeInterval, prev_frame, and next_frame are used to determine the frame rate of the video, how often do we check for new frames
    #     frameTimeInterval = 1/self.videoFps
    #     prev_frame = time.time()
    #     next_frame = prev_frame + frameTimeInterval

    #     # actual_prev_frame and actual_next_frame are for determining the actual fps/frame time when displayed. The deviation from the actual fps/frametime values is because
    #     # of rendering and image processings applied to the image
    #     actual_prev_frame = prev_frame
    #     actual_next_frame = 0

    #     bool_DisplayingVideo = True

    #     # startTime, endTime, frameCount are used for debugging (checking average fps)
    #     debug_startTime = time.time()
    #     debug_frameCount = 0
    #     debug_frameSkip = 0

    #     while bool_DisplayingVideo:

    #         # print(f"DEBUG currentDetectedFacesManager length {len(currentDetectedFacesManager)}")
    #         # print(f"DEBUG shadowDetectedFaces length {len(shadowDetectedFaces)}")

    #         # Capture frame-by-frame
    #         ret, frame = self.video.read()

    #             # if frame is read correctly/ new frame is available, then ret is True.
    #         if not ret:
    #             print("Can't receive frame. Exiting ...")
    #             bool_DisplayingVideo = False
    #             continue

    #         # if its next time for the next frame to be drawn
    #         if time.time() >= next_frame:

    #         # resizing the frame to fit my tiny monitor, and to increase haarcascade detection performance too ye
    #         min500frame = resizeMinTo500(frame)

    #         # set new display time target (next_frame) after check if new frame is available instead of before. This is to make sure when live capturing
    #         # the newest image will be shown as soon as they are ready (variable fps maybe). If this was to be put before checking if new frame is available, newest
    #         # frame have to wait till the next frameTimeInterval to be shown if it misses the frame display time target, which increases latency (MAYBE :thonking:?)
    #         prev_frame = time.time()
    #         next_frame = prev_frame + frameTimeInterval

    #         # Our operations on the frame come here
    #         debug_frameCount = debug_frameCount + 1

    #         # variable face detection scanning fps
    #         if not faceDetectionThread.is_alive():
    #             # --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- MACRO FACIAL DETECTION SECTION --- #

    #             # If the previous thoroughly facial detection (my implementation) has finished running
    #             # print(f"DEBUG frameSkip {debug_frameSkip + 1} frames")
    #             debug_frameSkip = 0

    #             # delayDetectedFaces is the array that store detectedArea objs returned by previously called findFacesCounterClockwiseMultipleAngles function.
    #             # These objs dont represent the current faces, but faces found in 2-6 frames previously when the function was called
    #             delayDetectedFaces = []
    #             # getting the most recent thoroughly detected faces' positions
    #             if not outputQueue.empty():
    #                 delayDetectedFaces = outputQueue.get()

    #                 # print(f"DEBUG delayDetectedFaces length {len(delayDetectedFaces)}")

    #             # Merging the macro facial detection into the micro facial detection list
    #             if len(shadowDetectedFaces) == 0:
    #                 shadowDetectedFaces = delayDetectedFaces
    #                 for shadowFace in shadowDetectedFaces:
    #                     currentDetectedFacesManager.append([shadowFace.copy(), 0])
    #             elif len(delayDetectedFaces) != 0:

    #                 # check if there's a new face detected by macro facial detection
    #                 for shadowFace in shadowDetectedFaces:

    #                     bool_innerLoop = False
    #                     i = 0
    #                     if i < len(delayDetectedFaces):
    #                         bool_innerLoop = True

    #                     while bool_innerLoop:
    #                         delayFace = delayDetectedFaces[i]
    #                         # if the face returned by macro facial detection has already exists then pop it off
    #                         if shadowFace.similarSize(delayFace, self.HARDCODED_similarSizeScale) and shadowFace.overlap(delayFace):
    #                             delayDetectedFaces.pop(i)
    #                             i = i - 1
    #                         i = i + 1
    #                         if i >= len(delayDetectedFaces):
    #                             bool_innerLoop = False

    #                 # if there are faces that haven't been detected then add them to the currentDetectedFacesManager list
    #                 if len(delayDetectedFaces) != 0:
    #                     for newFace in delayDetectedFaces:
    #                         currentDetectedFacesManager.append([newFace, 0])

    #             # Run the next macro facial detection (my implementation) in a seperate thread to avoid interfering with updating new frames
    #             frameMgnr = ImageManager(min500frame)
    #             args = [(45, 0, -45), 1.1, 10]
    #             faceDetectionThread = threading.Thread(target = self.HELPER_findFacesCounterClockwiseMultipleAngles_to_outputQueue, args= [frameMgnr, args, outputQueue])
    #             faceDetectionThread.start()

    #             # update the shadowDetectedFace list for when the thread finished running
    #             for currentFaceManager in currentDetectedFacesManager:
    #                 currentFace = currentFaceManager[0]
    #                 shadowDetectedFaces.append(currentFace.copy())

    #         else:

    #             debug_frameSkip = debug_frameSkip + 1

    #             # --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- MICRO FACIAL DETECTION SECTION --- #

    #             # updating the detected faces in the last frame by searching around their area
    #             if len(currentDetectedFacesManager) != 0:
    #                 bool_loop = True
    #                 index_currentDetectedFacesManager = 0
    #                 if index_currentDetectedFacesManager >= len(currentDetectedFacesManager):
    #                     bool_loop = False
    #                 while bool_loop:
    #                     face = currentDetectedFacesManager[index_currentDetectedFacesManager][0]

    #                     dimensions = face.dimensions
    #                     counterClockwiseAngle = face.counterClockwiseAngle
    #                     faceCenter = Point(face.center.x, face.center.y)

    #                     # rotate clockwise because the face's angle is the counter clockwise angle of the right eye to the left eye
    #                     frameOrigin = Point(min500frame.shape[1]/2, min500frame.shape[0]/2)
    #                     rotatedFrame = rotateClockwise(min500frame, counterClockwiseAngle)
    #                     rotatedFrameOrigin = Point(rotatedFrame.shape[1]/2, rotatedFrame.shape[0]/2)

    #                     # rotate clockwise the faceCenter and project it to the new image
    #                     rotatedFaceCenter = faceCenter.rotatePointClockwise(frameOrigin, counterClockwiseAngle)
    #                     rotatedFaceCenter = rotatedFaceCenter.projectPoint(frameOrigin, rotatedFrameOrigin)

    #                     # Calculate the region of the potential face
    #                     searchingDimensions = (dimensions[0] * self.HARDCODED_updateFaceLocationSearchMultiplier, dimensions[1] * self.HARDCODED_updateFaceLocationSearchMultiplier)
    #                     cropRangeMinY = int(max(0, rotatedFaceCenter.y - searchingDimensions[1] / 2))
    #                     cropRangeMaxY = int(min(rotatedFrame.shape[0], rotatedFaceCenter.y + searchingDimensions[1] / 2))

    #                     cropRangeMinX = int(max(0, rotatedFaceCenter.x - searchingDimensions[0] / 2))
    #                     cropRangeMaxX = int(min(rotatedFrame.shape[1], rotatedFaceCenter.x + searchingDimensions[0] / 2 ))

    #                     # Cropping out the region of the potential face
    #                     try:
    #                         croppedPotentialFace = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX), dtype='uint8')
    #                         croppedPotentialFace[0: croppedPotentialFace.shape[0], 0: croppedPotentialFace.shape[1]] = \
    #                             rotatedFrame[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]

    #                     except:
    #                         croppedPotentialFace = np.zeros((cropRangeMaxY - cropRangeMinY, cropRangeMaxX - cropRangeMinX, 3), dtype ='uint8')
    #                         croppedPotentialFace[0: croppedPotentialFace.shape[0], 0: croppedPotentialFace.shape[1]] = \
    #                             rotatedFrame[cropRangeMinY:cropRangeMaxY, cropRangeMinX:cropRangeMaxX]

    #                     croppedPotentialFaceCenter = Point(croppedPotentialFace.shape[1]/2 , croppedPotentialFace.shape[0]/2)

    #                     # gray scale the image
    #                     grayCroppedPotentialFace = cv.cvtColor(croppedPotentialFace, cv.COLOR_BGR2GRAY)
    #                     # run basic haarcascade facial detection on it
    #                     # CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ?
    #                     detectedFaces = haarcascade_FacialDetection(grayCroppedPotentialFace, 1.3, 6)

    #                     # if found nothing then move to the next face's last frame location
    #                     if len(detectedFaces) == 0:

    #                         # increment the faceNotFoundCount
    #                         faceNotFoundCount = currentDetectedFacesManager[index_currentDetectedFacesManager][1]
    #                         faceNotFoundCount = faceNotFoundCount + 1
    #                         if faceNotFoundCount >= self.HARDCODED_faceNotFoundCountLimit:

    #                             currentDetectedFacesManager.pop(index_currentDetectedFacesManager)
    #                             # if there's still more currentFaceManager to iterate
    #                             if index_currentDetectedFacesManager < len(currentDetectedFacesManager):
    #                                 continue

    #                             # if there's no more currentFaceManager to iterate then break the loop
    #                             bool_loop = False
    #                             continue

    #                         # update faceNotFoundCount
    #                         currentDetectedFacesManager[index_currentDetectedFacesManager][1] = faceNotFoundCount

    #                         # if still under the faceNotFoundLimit then continue iterating
    #                         continue

    #                     # if found anything, then convert the rectangle coordinates received from detectMultiScale function to a detectedFace obj.
    #                     biggestFace = detectedFaces[0]
    #                     for i in range(1, len(detectedFaces)):
    #                         # comparing the dimensions (width only, cause if width is > then height is also >) among all the faces found.
    #                         if biggestFace[2] < detectedFaces[i][2]:
    #                             biggestFace = detectedFaces[i]

    #                     # Creating DetectedFace obj representing the biggest face found in the potential region above
    #                     biggestFace = DetectedFace((biggestFace[0], biggestFace[1]), (biggestFace[2], biggestFace[3]))

    #                     # Finding biggestFace's angle
    #                     # CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ? CAN IMPROVE PERFORMANCE BY ADDING APPROPRIATE MINSIZE AND MAXSIZE ?

    #                     # find the eyes
    #                     tempEyesArray = haarcascade_EyeDetection(grayCroppedPotentialFace, 1.3, 10)
    #                     detectedEyes = []

    #                     if face.leftEye != None:
    #                         referenceEye = face.leftEye
    #                     elif face.rightEye != None:
    #                         referenceEye = face.rightEye
    #                     else:
    #                         faceUpperLeftPoint = biggestFace.upperLeft
    #                         # if the face has no eyes found, then create a standard reference eye
    #                         # HARDCODED ___ Eye's width is usually at most 3/8 of face's width. Eye's height is usually at most 1/3 of face's height
    #                         referenceEye = DetectedArea((faceUpperLeftPoint.x + biggestFace.dimensions[0]/8, faceUpperLeftPoint.y + biggestFace.dimensions[1]/6), (biggestFace.dimensions[0]* 3/ 8, biggestFace.dimensions[1]/3) )

    #                     for (x, y, w, h)in tempEyesArray:
    #                         eye = DetectedArea((x, y), (w, h))
    #                         # If the shape and size of the eye found in the image is approximately the same as the reference eye
    #                         if eye.similarSize(referenceEye, self.HARDCODED_similarSizeScale):
    #                             detectedEyes.append(eye)

    #                     # find the pairs of eyes
    #                     if len(detectedEyes) > 1:
    #                         # This might be slightlyyyyy overkill but usually there's like at most 2-3 eyes so it wont be too computationally heavy.
    #                         for i in range(len(detectedEyes)):
    #                             j = i + 1
    #                             for j in range(len(detectedEyes)):
    #                                 if detectedEyes[i].appropriateDistanceTo(detectedEyes[j], self.HARDCODED_pairOfEyesDistanceRange[0], self.HARDCODED_pairOfEyesDistanceRange[1]):
    #                                     leftEye = detectedEyes[i]
    #                                     rightEye = detectedEyes[j]
    #                                     if detectedEyes[i].center.x > detectedEyes[j].center.x:
    #                                         leftEye = detectedEyes[j]
    #                                         rightEye = detectedEyes[i]
    #                                     # if the distance of LeftEye's center to face's upperLeft Point ~ that of RightEye's center to face's upperRight Point
    #                                     # and distance of LeftEye's center to face's lowerLeft Point ~ that of RightEye's center to face's lowerRight Point
    #                                     # then this is probably the right pair of eyes
    #                                     upperLeftDist = leftEye.center.distTo(biggestFace.upperLeft)
    #                                     upperRightDist = rightEye.center.distTo(biggestFace.upperRight)
    #                                     lowerLeftDist = leftEye.center.distTo(biggestFace.lowerLeft)
    #                                     lowerRightDist = rightEye.center.distTo(biggestFace.lowerRight)
    #                                     criterias = 0
    #                                     if upperLeftDist < upperRightDist and upperLeftDist > upperRightDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
    #                                         criterias = criterias + 1
    #                                     if upperRightDist <= upperLeftDist and upperRightDist > upperLeftDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
    #                                         criterias = criterias + 1
    #                                     if lowerLeftDist < lowerRightDist and lowerLeftDist > lowerRightDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
    #                                         criterias = criterias + 1
    #                                     if lowerRightDist <= lowerLeftDist and lowerRightDist > lowerLeftDist * self.HARDCODED_distanceVariation_EyeToCorners_MinOfMax:
    #                                         criterias = criterias + 1
    #                                     if criterias == 2:
    #                                         biggestFace.leftEye = leftEye
    #                                         biggestFace.rightEye = rightEye
    #                                         break

    #                     # find the face's angle using the pair of eyes
    #                     if biggestFace.leftEye != None and biggestFace.rightEye != None:
    #                         # calculate teh relative angle created by the two eyes
    #                         relativeCounterClockwiseAngle = biggestFace.leftEye.center.relativeCounterClockwiseAngle(biggestFace.rightEye.center)
    #                         # the face's current angle is the already rotated angle pre-haarcascade detection + the relative angle
    #                         biggestFace.counterClockwiseAngle = counterClockwiseAngle + relativeCounterClockwiseAngle
    #                         biggestFace.rotateAreaCounterClockwise(biggestFace.center, relativeCounterClockwiseAngle)
    #                     else:
    #                         # if there's not a pair of eyes then the face's angle = the previous face that was detected in the same area's angle
    #                         biggestFace.counterClockwiseAngle = counterClockwiseAngle

    #                     # Returning the face's coordinates to the original frame
    #                     biggestFace.projectArea(croppedPotentialFaceCenter, rotatedFaceCenter)
    #                     biggestFace.rotateAreaCounterClockwise(rotatedFrameOrigin, biggestFace.counterClockwiseAngle)
    #                     biggestFace.projectArea(rotatedFrameOrigin, frameOrigin)

    #                     # update the face's location and reset faceNotFoundCount value to 0
    #                     currentDetectedFacesManager[index_currentDetectedFacesManager][0] = biggestFace
    #                     currentDetectedFacesManager[index_currentDetectedFacesManager][1] = 0

    #                     # continue iterating the currentFaceManager
    #                     index_currentDetectedFacesManager = index_currentDetectedFacesManager + 1
    #                     if index_currentDetectedFacesManager >= len(currentDetectedFacesManager):
    #                         bool_loop = False

    #         for faceManager in currentDetectedFacesManager:
    #             face = faceManager[0]
    #             face.draw(min500frame, (255, 255, 255), 2)

    #         # Calculating fps, timeFrame, video speed
    #         actual_next_frame = time.time()
    #         frameTime = actual_next_frame - actual_prev_frame

    #         actual_prev_frame = actual_next_frame
    #         fps = 1/frameTime
    #         videoSpeed = frameTimeInterval * 100 / frameTime

    #         # Display the resulting frame
    #         cv.putText(min500frame, f"DEBUG {fps:0.2f} FPS", (12, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    #         cv.putText(min500frame, f"DEBUG FrameTime {frameTime*1000:0.2f}ms", (12, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    #         cv.putText(min500frame, f"DEBUG video speed {videoSpeed:0.2f}%", (12, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    #         cv.imshow('video EVEN MORE optimized', min500frame)

    #         if cv.waitKey(1) & 0xFF == ord('q'):
    #             bool_DisplayingVideo = False
    #             continue

    #     print(f"DEBUG Video length {time.time() - debug_startTime:0.2f}")
    #     print(f"DEBUG Average fps {debug_frameCount / (time.time() - debug_startTime):0.2f} ")

    def HELPER_findFacesCounterClockwiseMultipleAngles_to_outputQueue(self, frameManager, arguments, outputQueue):
        """
        Help multithreading a function by being a dummy function which only purpose is to call the threadedly-desired? function and save its output into the
        argument queue instead of returning it.
            :param function: the function that needs to be multithreaded
            :param arguments: a list of function's arguments []
            :param outputQueue: the queue that stores the function output
        """
        outputQueue.put(frameManager.findFacesCounterClockwiseMultipleAngles(
            arguments[0], arguments[1], arguments[2]))



  