from threading import Thread
import threading

class FrameThread(Thread):
    def __init__(self, threadID, frameID, customFunction, frame, frameLock, endPoint):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.frameID = frameID
        self.task = customFunction
        self.frame = frame
        self.frameLock = frameLock
        self.endPoint = endPoint
        self.output = None

    def run(self):
        self.output = self.task(self.frame)
        # Stop thread from stop running => changing the VideoManager frameThread array status
        # Affecting the EmittingThread from getting correct values and send to the clients
        # self.frameLock.acquire()
        # self.frameLock.release()
        

