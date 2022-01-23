from concurrent.futures import thread
from threading import Thread
import threading
from time import sleep
from tkinter.messagebox import NO

class FrameThread(Thread):
    def __init__(self, threadID, frameID, customFunction, args, endPoint):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.frameID = frameID
        self.task = customFunction
        self.args = args
        self.endPoint = endPoint
        self.output = None

    def run(self):
        self.output = self.task(self.args)
        

# class EnhancedFrameThread(Thread):
#     def __init__(self, frameID, customFunction, frameManager, angles, scaleFactor, minNeighbours):
#         threading.Thread.__init__(self)
#         self.frameID = frameID
#         self.task = customFunction
#         self.args = {
#             'frameManager': frameManager,
#             'angles': angles,
#             'scaleFactor': scaleFactor,
#             'minNeighbours': minNeighbours,
#         }
#         self.output = None
#     def run(self):
#         self.output = self.task(
#             self.args.frameManager, self.args.angles, 
#             self.args.scaleFactor, self.args.minNeighbours)