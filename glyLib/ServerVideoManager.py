from threading import Thread, Lock
from .FrameThread import FrameThread
import time
import json

class ServerVideoManager:
    def __init__(self, frameRate, socket, clientID, isEnhanced):
        self.isEnhanced = isEnhanced

        # Used for the emitting thread
        self.playing = True
        self.frameRate = frameRate

        self.frameThreads = [None, None, None, None, None, None, None, None, None, None]
        self.freeThreadIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.threadCount = 10
        self.returnedFrameID = 0
        
        self.socket = socket
        self.clientID = clientID

        # If no new frames for 3 seconds, stop the background thread. * 6 is because background interval = 1/(6*frameRate) when no thread running
        self.maxTimeOut = frameRate * 3 * 6 
        self.timeOutCount = 0
        self.emittingThread = None
        self.frameLock = Lock()


    def start(self):
        self.playing = True
        self.emittingThread = Thread(target=emitter, args=[self])
        self.emittingThread.start()
        
    def stop(self):
        if self.playing:
            self.playing = False



    def processNextFrame(self, customFunction, frame, frameID, endPoint):
        # Its pretty impossible to have all 5 threads taken up
        if not len(self.freeThreadIDs):
            print('*****************************************************************************************\n*  ERROR: in VideoManager file: No more free threads. Sir can I have sum more threads?  *\n*****************************************************************************************\n')
            return

        threadID = self.freeThreadIDs.pop(0)

        self.frameThreads[threadID] = FrameThread(threadID, frameID, customFunction, frame, self.frameLock, endPoint)
        self.frameThreads[threadID].start()
        
    
    



def emitter(videoManager):
    
    while videoManager.playing: 
        if len(videoManager.freeThreadIDs) < videoManager.threadCount:
            # Reset time out count
            # videoManager.timeOutCount = 0

            i = 0
            frameThreads = videoManager.frameThreads
            freeThreadIDs = videoManager.freeThreadIDs
            finishedThreadIDs = []


            videoManager.frameLock.acquire()
            
            while i < len(frameThreads):

                # If thread t has finished running
                
                if frameThreads[i] != None and not frameThreads[i].is_alive():
                    # add the finished thread id onto the freeThreadIDs attribute of the videoManager
                    freeThreadIDs.append(i)

                    # add the finished thread id onto the finishedThreadArray thats gonna be used later
                    finishedThreadIDs.append(i)

                i += 1            

            videoManager.frameLock.release()
            

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
                    temp = frameThreads[finishedThreadIDs[latestFrameIDIndex]]
                    videoManager.returnedFrameID = latestFrameID

                    if videoManager.isEnhanced:
                        data = {'faceCoordinates': temp.output, 'frameID': temp.frameID}
                        videoManager.socket.emit(temp.endPoint, data, to=videoManager.clientID)
                    else:
                        data = {'base64_responseFrame': temp.output, 'frameID': temp.frameID}
                        videoManager.socket.emit(temp.endPoint, data, to=videoManager.clientID)

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
                



