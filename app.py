from socket import socket
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2 as cv
import base64
import io
import os
import time
import numpy as np
from PIL import Image
from glyLib.ServerVideoManager import ServerVideoManager
from glyLib import GlyLibrary

app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

if __name__ == '__main__':
    socketio.run(app, port=int(os.environ.get('PORT', 8080)))

@app.route('/')
def index():
    return render_template("index.html")


# @socketio.on('connect')
# def handle_connection():
#     print("connected")


# @socketio.on('chatMessage')
# def handle_chatMessage(msg, id):
#     print("client said:", msg)
#     socketio.emit('response', msg, to=id)


# @socketio.on('disconnect')
# def handle_disconnect():
#     print('disconnected')

########################
SVMDict = dict()


@socketio.on('connect')
def handle_connect():
    print("-----------------------------------------------------\n---------      CONNECTION ESTABLISHED      ----------\n-----------------------------------------------------")
 
 
@socketio.on('initialize')
def handle_initialize(clientID, frameRate, isEnhanced): 
    print("initialize")
    SVMDict[clientID] = ServerVideoManager(frameRate, socketio, clientID, isEnhanced)
    SVMDict[clientID].start()



@socketio.on('cleanup')
def handle_cleanup(clientID):
    if clientID in SVMDict and SVMDict[clientID] != None:
        SVMDict[clientID].stop()
        SVMDict[clientID] = None


@socketio.on('disconnect')
def handle_disconnect():
    print("-----------------------------------------------------\n---------      CONNECTION DISCONNECTED      ----------\n-----------------------------------------------------")




@socketio.on('frameToServer')
def handle_frame_to_server(clientID, type_and_base64_frame, frameID):
    
    if not (clientID in SVMDict):
        print("ERROR in frameToServer")
        print("ClientID", clientID, "does not exist in the SVMDict")

    elif SVMDict[clientID] == None:
        print("EXCEPTION in frameToServer")
        print("SVM instance for clientID", clientID, "has already been cleared. This might be because client lost connection")
    else:
        start_time = time.time()
        SVMDict[clientID].processNextFrame(detectFaceVanilla, type_and_base64_frame, frameID, 'frameToClient')
        print("----------- %s seconds -----------" % (time.time() - start_time))

@socketio.on('frameToServer_coordinates')
def handle_frame_to_server(clientID, type_and_base64_frame, frameID):

    if not (clientID in SVMDict):
        print("ERROR in frameToServer")
        print("ClientID", clientID, "does not exist in the SVMDict")

    elif SVMDict[clientID] == None:
        print("EXCEPTION in frameToServer")
        print("SVM instance for clientID", clientID, "has already been cleared. This might be because client lost connection")

    else:
        start_time = time.time()
        SVMDict[clientID].processNextFrame(detectFaceVanilla_coordinates, type_and_base64_frame, frameID, 'frameToClient_coordinates')
        print("----------- %s seconds -----------" % (time.time() - start_time))

def detectFaceVanilla(type_and_base64_image):
    
    
    type, base64_image = type_and_base64_image.split(',')

    # Creating a PIL image (RGB)
    imgData = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(imgData))

    # Converting to BGR format that openCV reads, then convert to grayscale to increase faces detected
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    GlyLibrary.findAndDrawFacesVanilla(img, 'classifier/haarcascade_frontalface_default.xml')


    buffered = io.BytesIO()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(img))
    img.save(buffered, format='JPEG')
    
    b64_newImage = base64.b64encode(buffered.getvalue())
    return type + ',' + str(b64_newImage)[2:-1]

    # socketio.emit('messageResponse',type + ',' + str(b64_newImage)[2:-1])

def detectFaceVanilla_coordinates(type_and_base64_image):
    
    type, base64_image = type_and_base64_image.split(',')

    # Creating a PIL image (RGB)
    imgData = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(imgData))

    # Converting to BGR format that openCV reads, then convert to grayscale to increase faces detected
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    temp = GlyLibrary.findAndDrawFacesVanilla_coordinates(img, 'classifier/haarcascade_frontalface_default.xml')
    
    return temp

