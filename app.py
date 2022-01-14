
from flask import Flask, Response, render_template
from flask_socketio import SocketIO
import cv2 as cv
import base64
import io
import numpy as np
from PIL import Image
from glyFacialDetection import GlyLibrary
from glyFacialDetection.ServerVideoManager import ServerVideoManager

from threading import Thread
import time

from werkzeug.wrappers import request


app = Flask(__name__, template_folder="templates")

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
SVMDict = dict()
vm = None
# tempThread = None

if __name__ == '__main__':
    socketio.run(app)

@socketio.on('/')
def handle_connect():
    return 'heroku'
 

@socketio.on('connect')
def handle_connect():
    print("-----------------------------------------------------")
    print("---------      CONNECTION ESTABLISHED      ----------")
    print("-----------------------------------------------------")
 
 
@socketio.on('initialize')
def handle_initialize(clientID, frameRate, isEnhanced): 
    SVMDict[clientID] = ServerVideoManager(frameRate, socketio, clientID, isEnhanced)
    SVMDict[clientID].start()


    # tempThread = Thread(target=emitter, args=[vm])
    # tempThread.start()
@socketio.on('cleanup')
def handle_cleanup(clientID):
    if SVMDict[clientID] != None:
        SVMDict[clientID].stop()
        SVMDict[clientID] = None


@socketio.on('disconnect')
def handle_disconnect():
    return
 




@socketio.on('frameToServer')
def handle_frame_to_server(clientID, type_and_base64_frame, frameID):
    data = {'base64_responseFrame': type_and_base64_frame, 'frameID': frameID}
    socketio.emit('frameToClient', data, to=clientID)
    # SVMDict[clientID].processNextFrame(detectFaceVanilla, type_and_base64_frame, frameID, 'frameToClient')

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

def detectFaceVanillaGrayScale(type_and_base64_image):
    
    
    type, base64_image = type_and_base64_image.split(',')

    # Creating a PIL image (RGB)
    imgData = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(imgData))

    # Converting to BGR format that openCV reads, then convert to grayscale to increase faces detected
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    GlyLibrary.findAndDrawFacesVanilla(img)


    buffered = io.BytesIO()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = Image.fromarray(np.uint8(img))
    img.save(buffered, format='JPEG')
    
    b64_newImage = base64.b64encode(buffered.getvalue())
    return type + ',' + str(b64_newImage)[2:-1]

##############################################################################

# @socketio.on('connect')
# def handle_connect():
#     print("-----------------------------------------------------")
#     print("---------      CONNECTION ESTABLISHED      ----------")
#     print("-----------------------------------------------------")
 

# @socketio.on('initialize')
# def handle_initialize(frameRate):
#     global vm, tempThread
#     print("-------------", frameRate)
#     vm = ServerVideoManager(frameRate, socketio)
#     vm.start()
#     # tempThread = Thread(target=emitter, args=[vm])
#     # tempThread.start()


# @socketio.on('disconnect')
# def handle_disconnect():
#     if vm != None:
#         vm.stop()
#     print("*****************************************************")
#     print("*********        SOCKET HAS CLOSED        ***********")
#     print("*****************************************************")
 




# @socketio.on('frameToServer')
# def handle_connect(type_and_base64_image):
#     vm.processNextFrame(detectFaceVanilla, type_and_base64_image, 'frameToClient')
    

# def detectFaceVanilla(type_and_base64_image):
    
    
#     type, base64_image = type_and_base64_image.split(',')

#     # Creating a PIL image (RGB)
#     imgData = base64.b64decode(base64_image)
#     img = Image.open(io.BytesIO(imgData))

#     # Converting to BGR format that openCV reads, then convert to grayscale to increase faces detected
#     img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
#     # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     GlyLibrary.findAndDrawFacesVanilla(img)


#     buffered = io.BytesIO()
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img = Image.fromarray(np.uint8(img))
#     img.save(buffered, format='JPEG')
    
#     b64_newImage = base64.b64encode(buffered.getvalue())
#     return type + ',' + str(b64_newImage)[2:-1]

#     # socketio.emit('messageResponse',type + ',' + str(b64_newImage)[2:-1])


