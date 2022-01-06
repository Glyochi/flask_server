
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


app = Flask(__name__, template_folder="templates")

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

vm = None
# tempThread = None

if __name__ == '__main__':
    socketio.run(app)


@socketio.on('connect')
def handle_connect(frameRate):
    
    print("-----------------------------------------------------")
    print("---------      CONNECTION ESTABLISHED      ----------")
    print("-----------------------------------------------------")
 
@socketio.on('initialize')
def handle_initialize(frameRate):
    global vm, tempThread
    print("-------------", frameRate)
    vm = ServerVideoManager(frameRate, socketio)
    vm.start()
    # tempThread = Thread(target=emitter, args=[vm])
    # tempThread.start()


@socketio.on('disconnect')
def handle_disconnect():
    if vm != None:
        vm.stop()
    print("*****************************************************")
    print("*********        SOCKET HAS CLOSED        ***********")
    print("*****************************************************")
 




@socketio.on('frameToServer')
def handle_connect(type_and_base64_image):
    vm.processNextFrame(detectFaceVanilla, type_and_base64_image, 'frameToClient')
    

def detectFaceVanilla(type_and_base64_image):
    
    
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
    img = Image.fromarray(np.uint8(img))
    img.save(buffered, format='JPEG')
    
    b64_newImage = base64.b64encode(buffered.getvalue())
    return type + ',' + str(b64_newImage)[2:-1]

    # socketio.emit('messageResponse',type + ',' + str(b64_newImage)[2:-1])













################################################################################

# from flask import Flask, Response, render_template
# from flask_socketio import SocketIO
# import cv2 as cv
# import base64
# import io
# import numpy as np
# from PIL import Image
# from glyFacialDetection import VideoManagerForServer, ServerVideoManager

# from threading import Thread
# import time


# app = Flask(__name__, template_folder="templates")

# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, cors_allowed_origins="*")

# vmfs = VideoManagerForServer.VideoManagerForServer()
# vm = None
# # tempThread = None

# if __name__ == '__main__':
#     socketio.run(app)


# @socketio.on('connect')
# def handle_connect():
#     global vm, tempThread
#     vm = ServerVideoManager.ServerVideoManager(socketio)
#     vm.start()
#     # tempThread = Thread(target=emitter, args=[vm])
#     # tempThread.start()
#     print("-----------------------------------------------------")
#     print("---------      CONNECTION ESTABLISHED      ----------")
#     print("-----------------------------------------------------")
 
# @socketio.on('disconnect')
# def handle_disconnect():
#     vm.stop()
#     print("*****************************************************")
#     print("*********        SOCKET HAS CLOSED        ***********")
#     print("*****************************************************")
 




# @socketio.on('message')
# def handle_connect(message):

#     vm.processNextFrame(customFunc, 2)
#     vm.processNextFrame(customFunc, 1)
#     vm.processNextFrame(customFunc, 0)
    

# def customFunc(message):
#     if message == 0:
#         print("-----")
#     elif message == 1:
#         time.sleep(0.01)
#         print("-----")
#     else:
#         time.sleep(0.02)
#         print("-----")
        


#######################################################################################















# from flask import Flask, Response, render_template
# from flask_socketio import SocketIO
# import cv2 as cv
# import base64
# import io
# import numpy as np
# from PIL import Image
# from glyFacialDetection import VideoManagerForServer

# from threading import Thread

# app = Flask(__name__, template_folder="templates")

# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, cors_allowed_origins="*")

# vmfs = VideoManagerForServer.VideoManagerForServer()

# if __name__ == '__main__':
#     socketio.run(app)


# @socketio.on('connect')
# def handle_connect(socket):
#     print("-----------------------------------------------------")
#     print("---------      CONNECTION ESTABLISHED      ----------")
#     print("-----------------------------------------------------")
 
# @socketio.on('disconnect')
# def handle_connect():
#     print("*****************************************************")
#     print("*********        SOCKET HAS CLOSED        ***********")
#     print("*****************************************************")
 




# @socketio.on('message')
# def handle_connect(message):

    
#     type, base64_image = message.split(',')

#     # Creating a PIL image (RGB)
#     imgData = base64.b64decode(base64_image)
#     img = Image.open(io.BytesIO(imgData))

#     # Converting to BGR format that openCV reads, then convert to grayscale to increase faces detected
#     img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
#     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     vmfs.processNextFrame(img)


#     buffered = io.BytesIO()
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img = Image.fromarray(np.uint8(img))
#     img.save(buffered, format='JPEG')
    
#     b64_newImage = base64.b64encode(buffered.getvalue())

#     socketio.emit('messageResponse',type + ',' + str(b64_newImage)[2:-1])

#     # cv.imshow("Reee", img)
#     # while(True):
#     #     if cv.waitKey(1) & 0xFF == ord('q'): # wait for 1 millisecond
#     #         break
#     # print("afasf\n")


