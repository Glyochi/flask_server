import math
import cv2 as cv
import numpy as np

haar_cascasde_eye = cv.CascadeClassifier("classifier/haarcascade_eye.xml")
# Default frontal face xml => more accurate but slower
# haar_cascasde_face = cv.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")
# Lighter frontal face xml => less accurate (accurate enuf) but faster
haar_cascasde_face = cv.CascadeClassifier("classifier/lbpcascaade_frontalface_improved.xml")


# Haarcascade basic detection
def haarcascade_FacialDetection(image, scaleFactor, minNeighbors, minSize = None, maxSize = None):
    """
    This is to be used by videoManager only because there seems to be some error when used in imageManager (caused by multithreading probably)
    Just run the haarcascade facial detection on the image without modifying the input image
        :return the detectMultiScale output
    """
    if minSize == None:
        if maxSize == None:
            return haar_cascasde_face.detectMultiScale(image, scaleFactor, minNeighbors)
        return haar_cascasde_face.detectMultiScale(image, scaleFactor, minNeighbors, maxSize = maxSize)
    if maxSize == None:
        return haar_cascasde_face.detectMultiScale(image, scaleFactor, minNeighbors, minSize = minSize)
    return haar_cascasde_face.detectMultiScale(image, scaleFactor, minNeighbors, minSize = minSize, maxSize = maxSize)


def haarcascade_EyeDetection(image, scaleFactor, minNeighbors, minSize = None, maxSize = None):
    """
    Just run the haarcascade eye detection on the image without modifying the input image
        :return the detectMultiScale output
    """
    if minSize == None:
        if maxSize == None:
            return haar_cascasde_eye.detectMultiScale(image, scaleFactor, minNeighbors)
        return haar_cascasde_eye.detectMultiScale(image, scaleFactor, minNeighbors, maxSize = maxSize)
    if maxSize == None:
        return haar_cascasde_eye.detectMultiScale(image, scaleFactor, minNeighbors, minSize = minSize)
    return haar_cascasde_eye.detectMultiScale(image, scaleFactor, minNeighbors, minSize = minSize, maxSize = maxSize)
    


# Rotation
def rotateClockwise(img, angle):
    """
    This functions return a clockwise rotated image without cropping any part of the original image and keeping the center of the original 
    image at the center of the new rotated image. The returned image may have larger dimensions than the original but as small as possible.
        :param img: the original image
        :param angle: the rotating angle clockwise
        :return: the rotated-around-the-center image without cropping
    """
    angle = -angle
    return rotateCounterClockwise(img, angle)


def rotateCounterClockwise(img, angle):
    """
    This functions return a counter-clockwise rotated image without cropping any part of the original image and keeping the center of the original 
    image at the center of the new rotated image. The returned image may have larger dimensions than the original but as small as possible.
        :param img: the original image
        :param angle: the rotating angle counter-clockwise
        :return: the rotated-around-the-center image without cropping
    """
    (height, width) = img.shape[:2]

    # Calculating the dimension of the new canvas to store the entire rotated image without cropping, diagAngleA is used to calculate fittingWidth, diagAngleB is for fittingHeight
    diagLen = math.sqrt(width**2 + height**2)
    diagAngleA = math.atan(width/height)
    diagAngleB = math.pi/2 - diagAngleA

    # TempAngle is used to keep the angle value in math.cos stays between -90 and 90 degree => consistence cos value 
    # cos value is the cosine of the offset of frame/img diagonal line from the x/y axis (diagAngle + tempAngle*math.pi/180 - math.pi/2). 
    # And using the value of angle, we will know which cos value is used to determined width and height.
    # angle < 360 to avoid edge cases
    angle = angle % 360
    tempAngle = angle%90
    if angle == 0 or angle == 180:
        fittingWidth = width
        fittingHeight = height
    elif angle == 90 or angle == 270:
        fittingWidth = height
        fittingHeight = width
    elif (angle > 0 and angle < 90) or (angle > 180 and angle < 270):
        
        fittingWidth = math.floor(math.cos(diagAngleA + tempAngle*math.pi/180 - math.pi/2) * diagLen)
        fittingHeight = math.floor(math.cos(diagAngleB + tempAngle*math.pi/180 - math.pi/2) * diagLen)
    else :
        fittingWidth = math.floor(math.cos(diagAngleB + tempAngle*math.pi/180 - math.pi/2) * diagLen)
        fittingHeight = math.floor(math.cos(diagAngleA + tempAngle*math.pi/180 - math.pi/2) * diagLen)

    # Drawing pre rotated image ontop of bigger canvas
    # rotatedCanvas initially takes the largest horizontal/vertical value among newWidth, newHeight, img.shape[1], img.shape[2] to prevent the image from cropping and index out of bound
    # Once the image is rotated on rotatedCanvas, we will crop out the excess part
    try:
        rotatedCanvas = np.zeros((max(fittingHeight, img.shape[0]), max(fittingWidth, img.shape[1]), 3), dtype='uint8')
        rotatedCanvas[math.floor(rotatedCanvas.shape[0]/2 - img.shape[0]/2): math.floor(rotatedCanvas.shape[0]/2 + img.shape[0]/2), math.floor(rotatedCanvas.shape[1]/2 - img.shape[1]/2): math.floor(rotatedCanvas.shape[1]/2 + img.shape[1]/2)] = img
    except:
        rotatedCanvas = np.zeros((max(fittingHeight, img.shape[0]), max(fittingWidth, img.shape[1])), dtype='uint8')
        rotatedCanvas[math.floor(rotatedCanvas.shape[0]/2 - img.shape[0]/2): math.floor(rotatedCanvas.shape[0]/2 + img.shape[0]/2), math.floor(rotatedCanvas.shape[1]/2 - img.shape[1]/2): math.floor(rotatedCanvas.shape[1]/2 + img.shape[1]/2)] = img
    # rotatedCanvas[100:200,400:500] = 0,255,0
    # [upper height bound: lower height bound, left width bound: right width bound]
    # cv.imshow("Test draw on cavnas before rotate", rescaleFrame(rotatedCanvas, 500))

    #rotPoint (width, height)
    rotPoint = (rotatedCanvas.shape[1]//2, rotatedCanvas.shape[0]//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    excessDimensions = (rotatedCanvas.shape[1], rotatedCanvas.shape[0])
    rotatedCanvas = cv.warpAffine(rotatedCanvas, rotMat, excessDimensions)
    # rotatedCanvas = rotatedCanvas[100:1200, 100:600]
    # Cropping excess borders
    rotatedCanvas = rotatedCanvas[math.floor((rotatedCanvas.shape[0] - fittingHeight)/2) : math.floor((rotatedCanvas.shape[0] + fittingHeight)/2), math.floor((rotatedCanvas.shape[1] - fittingWidth)/2) : math.floor((rotatedCanvas.shape[1] + fittingWidth)/2)]
    return rotatedCanvas





# Resize
def resizeMinTo500(img):
    """
    Create a resized clone of the original image such that the smaller dimension of the image = 500px, and the other dimension is kept to scale
        :param img: the image need to be resized
        :return a resized copy of the iamge
    """
    
    (height, width) = img.shape[:2]

    resizedImage = img.copy()

    if height < width:
        width = int(width * 500/ height)
        height = 500
        
    else:
        height = int(height * 500/ width)
        width = 500
        
    dimensions = (width, height)
    return cv.resize(resizedImage, dimensions, interpolation= cv.INTER_LINEAR)

# Resize
def resizeMinTo(img, size):
    """
    Create a resized clone of the original image such that the smaller dimension of the image = 500px, and the other dimension is kept to scale
        :param img: the image need to be resized
        :return a resized copy of the iamge
    """
    size = int(size)
    (height, width) = img.shape[:2]

    resizedImage = img.copy()

    if height < width:
        width = int(width * size/ height)
        height = size
        
    else:
        height = int(height * size/ width)
        width = size
        
    dimensions = (width, height)
    return cv.resize(resizedImage, dimensions, interpolation=cv.INTER_LINEAR)


# Resize TEST INTERPOLATION SPEED
def TEST_INTERPOLATION_TYPE_resizeMinTo(img, size, interpolationType):
    """
    Create a resized clone of the original image such that the smaller dimension of the image = 500px, and the other dimension is kept to scale
        :param img: the image need to be resized
        :param interapolationType: the interpolation type with
            0: INTER_AREA
            1: INTER_CUBIC
            2: INTER_LANCZOS4
            3: INTER_NEAREST
            4: INTER_LINEAR
        :return a resized copy of the iamge
    """
    size = int(size)
    (height, width) = img.shape[:2]

    resizedImage = img.copy()

    if height < width:
        width = int(width * size/ height)
        height = size
        
    else:
        height = int(height * size/ width)
        width = size
        
    dimensions = (width, height)
    if interpolationType == 0:
        return cv.resize(resizedImage, dimensions, interpolation=cv.INTER_AREA)
    elif interpolationType == 1:
        return cv.resize(resizedImage, dimensions, interpolation=cv.INTER_CUBIC)
    elif interpolationType == 2:
        return cv.resize(resizedImage, dimensions, interpolation=cv.INTER_LANCZOS4)
    elif interpolationType == 3:
        return cv.resize(resizedImage, dimensions, interpolation=cv.INTER_NEAREST)
    elif interpolationType == 4:
        return cv.resize(resizedImage, dimensions, interpolation=cv.INTER_LINEAR)

