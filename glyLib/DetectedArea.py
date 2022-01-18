from .Point import Point
import cv2 as cv

"""
NOTE:
    Only faces have their own class DetectedFace because I want to keep track of the angle of the face using the relative angle between the two eyes (and potentially an eye and a nose or nose and mouth, etc)
    The other facial features such as eyes, noses, mouths dont need to have their own class and can just use DetectedArea to store their locations
"""


class DetectedArea:
    """
    Object that stores the center of a rectangle encapsulates a detected object returned by cv.detectMultiscale(...), and the dimensions of the rectangle.
    OpenCV haarcascade detection returns 4-tuples (x,y,w,h) of rectangle marking down where the detected objects are. These information will be translated 
    to DetectedArea objects.
    """
    def __init__(self, upperLeftPoint = (0,0), dimensions = (0,0)):
        """
        Construct a DetectedArea obj.
            :param upperLeftPoint (x,y): the 2-tuple coordinates of the upper left point of the rectangle encapsulates detected objects.
            :param dimension (w,h): the dimension of the box encapsulates the detected object.
            :param angle: the angle of the image when the detected object was found and returned by openCV.
        """
        self.dimensions = dimensions
        self.upperLeft = Point(upperLeftPoint[0], upperLeftPoint[1])
        self.upperRight = Point(upperLeftPoint[0] + dimensions[0], upperLeftPoint[1])
        self.lowerRight = Point(upperLeftPoint[0] + dimensions[0], upperLeftPoint[1] + dimensions[1])
        self.lowerLeft = Point(upperLeftPoint[0], upperLeftPoint[1] + dimensions[1])
        self.center = Point(upperLeftPoint[0] + dimensions[0]/2, upperLeftPoint[1] + dimensions[1]/2)
        self.radius = self.center.distTo(self.upperLeft)
    
    def copy(self):
        """
        Return a deep copy of the detectedArea caller
            :return a deep copy of itself
        """
        copyArea = DetectedArea()
        copyArea.dimensions = (self.dimensions[0], self.dimensions[1])
        copyArea.upperLeft = self.upperLeft.copy()
        copyArea.upperRight = self.upperRight.copy()
        copyArea.lowerLeft = self.lowerLeft.copy()
        copyArea.lowerRight = self.lowerRight.copy()
        copyArea.center = self.center.copy()
        return copyArea
    
    def rotateAreaCounterClockwise(self, origin, angle):
        """
        Rotate the detectedArea object counter-clockwise around an origin by a given degree.
            :param origin: the point where detectedArea is going to rotate around
            :param angle: the angle the detectedArea is going to rotate by
        """
        self.upperLeft = self.upperLeft.rotatePointCounterClockwise(origin, angle)
        self.upperRight = self.upperRight.rotatePointCounterClockwise(origin, angle)
        self.lowerLeft = self.lowerLeft.rotatePointCounterClockwise(origin, angle)
        self.lowerRight = self.lowerRight.rotatePointCounterClockwise(origin, angle)
        self.center = self.center.rotatePointCounterClockwise(origin, angle)



    def rotateAreaClockwise(self, origin, angle):
        """
        Rotate the detectedArea object clockwise around an origin by a given degree.
            :param origin: the point where detectedArea is going to rotate around
            :param angle: the angle the detectedArea is going to rotate by
        """
        self.upperLeft = self.upperLeft.rotatePointClockwise(origin, angle)
        self.upperRight = self.upperRight.rotatePointClockwise(origin, angle)
        self.lowerLeft = self.lowerLeft.rotatePointClockwise(origin, angle)
        self.lowerRight = self.lowerRight.rotatePointClockwise(origin, angle)
        self.center = self.center.rotatePointClockwise(origin, angle)
        
    def projectArea(self, oldOrigin, newOrigin):
        """
        Project the rectangle such that its new position relative to newOrigin is 
        the same as its current relative position to oldOrigin.
            :param oldOrigin: the old origin point
            :param newOrigin: the projected old Origin
        """        
        self.upperLeft = self.upperLeft.projectPoint(oldOrigin, newOrigin)
        self.upperRight = self.upperRight.projectPoint(oldOrigin, newOrigin)
        self.lowerLeft = self.lowerLeft.projectPoint(oldOrigin, newOrigin)
        self.lowerRight = self.lowerRight.projectPoint(oldOrigin, newOrigin)
        self.center = self.center.projectPoint(oldOrigin, newOrigin)

    def overlap(self, otherArea):
        """
        Check for overlap between two detectedAreas. (NOT FINALIZED CONDITION PARAMETERS)
            :param otherArea: the DetectedArea that we are going to check for overlap
            :return True/False
        """
        # print("DetectedArea overlap: NOT FINALIZED CONDITION PARAMETERS")
        distance = self.center.distTo(otherArea.center)
        if distance < (self.center.distTo(self.upperLeft) + otherArea.center.distTo(otherArea.upperLeft))/4:
            return True
        return False
    
    def similarSize(self, otherArea, scale):
        """
        Compare the size with another area to see if the two DetectedAreas are similar. (NOT FINALIZED CONDITION PARAMETERS)
            :param otherArea: the DetectedArea that we are comparing to
            :param scale: the smallest possible size the smaller area can be compared to the bigger area to be considered "similarSize"
            :return True/False
        """
        thisSize = self.center.distTo(self.upperLeft)
        otherSize = otherArea.center.distTo(otherArea.upperLeft)

        minSize = min(thisSize, otherSize)
        maxSize = max(thisSize, otherSize)

        if minSize > maxSize * scale:
            return True
        return False
    
    def appropriateDistanceTo(self, otherArea, minAverageRadiusDistScale, maxAverageRadiusDistScale):
        """
        Decide whether the two detectedAreas distance to each other is appropriate.
            :param otherArea: the other detectedArea object
            :param minAverageRadiusDistScale: the mininum multiplier for the average radius between two detectedAreas 
            :param maxAverageRadiusDistScale: the maximum multiplier for the average radius between two detectedAreas 
            :return True if the distance between two objects is between average radius * minimum multiplier and average radius * maximum multiplier, False otherwise
        """
        dist = self.center.distTo(otherArea.center)
        averageRadius = (self.radius + otherArea.radius)/2
        if dist < maxAverageRadiusDistScale * averageRadius and dist > minAverageRadiusDistScale * averageRadius:
            return True
        return False

    def merge(self, otherArea):
        """
        Merge another area into itself to make a more accurate area. This does not delete the other area object. (NOT FINALIZED CONDITION PARAMETERS)
        For right now, merge will trust the user to call appropriately since all it does is make the caller object becomes the smaller object between the twos.
            :param otherArea: the DetectedArea that we are merging with
        """
        if self.center.distTo(self.upperLeft) > otherArea.center.distTo(otherArea.upperLeft):
            self = otherArea
        
        

    def draw(self, canvas, color, thickness):
        """
        Draw the shape of the detectedArea on top of the given image with specified color and thickness
            :param canvas: the given image to be drawn over
            :param color: the BGR-color specified for the detectedArea object
            :param thickness: the specified thickness of the detectedArea object
        """
        upperLeft = self.upperLeft.exportCoordinates()
        upperRight = self.upperRight.exportCoordinates()
        lowerLeft = self.lowerLeft.exportCoordinates()
        lowerRight = self.lowerRight.exportCoordinates()
        canvas = cv.line(canvas, upperLeft, upperRight, color, thickness = thickness)
        canvas = cv.line(canvas, upperRight, lowerRight, color, thickness = thickness)
        canvas = cv.line(canvas, lowerRight, lowerLeft, color, thickness = thickness)
        canvas = cv.line(canvas, lowerLeft, upperLeft, color, thickness = thickness)
        


class DetectedFace(DetectedArea):
    def __init__(self, upperLeftPoint = (0,0), dimensions = (0,0), angle = None):
        """
        Create an detectedFace object that is a detectedArea object but with angle property.
            :param upperLeftPoint (x,y): the 2-tuple coordinates of the upper left point of the rectangle encapsulates detected objects.
            :param dimension (w,h): the dimension of the box encapsulates the detected object.
            :param angle: the angle of the image when the detected object was found and returned by openCV.
        """
        super().__init__(upperLeftPoint=upperLeftPoint, dimensions=dimensions)
        # face's angle is the counter clockwise angle of the right eye to the left eye
        self.counterClockwiseAngle = angle
        self.leftEye = None
        self.rightEye = None
        
    
    def copy(self):
        """
        Return a deep copy of the detectedArea caller
            :return a deep copy of itself
        """
        copyArea = DetectedFace()
        copyArea.dimensions = (self.dimensions[0], self.dimensions[1])
        copyArea.upperLeft = self.upperLeft.copy()
        copyArea.upperRight = self.upperRight.copy()
        copyArea.lowerLeft = self.lowerLeft.copy()
        copyArea.lowerRight = self.lowerRight.copy()
        copyArea.center = self.center.copy()
        copyArea.counterClockwiseAngle = self.counterClockwiseAngle
        copyArea.leftEye = self.leftEye
        copyArea.rightEye = self.rightEye
        return copyArea