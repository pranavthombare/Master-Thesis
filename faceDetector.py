import cv2

# HaarCascade Face Detector
class faceDetector: 
    
    def __init__(self, path):
        self.faceCascade = cv2.CascadeClassifier(path)
        
    def detect(self, image, scaleFactor= 1.02, minNeb = 15, minSize = (100, 100)):
            
        rects =  self.faceCascade.detectMultiScale(image, scaleFactor= scaleFactor,
                                                   minNeighbors= minNeb, minSize= minSize,
                                                   flags= cv2.CASCADE_SCALE_IMAGE)
        
        return rects


#### NOTE
####
####
#### Self is used as an instance of the class.
#### def __init__ is the constructor in terms of OOP. Clarify later what it is in terms of FP
#### diff between "self" and "__init__" is that init will be automatically called upon 
#### the creation of an object. (will be called when self is called)