import numpy
import tensorflow
import cv2
from .Variables import *

class PreprocessInput:
    def __init__(self,face_cascade):
        print("🚀🚀Image Processor is connected🚀🚀")
        self.face_cascade=face_cascade
    def preprocessInput(self,image_path):
        image_array=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        faces_rect = self.face_cascade.detectMultiScale(image_array)
        for (x, y, w, h) in faces_rect:
            image_array=image_array[y:y+h,x:x+w]
            print(image_array)
            image_array=cv2.resize(image_array,resize_dimension)
            image_array=numpy.expand_dims([image_array],axis=-1)
            return numpy.array([image_array])
        return False
    def preprocessTransferLearningInput(self,image_path):
        image_array=cv2.imread(image_path,cv2.IMREAD_COLOR)
        faces_rect = self.face_cascade.detectMultiScale(image_array)
        for (x, y, w, h) in faces_rect:
            image_array=image_array[y:y+h,x:x+w]
            print(image_array)
            image_array=cv2.resize(image_array,transfer_learning_dimension)
            print(image_array.shape)
            image_array=numpy.array([image_array])
            return image_array
        return False


class ModelProcessor:
    def __init__(self):
        print("🚀🚀🚀 Model Processor is Initialized🚀🚀")
    def setModel(self,ModelPath):
        self.model=tensorflow.keras.models.load_model(ModelPath)
        print("Model is Connected")
    def predictClass(self,image_array):
        print(image_array.shape)
        prediction=self.model.predict(image_array)
        print(prediction)
        prediction=numpy.argmax(prediction[0])
        print(prediction)
        return class_list[prediction]
    def predictClasses(self,array_of_images):
        prediction=self.model.predict(array_of_images)
        prediction=[class_list[numpy.argmax(values)] for values in prediction]
        return prediction

