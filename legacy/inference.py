from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model,load_model
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
#import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer


from fr_utils import *
from inception_blocks import *


class FaceNetModel:

    def __init__(self):
        self.FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        self.FRmodel.compile(optimizer = 'adam', loss = self.triplet_loss, metrics = ['accuracy'])
        load_weights_from_FaceNet(self.FRmodel)

    def triplet_loss(self,y_true, y_pred, alpha = 0.2):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss,0))
        return loss

    def returnModel(self):
        return self.FRmodel


    def verify(self,image_path, identity, database, FRmodel):
        encoding = img_to_encoding(image_path=image_path,model=self.FRmodel)
        dist = np.linalg.norm(encoding - database[identity])
        return dist

    def recognize(self,image_path):
        import pickle
        handle = open("encoding.pickle","rb")
        database = pickle.load(handle)
        handle.close()
        
        encoding = img_to_encoding(image_path=image_path,model=self.FRmodel)
        min_dist = 100
        person = None
        for (key,val) in database.items():
            dist = np.linalg.norm(encoding - val)
            print("Key - {}, Distance:= {}".format(key,dist))
            if dist<min_dist:
                person = key
                min_dist = dist
                
                
        handle = open("encoding.pickle","wb")
        pickle.dump(database,handle,protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        if min_dist>0.85:
            person = None
        return person
