import cv2
import numpy as np
from keras.layers import (Conv2D, Dense, Flatten, Input, MaxPool2D, Permute)
from keras.layers.advanced_activations import PReLU
from keras.models import Model

import utils


# roughly get the face box (first net of MTCNN)
def create_Pnet(weight_path):
    inputs = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # no activation function, linear
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# Gets the box with higher precision (second net of MTCNN)
def create_Rnet(weight_path):
    inputs = Input(shape=[24, 24, 3])
    # 24,24,3 -> 22,22,28 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    # 11,11,28 -> 9,9,48 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)

    # 128 -> 2
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    # 128 -> 4
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# improve accuracy again and get five feature points (third net of MTCNN)
def create_Onet(weight_path):
    inputs = Input(shape = [48,48,3])
    # 48,48,3 -> 46,46,32 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 23,23,32 -> 21,21,64 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)

    # 3,3,128 -> 128,12,12
    x = Permute((3,2,1))(x)
    x = Flatten()(x)

    # 1152 -> 256
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    # 256 -> 2 
    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    # 256 -> 4 
    bbox_regress = Dense(4,name='conv6-2')(x)
    # 256 -> 10 
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([inputs], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# encapsualte the mtcnn network
class mtcnn():
    # upload the model
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # normaliztion
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        # get all the scales
        scales = utils.calculateScales(img)

        out = []
        for scale in scales:
            # using scale to change the image
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = np.expand_dims(scale_img, 0)
            # use the first network
            ouput = self.Pnet.predict(inputs)
            ouput = [ouput[0][0], ouput[1][0]]
            out.append(ouput)

        rectangles = []
        # get all the rectangles from pnet
        for i in range(len(scales)):
            # probability that has face
            cls_prob = out[i][0][:, :, 1]
            # position of the rectangle
            roi = out[i][1]
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            # encoding 
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        # get the max rectangle from that scale
        rectangles = np.array(utils.NMS(rectangles, 0.7))
        if len(rectangles) == 0:
            return rectangles

        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # rnet need the exact 24*24 image
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        cls_prob, roi_prob = self.Rnet.predict(np.array(predict_24_batch))
       # encoding 
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles

        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # onet need the exact 48*48 image
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        cls_prob, roi_prob, pts_prob = self.Onet.predict(np.array(predict_batch))
        
        # encoding
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles