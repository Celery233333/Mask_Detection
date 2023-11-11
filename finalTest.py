import cv2 as cv
import numpy as np
import os
from net.mobilenet import MobileNet
from keras.applications.imagenet_utils import preprocess_input

if __name__ == "__main__":
    mask_model = MobileNet(input_shape=[224,224,3], classes=2)
    mask_model.load_weights("./logs/last_one.h5")
    success=0

    after_generate = os.listdir("./data/face/Non Mask/")
    for image in after_generate:
        img = cv.imread("./data/face/Non Mask/"+image)
        img = cv.resize(img,(224,224))
        img = preprocess_input(np.reshape(np.array(img, np.float64),[1, 224, 224, 3]))
        name = np.argmax(mask_model.predict(img)[0])
        if name == 1 :
            success = success + 1
    
    after_generate1 = os.listdir("./data/face/Mask/")
    for image in after_generate1:
        img = cv.imread("./data/face/Mask/"+image)
        img = cv.resize(img,(224,224))
        img = preprocess_input(np.reshape(np.array(img, np.float64),[1, 224, 224, 3]))
        name = np.argmax(mask_model.predict(img)[0])
        if name == 0 :
            success = success + 1

    total = 100
    print("Accuracy rate: ", float(success/total))