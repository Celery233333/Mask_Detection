import cv2 as cv
import os
import utils
import numpy as np
from net.mtcnn import mtcnn
import os

class face():

    # initialize the models
    def __init__(self):

        # set up the mtcnn
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5,0.5,0]

    # detect and process human face from the input image
    def face_detect(self,path,choose):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # detect all the faces in the image
        faces = self.mtcnn_model.detectFace(img, self.threshold)
        #print(path.split("/")[-1])
        if len(faces) > 0:
            print(1)
            face = utils.face_process(faces[0],img)
            face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
            if (choose == 1):
                cv.imwrite("./data/face/Mask/"+path.split("/")[-1],face)
            else:
                cv.imwrite("./data/face/Non Mask/"+path.split("/")[-1],face)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    celery = face()
    after_generate1 = os.listdir("./data/denoise/Mask")
    after_generate2 = os.listdir("./data/denoise/Non Mask")
    for image in after_generate1:
        celery.face_detect("./data/denoise/Mask/"+image,1)
    for image in after_generate2:
        celery.face_detect("./data/denoise/Non Mask/"+image,2)