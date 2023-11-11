import cv2
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import adam_v2
from tensorflow.keras.utils import get_file
from keras.utils import np_utils
from PIL import Image

from net.mobilenet import MobileNet
from utils import get_random_data

K.image_data_format() == "channels_first"


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def generate_arrays_from_file(lines,batch_size,train):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []

        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)

            name = lines[i].split(';')[0]
            label = lines[i].split(';')[1]

            if (int(label) == 0):
                img = Image.open("./data/processed/Mask/" + name)
            else:
                img = Image.open("./data/processed/Non Mask/" + name)

            if train == True:
                img = np.array(get_random_data(img,[HEIGHT,WIDTH]), dtype = np.float64)
            else:
                img = np.array(letterbox_image(img,[WIDTH,HEIGHT]), dtype = np.float64)

            X_train.append(img)
            Y_train.append(label)

            i = (i+1) % n


        X_train = preprocess_input(np.array(X_train))
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= NUM_CLASSES)   
        yield (X_train, Y_train)


if __name__ == "__main__":
    HEIGHT = 224
    WIDTH = 224
    NUM_CLASSES = 2
    

    log_dir = "./logs/"


    model = MobileNet(input_shape=[HEIGHT,WIDTH,3],classes=NUM_CLASSES)
    

    model_name = './model_data/mobilenet_1_0_224_tf_no_top.h5'
    model.load_weights(model_name, by_name=True,skip_mismatch=True)


    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


    with open("./data/train.txt","r") as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val


    trainable_layer = 80
    for i in range(trainable_layer):
        model.layers[i].trainable = False
    print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    if True:
        lr = 1e-3
        batch_size = 4
        model.compile(loss = 'categorical_crossentropy',
                optimizer = adam_v2.Adam(lr=lr),
                metrics = ['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[checkpoint, reduce_lr, early_stopping])

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    if True:
        lr = 1e-4
        batch_size = 8
        model.compile(loss = 'categorical_crossentropy',
                optimizer = adam_v2.Adam(lr=lr),
                metrics = ['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=50,
                callbacks=[checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir+'last_one.h5')