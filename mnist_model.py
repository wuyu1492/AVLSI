from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Masking
from utils import *
from keras import backend as K
from keras.engine.topology import Layer
import sys


def weight_prune(model, layer_name, psize):
    layer = model.layers



def get_model():
    print("Building model...")
    inputs = Input(shape=(28,28,1))
    conv0 = Conv2D(32, (5,5), padding='same',input_shape=(28,28,1),name="conv0")(inputs)
    relu0 = Activation("relu")(conv0)
    maxpool0 = MaxPooling2D(pool_size=5, strides=2)(relu0)

    conv3 = Conv2D(64, (3,3), padding='same', name="conv3")(maxpool0)
    relu3 = Activation("relu")(conv3)
    maxpool1 = MaxPooling2D(pool_size=3, strides=2)(relu3)

    conv4 = Conv2D(128, (3,3), padding='same', name="conv4")(maxpool1)
    relu4 = Activation("relu")(conv4)
    conv5 = Conv2D(128, (3,3), padding='same', name="conv5")(relu4)
    relu5 = Activation("relu")(conv5)
    maxpool2 = MaxPooling2D(pool_size=3, strides=1)(relu5)
    drop2 = Dropout(0.2)(maxpool2)

    flatten = Flatten()(drop2)
    dense0 = Dense(1024, name='dense0')(flatten)
    relu6 = Activation("relu")(dense0)
    drop3 = Dropout(0.2)(relu6)
    dense1 = Dense(10, name='dense1')(drop3)
    softmax = Activation("softmax")(dense1)

    model = Model(inputs=inputs, outputs=softmax)
    return model

def train(mode="pretrain"):
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # 28*28
    x_train = np.expand_dims(x_train, axis=3).astype("float64")
    x_test = np.expand_dims(x_test, axis=3).astype("float64")
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train -= np.mean(x_train, axis=(1,2,3), keepdims=True)
    x_train /= np.std(x_train, axis=(1,2,3), keepdims=True)

    print("Datashape: ", x_train.shape, y_train.shape)
    
    model = get_model()
    model.compile(loss="categorical_crossentropy",
            optimizer='rmsprop',
            metrics=['accuracy'])
    model.summary()
    print(model.layers[14].get_config()['name'])

    #train model
    batch = 512
    epo = 10
#    history = model.fit(x_train,
#            y_train,
#            batch_size=batch,
#            epochs=epo,
#            validation_data=(x_test, y_test))
    p_size = 992
    if mode=='prune':
        model = load_model("model_mnist.h5")
        model = zero_weight(model, "dense0", 992)
        #model = zero_channels(model, "conv4", 64)
        model.compile(loss="categorical_crossentropy",
                optimizer='rmsprop',
                metrics=['accuracy'])
        model.summary()
        acc = model.evaluate(x_test, y_test, batch_size=1024)
        print("Post pruning accuracy =", acc)
        model.save("model_mnist_prune.h5")
    elif mode=="fine-tune":
        model = load_model("model_mnist_prune.h5")
        epo = 5
        history = model.fit(x_train,
                y_train,
                batch_size=batch,
                epochs=epo,
                validation_data=(x_test, y_test))
        model.save("model_mnist_"+str(p_size)+".h5")
    else:
        epo = 10
        history = model.fit(x_train,
                y_train,
                batch_size=batch,
                epochs=epo,
                validation_data=(x_test, y_test))
        model = load_model("model_mnist.h5")

        

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "pretrain"
    train(mode)

if __name__=="__main__":
    main()
