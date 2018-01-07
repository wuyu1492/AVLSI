from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Masking
from utils_mnist import *
from keras import backend as K
from keras.engine.topology import Layer
import sys


def weight_prune(model, layer_name, psize):
    layer = model.layers



def get_model():
    print("Building model...")
    model = Sequential()
    model.add(Dense(512,input_shape=(28*28,),name="dense0"))
    model.add(Activation("relu"))
    model.add(Dense(256,name="dense1"))
    model.add(Activation("relu"))
    model.add(Dense(128,name="dense2"))
    model.add(Activation("relu"))
    model.add(Dense(128,name="dense3"))
    model.add(Activation("relu"))
    model.add(Dense(64,name="dense4"))
    model.add(Activation("relu"))
    model.add(Dense(64,name="dense5"))
    model.add(Activation("relu"))
    model.add(Dense(32,name="dense6"))
    model.add(Activation("relu"))
    model.add(Dense(10, name='dense_o'))
    model.add(Activation("softmax"))

    return model

def train(mode="pretrain"):
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # 28*28
    x_train = np.reshape(x_train,(-1, 28*28)).astype("float64")
    x_test = np.reshape(x_test, (-1,28*28)).astype("float64")
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train -= np.mean(x_train, axis=1, keepdims=True)
    x_train /= np.std(x_train, axis=1, keepdims=True)

    print("Datashape: ", x_train.shape, y_train.shape)
    
#    model = get_model()
#    model.compile(loss="categorical_crossentropy",
#            optimizer='rmsprop',
#            metrics=['accuracy'])
#    model.summary()
#    print(model.layers[14].get_config()['name'])

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
        model.summary()
        epo = 5
        history = model.fit(x_train,
                y_train,
                batch_size=batch,
                epochs=epo,
                validation_data=(x_test, y_test))
        model.save("model_mnist_"+str(p_size)+".h5")
    else:
        model = get_model()
        model.compile(loss="categorical_crossentropy",
                optimizer='rmsprop',
                metrics=['accuracy'])
        model.summary()
        epo = 10
        history = model.fit(x_train,
                y_train,
                batch_size=batch,
                epochs=epo,
                validation_data=(x_test, y_test))
        model.save("model_mnist.h5")

        

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "pretrain"
    train(mode)

if __name__=="__main__":
    main()
