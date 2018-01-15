import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adadelta
#from utils_mnist import *
import sys
# plot


def read_train(fn):
    df = pd.read_csv(fn, sep=" |,",header=None,
            engine='python',
            skiprows=[0])
    data = df.values
    label_int = data[:,0]
    label = to_categorical(label_int, 7)
    feature = data[:,1:].astype(np.float)
    feature -= np.mean(feature, axis=1).reshape(-1,1)
    feature /= (np.std(feature, axis=1).reshape(-1,1) + 1e-8)
    return label, feature

def show(img, label):
    plt.figure
    plt.imshow(img, cmap='gray')
    label_str = "image with label : " + str(np.argmax(label))
    plt.title(label_str)
    plt.show()
    plt.savefig('show_pic.png')


def train_cnn_model(label, feature, mode):
    print("Mode == ", mode)
    feature = np.reshape(feature, (-1, 48, 48, 1))
    split = 0.8
    split *= len(label)
    split = np.round(split).astype(np.int)
    label_train = label[:split]
    x_train = feature[:split]
    label_val = label[split:]
    x_val = feature[split:]

    def get_model():
        # cnn model
        model = Sequential()
        model.add(Conv2D(64,(5,5),padding='same',input_shape=(48,48,1), name='conv0'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.4))
        model.add(Conv2D(23,(5,5),padding='same', name='conv1'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=5,strides=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(12,(3,3),padding='same', name='conv2'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.4))
        model.add(Conv2D(13,(3,3),padding='same', name='conv3'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=3,strides=2))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(40,(3,3),padding='same', name='conv4'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.4))
        model.add(Conv2D(65,(3,3),padding='same', name='conv5'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=3,strides=2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(76, name='dense0'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.4))
        model.add(Dense(38, name='dense1'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.4))
        model.add(Dense(7, name='dense_o'))
        model.add(Activation("softmax"))
        return model
    
    # train model
    batch = 256
    epo = 50
    
    """
    history = model.fit(x_train,
            label_train, 
            batch_size=batch,
            epochs=epo,
            validation_data=(x_val, label_val))
    """

    if mode == 'prune' :
        model = load_model("tiny_cnn.h5")
        layer_name = "dense0"
        print("Prining layer : {} ".format(layer_name))
        model = zero_weight(model, layer_name)
        model = zero_channels_all(model,psize=0.95)
        model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        model.summary()
    
        acc = model.evaluate(x_val, label_val, batch_size=512)
        print("evaluate merits", acc)
        model.save("tiny_cnn_prune.h5")
    
    elif mode == 'fine-tune':
        model = load_model("tiny_cnn_15.h5")
        model.summary()
        epo=40
        history = model.fit(x_train,
                label_train, 
                batch_size=batch,
                epochs=epo,
                validation_data=(x_val, label_val))
        model.save("tiny_cnn_finetune.h5")
        np.save("retrain_acc"+str(epo)+".npy", history.history['val_acc'])
    else:
        #model = load_model("model_cnn.h5")
        model = get_model()
        #model = load_model("tiny_cnn.h5")
        model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        model.summary()
        history = model.fit(x_train,
                label_train, 
                batch_size=batch,
                epochs=epo,
                validation_data=(x_val, label_val))
        model.save("tiny_cnn.h5")
        np.save("tiny_hist", history.history['val_acc'])

#    np.save("history_acc.npy", history.history['acc'])
#    np.save("history_val_acc.npy", history.history['val_acc'])
#    np.save("cnn_x_val.npy", x_val)
#    np.save("cnn_label_val.npy", label_val)
    
    #plot_saliency(model, x_val[0], str(np.argmax(label_val[0])))

def main(argv):
    label, feature = read_train(argv[0])
    train_cnn_model(label, feature, argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])
