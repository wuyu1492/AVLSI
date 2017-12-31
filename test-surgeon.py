import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adadelta
from kerassurgeon.operations import delete_channels
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
    #feature /= (np.std(feature, axis=1).reshape(-1,1) + 1e-8)
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

    # cnn model
    model = Sequential()
    model.add(Conv2D(16,(5,5),padding='same',input_shape=(48,48,1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=5,strides=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(32,(3,3),padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=3,strides=2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=3,strides=2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    model.add(Dense(7))
    model.add(Activation("softmax"))
    
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    model.summary()
    
    # train model
    batch = 512
    epo = 4
    
    history = model.fit(x_train,
            label_train, 
            batch_size=batch,
            epochs=epo,
            validation_data=(x_val, label_val))

    if mode == 'prune' :
        layer_i = 8
        print("Prining layer = {} ".format(layer_i))
        model = delete_channels(model, model.layers[layer_i], [0, 4, 16, 32])
        model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        model.summary()
    
    epo = 1
    history = model.fit(x_train,
            label_train, 
            batch_size=batch,
            epochs=epo,
            validation_data=(x_val, label_val))

    model.save("model_cnn.h5")
    np.save("history_acc.npy", history.history['acc'])
    np.save("history_val_acc.npy", history.history['val_acc'])
    np.save("cnn_x_val.npy", x_val)
    np.save("cnn_label_val.npy", label_val)
    
    #plot_saliency(model, x_val[0], str(np.argmax(label_val[0])))

def main(argv):
    label, feature = read_train(argv[0])
    train_cnn_model(label, feature, argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])
