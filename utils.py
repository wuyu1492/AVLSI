import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from kerassurgeon.operations import delete_channels

def zero_weight(model,layer_name, psize):
    print("Estimate weight in DNN, layer[{}]".format(layer_name))
    layer = model.get_layer(name=layer_name)
    print("layer type = ", layer.__class__.__name__)
    weights = layer.get_weights()
    weight = weights[0]
    print("weight shape ", weight.shape)
    node_sum = np.sum(np.absolute(weight), axis=0)
    idx_sort = np.argsort(node_sum, axis=-1)
    pidx = idx_sort[:psize]
    print("pruning channel:", pidx)
    model = delete_channels(model, layer, pidx.tolist())
    return model

def zero_channels(model, layer_name, psize):
    print("Estimate weight in CNN, layer[{}]".format(layer_name))
    layer = model.get_layer(name=layer_name)
    weights = layer.get_weights()
    print("CNN weight dim axis=0 : ", len(weights)) 
    # weights[0] shape = (ch_dim,ch_dim, input_ch_num,ch_num)
    # weights[1] shape = (ch_num,)
    weight = np.array(weights[0])
    print("CNN weights[0] shape:", weight.shape)
    ch_num = weight.shape[3]
    w_sum = np.zeros(ch_num)
    for i in range(ch_num):
        w_sum[i] = np.sum(np.absolute(weight[:,:,:,i]))
    print("sum shape:", w_sum.shape)
    idx_sort = np.argsort(w_sum, axis=-1)
    pidx = idx_sort[:psize]
    print("CNN pruning channel:", pidx)
    model = delete_channels(model, layer, pidx.tolist())
    return model

def plot_acc(layer='dense0'):
    import matplotlib.pyplot as plt
    plt.figure()
    retrain = {}
    if layer == 'dense0':
        p_acc = np.load(str(layer)+"_prune_acc.npy")
        p_acc = np.expand_dims(p_acc[:,1],axis=1)
        psize = [64, 128, 256, 512, 768,992]
        for idx, p in enumerate(psize):
            retrain[p] = np.load("retrain_"+str(p)+".npy")
            retrain[p] = np.concatenate((p_acc[idx],retrain[p]), axis=0)
        for key, hist in retrain.items():
            plt.plot(hist, label=str(key))
        plt.legend()
        plt.title("DNN prune criteria=zero_weights")
        plt.xlabel("retrain epoch")
        plt.ylabel("accuracy")
        plt.ylim((0,0.7))
        #plt.show()
        plt.savefig('DNN_weight_acc.png')
    elif layer == 'conv3':
        psize = [16,32,64]
        p_acc = np.load(str(layer)+"_prune_acc.npy")
        p_acc = np.expand_dims(p_acc[:,1], axis=1)
        for idx,p in enumerate(psize):
            retrain[p] = np.load("retrain_"+str(layer)+"_"+str(p)+".npy")
            retrain[p] = np.concatenate((p_acc[idx], retrain[p]), axis=0)
        for key, hist in retrain.items():
            plt.plot(hist, label=str(key))
        plt.legend()
        plt.title("CNN prune criteria=zero_weights")
        plt.xlabel("retrain epoch")
        plt.ylabel("accuracy")
        plt.ylim((0,0.7))
        #plt.show()
        plt.savefig('CNN_weight_acc.png')
