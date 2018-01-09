import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.engine.topology import Layer
from kerassurgeon.operations import delete_channels


def get_layer_index(model, name='dense0'):
    idx = -1
    count = 0
    for layer in model.layers:
        if layer.get_config()['name'] == name:
            idx = count
            break
        else:
            count += 1
    return idx

def mean_activation_rank(model, input_img, name="conv3", psize=0.5):
    """
    use mean of layer output as sign of activation
    for pruning conv layer
    psize: floating point 0 to 1
    """
    layer = model.get_layer(name=name)
    n_node = layer.output_shape[-1]
    psize = int(round(psize*n_node))
    out_shape = layer.output_shape
    if len(out_shape)>2:
        mean_axis = (0,1,2)
    else:
        mean_axis = 0
    inputs = model.input
    mean = K.mean(model.get_layer(name=name).output, axis=mean_axis)
    iterate = K.function([inputs], [mean])
    mean_values = iterate([input_img])
    print("mean values:", mean_values)
    idx_sort = np.argsort(mean_values)[0]
    pidx = idx_sort[:psize]
    print(pidx)
    model = delete_channels(model, model.get_layer(name=name),pidx.tolist())
    return model

def grad_activation_rank(model, input_img, name="conv3",psize=0.5):
    """ use gradient as sign of activation """
    layer = model.get_layer(name=name)
    n_node = layer.output_shape[-1]
    psize = int(round(psize*n_node))
    inputs = model.input
    out_shape = layer.output_shape
    if len(out_shape)>2:
        mean_axis = (0,1,2)
    else:
        mean_axis = 0
    mean = K.mean(layer.output, axis=mean_axis)
    print("mean shape:", mean.shape)
    std_grads = []
    for i in range(n_node):
        print("grads[{}]".format(i))
        grads = K.gradients(mean[i], inputs)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))))
        grads = K.sum(grads)
        iterate = K.function([inputs],[grads])
        grads = iterate([input_img])
        std_grads.append(grads[0])
    print("std_grads:", std_grads)
    pidx = np.argsort(std_grads)
    print(pidx)
    pidx = pidx[:psize]
    model = delete_channels(model, layer, pidx.tolist())
    return model

def random_conv_channel(model, name="conv3", psize=0.5):
    n_filter = model.get_layer(name=name).output_shape[-1]
    n_node = model.get_layer(name=name).output_shape[-1]
    psize = int(round(psize*n_node))
    pidx = np.arange(n_filter)
    np.random.shuffle(pidx)
    pidx = pidx[:psize]
    model = delete_channels(model, model.get_layer(name=name), pidx.tolist())
    return model

def zero_weight_all(model, psize=0.5):
    """ ranking across all dense layers """
    node_sums = []
    layer_start = {}
    pidx_dict = {}
    sidx_dict = {}
    node_num = 0
    for l in range(len(model.layers)):
        layer = model.layers[l]
        layer_class = layer.__class__.__name__
        if not layer_class == 'Dense':
            continue
        elif layer.name == 'dense_o':
            continue
        layer_start[layer.name] = node_num
        pidx_dict[layer.name] = []
        n_node = layer.output_shape[-1]
        for i in range(n_node):
            sidx_dict[i+node_num] = layer.name
        node_num += n_node
        weights = layer.get_weights()
        weight = weights[0]
        node_weight = np.sum(np.absolute(weight), axis=0)
        node_weight /= np.sqrt(np.mean(np.square(node_weight), keepdims=True))
        if len(node_sums)==0:
            node_sums = node_weight.tolist()
        else:
            node_sums += node_weight.tolist()
    idx_sort = np.argsort(node_sums, axis=-1)
    psize = int(round(psize*node_num))
    pidx = idx_sort[:psize]
    for p in pidx:
        layer_name = sidx_dict[p]
        pidx_dict[layer_name].append(p-layer_start[layer_name])
    for name, p_idx in pidx_dict.items():
        if len(p_idx) == 0:
            continue
        print("prune layer "+name+" for nodes: ", p_idx)
        model = delete_channels(model, model.get_layer(name=name), p_idx)
    return model


def zero_weight(model,layer_name, psize=0.5):
    print("Estimate weight in DNN, layer[{}]".format(layer_name))
    layer = model.get_layer(name=layer_name)
    n_node = model.get_layer(name=name).output_shape[-1]
    psize = int(round(psize*n_node))
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

def zero_channels_all(model, psize=0.5):
    """ ranking across all conv layers """
    node_sums = []
    layer_start = {}
    pidx_dict = {}
    sidx_dict = {}
    node_num = 0
    for l in range(len(model.layers)):
        layer = model.layers[l]
        layer_class = layer.__class__.__name__
        if not layer_class == 'Conv2D':
            continue
        elif layer.name == 'conv0':
            continue
        layer_start[layer.name] = node_num
        n_node = layer.output_shape[-1]
        pidx_dict[layer.name] = []
        for i in range(n_node):
            sidx_dict[i+node_num] = layer.name
        node_num += n_node
        weights = layer.get_weights()
        weight = np.array(weights[0])
        ch_num = weight.shape[3]
        w_sum = np.zeros(ch_num)
        for i in range(ch_num):
            w_sum[i] = np.sum(np.absolute(weight[:,:,:,i]))
        w_sum /= np.sqrt(np.mean(np.square(w_sum[i]), keepdims=True))
        if len(node_sums)==0:
            node_sums = w_sum.tolist()
        else:
            node_sums += w_sum.tolist()
    idx_sort = np.argsort(node_sums, axis=-1)
    psize = int(round(psize*node_num))
    pidx = idx_sort[:psize]
    for p in pidx:
        name = sidx_dict[p]
        pidx_dict[name].append(p-layer_start[name])
    for name, p_idx in pidx_dict.items():
        print("pruning "+name+" prune channels:", p_idx)
        model = delete_channels(model, model.get_layer(name=name), p_idx)
    return model


def zero_channels(model, layer_name, psize=0.5):
    print("Estimate weight in CNN, layer[{}]".format(layer_name))
    layer = model.get_layer(name=layer_name)
    n_node = model.get_layer(name=name).output_shape[-1]
    psize = int(round(psize*n_node))
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
