# AVLSI

## Dependenceies:

install:

- pandas
- numpy
- keras
- kerassurgeon

## Usage

train.csv and pretrained model can be found in release

### Big Model
We trained this model on human facial expression classification data set, and we used this model to prove that prining is useful.

pre-train: (without prune)
```
python test-surgeon.py train.csv noprune
```
prune: 
```
python test-surgeon.py train.csv prune
```
fine-tune:(after prune)
```
python test-surgeon.py train.csv fine-tune
```
### MNIST model
We trained a CNN and a DNN on MNIST database, and used them in most of our experiments.

pre-train:
```
python mnist_model.py train
```
prune
```
python mnist_model.py prune
```
fine-tune
```
python mnist_model.py fine-tune
```
### Utilities
In file `Utils_mnist.py`, there are our pruning functions
```
mean_activation_rank(model, input_img, name="conv3", psize=0.5) #use mean of layer output as sign of activation

grad_layerwise_rank(model, input_img, psize=1) #use gradient for layer-wise ranking 

grad_activation_rank(model, input_img, name="conv3",psize=0.5) #use gradient as sign of activation
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
        grads = K.sum(K.abs(grads))
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
    """ channel wise random pruning 
        for conv layer
    """
    n_filter = model.get_layer(name=name).output_shape[-1]
    n_node = model.get_layer(name=name).output_shape[-1]
    psize = int(round(psize*n_node))
    pidx = np.arange(n_filter)
    np.random.shuffle(pidx)
    pidx = pidx[:psize]
    model = delete_channels(model, model.get_layer(name=name), pidx.tolist())
    return model

def random_conv_global(model, psize=0.9):
    """ channel wise global pruning """
    layer_start = {}
    pidx_dict = {}
    sidx_dict = {}
    node_num = 0
    for l in range(len(model.layers)):
        layer = model.layers[l]
        layer_class = layer.__class__.__name__
        if not (layer_class == 'Conv2D' or layer_class == 'Dense'):
            continue
        elif layer.name == 'conv0':
            continue
        elif layer.name == 'dense1':
            continue
        layer_start[layer.name] = node_num
        pidx_dict[layer.name] = []
        n_node = layer.outut_shape[-1]
        for i in range(n_node):
            sidx_dict[i+node_num] = layer.name
        node_num += n_node
    idx_rand = np.arange(node_num)
    idx_rand = np.random.shuffle(idx_rand)
    psize = int(round(psize*node_num))
    pidx = idx_rand[:psize]
    for p in pidx:
        layer_name = sidx_dict[p]
        pidx_dict[layer_name].append(p-layer_start[layer_name])
    for name, p_idx in pidx_dict.items():
        if len(p_idx) == 0:
            continue
        print("Pruning layer "+name+" channels: ", p_idx)
        model = delete_channels(model, model.get_layer(name=name), p_idx)
    return model

def mean_weight_mask(model, name="dense0", th=0.05):
    layer = model.get_layer(name=name)
    layer_idx = get_layer_index(model, name=name)
    if layer_idx == 0:
        print("Not on the input layer")
        return
    layer_class = layer.__class__.__name__
    if not (layer_class == 'Dense'):
        print(" Please assign a dense or conv layer")
        return model
    weights = layer.get_weights()
    weight = weights[0]
    zeros_n = np.count_nonzero(weight)
    print("Before pruning: nonzeros in weights:", zeros_n)
    it  = np.nditer(weight, flags=['multi_index'])
    while not it.finished:
        if np.abs(it[0])<th:
            weight[it.multi_index] = 0.0
        it.iternext()
    zeros_n = np.count_nonzero(weight)
    print("After Pruning: nonzeros in weights:", zeros_n)
    weights[0] = weight
    layer.set_weights(weights)
    mask_layer = Masking(mask_value=0.0)
    model = insert_layer(model, model.layers[layer_idx-1],mask_layer)
    return model

def check_zeros(model, name="dense0"):
    layer = model.get_layer(name=name)
    layer_class = layer.__class__.__name__
    if not layer_class == 'Dense':
        print(" Please assign a dense layer")
        return 0
    weights = layer.get_weights()
    weight = weights[0]
    zeros_n = np.count_nonzero(weight)
    print("check: nonzeros in weights:", zeros_n)
    
    

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
        elif len(p_idx) >= model.get_layer(name=name).output_shape[-1]:
            p_idx.pop()
        print("prune layer "+name+" for nodes: ", p_idx)
        model = delete_channels(model, model.get_layer(name=name), p_idx)
    return model


def zero_weight(model,layer_name, psize=0.5):
    print("Estimate weight in DNN, layer[{}]".format(layer_name))
    layer = model.get_layer(name=layer_name)
    n_node = model.get_layer(name=layer_name).output_shape[-1]
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
        w_sum /= np.sqrt(np.mean(np.square(w_sum), keepdims=True))
        if len(node_sums)==0:
            node_sums = w_sum.tolist()
        else:
            node_sums += w_sum.tolist()
    idx_sort = np.argsort(node_sums, axis=-1)
    print("idx_sort:" ,idx_sort)
    psize = int(round(psize*node_num))
    pidx = idx_sort[:psize]
    for p in pidx:
        name = sidx_dict[p]
        pidx_dict[name].append(p-layer_start[name])
    for name, p_idx in pidx_dict.items():
        print("pruning "+name+" prune channels:", p_idx)
        if len(p_idx) >= model.get_layer(name=name).output_shape[-1]:
            p_idx.pop()
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
```

## model
To compare the difference between CNN and DNN pruning, we performed experiment on both kinds of models, while using the same MNIST database. 

The CNN model we used for pruning experiment is composed of 4 Conv2D layers and 2 Dense layers. After we pretraining it for 10 epochs, it can achieve 99% accuracy on MNIST testset.

The DNN model we used for experiment is composed of 7 Dense layers. After 10 epochs of pretraining, it can achieve 96% accuracy pm MNIST testset.

