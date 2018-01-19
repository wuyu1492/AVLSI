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
    
random_conv_channel(model, name="conv3", psize=0.5)#channel wise random pruning 

random_conv_global(model, psize=0.9)#channel wise global pruning 

zero_weight_all(model, psize=0.5)# minimum weight ranking across all dense layers 

zero_weight(model,layer_name, psize=0.5) #minimum weight channel wise pruning in one dense layer

zero_channels_all(model, psize=0.5) minimum weight ranking across all conv layers 

zero_channels(model, layer_name, psize=0.5) # minimum weight ranking channels in one conv layer
```

## model
To compare the difference between CNN and DNN pruning, we performed experiment on both kinds of models, while using the same MNIST database. 

The CNN model we used for pruning experiment is composed of 4 Conv2D layers and 2 Dense layers. After we pretraining it for 10 epochs, it can achieve 99% accuracy on MNIST testset.

The DNN model we used for experiment is composed of 7 Dense layers. After 10 epochs of pretraining, it can achieve 96% accuracy pm MNIST testset.

