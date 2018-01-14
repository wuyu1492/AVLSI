# AVLSI

## Dependenceies:

install:

pandas

numpy

keras

kerassurgeon

## How to use

train.csv and pretrained model can be found in release

train: (without prune)
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

## model
The CNN model we used for pruning experiment is composed of 4 Conv2D layers and 2 Dense layers. After we pretraining it for 10 epochs, it can achieve 99% accuracy on MNIST testset.

The DNN model we used for experiment is composed of 7 Dense layers. After 10 epochs of pretraining, it can achieve 96% accuracy pm MNIST testset.

## prune

