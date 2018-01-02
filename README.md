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

