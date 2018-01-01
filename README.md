# AVLSI

Dependenceies:
install: 
pandas
numpy
keras
kerassurgeon

train.csv can be found in release

train: (with prune)
```
python test-surgeon.py train.csv prune
```
fine-tune:(after prune)
```
python test-surgeon.py train.csv fine-tune
```
train: (without prune)
```
python test-surgeon.py train.csv noprune
```
