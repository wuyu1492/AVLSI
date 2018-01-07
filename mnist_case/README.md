# Prune on MNIST database
Put mnist_model.py utils_mnist.py train.csv in the same path
## Usage
pretrain
```
python mnist_model.py
```
prune
```
python mnist_model.py prune
```
fine-tune
```
python mnist_model.py fine-tune
```
### How to use delete layer
Before delete a layer, it is necessary to prune the former layer so that the output shape of former layer is compitable with the input shape of the target layer.
It is a bug of keras-surgeon package.

For example, to prune "dense4" in dnn model:
```
    model.add(Dense(128,name="dense3"))
    model.add(Activation("relu"))
    model.add(Dense(64,name="dense4"))
    model.add(Activation("relu"))
    model.add(Dense(64,name="dense5"))
    model.add(Activation("relu"))
```
Perform pruning on dense3 first:
```
model = zero_weight(model, "dense3", 64)
model = delete_layer(model, model.get_layer(name='dense4'))
model = delete_layer(model, model.get_layer(name='activation_5')) #this is optional
```
