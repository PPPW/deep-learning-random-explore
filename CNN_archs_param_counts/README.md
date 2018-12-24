# Parameter Count for Popular CNN Architectures

architecture | # parameters
---|---
resnet34 | 21.8 M
resnet50 | 25.6 M
resnet101 | 44.6 M
resnext\_50_32x4d | 25.1 M
resnext\_101_32x4d | 44.2 M
se\_resnext50_32x4d | 27.6 M
senet154 | 115.1 M
wrn\_50_2f | 68.9 M
densenet121 | 8 M
densenet201 | 20 M
InceptionResnetV2 | 55.9 M
inceptionv4 | 42.7 M

The above table lists the total number of parameters of the most popular CNN architectures. The model names are the built-in architectures provided by [fast.ai v0.7](https://github.com/fastai/fastai/tree/master/old). 

It's easy to count the parameter numbers using PyTorch directly, but as I was using fast.ai v0.7 for a classification problem with different architectures and already got these numbers, I just put here as a reference. Although the last few layers of the original models are cut out to adapt the problem, the total number of parameters are not affected much. 

Note that although we can take this as a reference for how large the architecture is, the memory usage does not only depend on the number of parameters, but also the network structures and other settings.  

Check [this notebook](cnn_arch_n_params.ipynb) for how these numbers are calculated. 