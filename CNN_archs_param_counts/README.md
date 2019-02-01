# Parameter Count for Popular CNN Architectures

architecture | # parameters
---|---
resnet18 | 11.7 M
resnet34 | 21.8 M
resnet50 | 25.6 M
resnet101 | 44.6 M
resnet152 | 60.3 M
resnext50_32x4d | 25.1 M
resnext101_32x4d | 44.2 M
resnext101\_64x4d | 83.5 M
se\_resnet50 | 28.1 M
se\_resnet101 | 49.4 M
se\_resnet152 | 66.9 M
se\_resnext50_32x4d | 27.6 M
se\_resnext101_32x4d | 49.0 M
senet154 | 115.1 M
wrn\_50_2f | 68.9 M
densenet121 | 8.0 M
densenet161 | 28.7 M
densenet169 | 14.2 M
densenet201 | 20.0 M
inceptionresnetv2 | 55.9 M
inceptionv4 | 42.7 M
xception | 22.9 M
squeezenet1_0 | 1.26 M
squeezenet1_1 | 1.25 M
vgg16_bn | 15.3 M
vgg19_bn | 20.6 M
alexnet | 2.74 M

The above table lists the total number of parameters of the most popular CNN architectures. The model names are the built-in architectures provided by [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) and [Cadene](https://github.com/Cadene/pretrained-models.pytorch). 

It's easy to count the parameter numbers using PyTorch directly, but as I was using fast.ai v0.7 for a classification problem with different architectures and already got these numbers, I just put here as a reference. Although the last few layers of the original models are cut out to adapt the problem, the total number of parameters are not affected much. 

Note that although we can take this as a reference for how large the architecture is, the memory usage does not only depend on the number of parameters, but also the network structures and other settings.  

Check [this notebook](cnn_arch_n_params.ipynb) for how these numbers are calculated. 