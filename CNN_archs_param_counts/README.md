# Parameter Count for Popular CNN Architectures

architecture | # parameters
---|---
resnet34 | 21.8 M
resnet50 | 25.6 M
resnet101 | 44.6 M
resnext50 | 25.1 M
resnext101 | 44.2 M
se_resnext50 | 27.6 M
senet154 | 115.1 M
wrn | 68.9 M
dn121 | 8 M
dn201 | 20 M
inceptionresnet_2 | 55.9 M
inception_4 | 42.7 M

The above table lists the total number of parameters of the most popular CNN architectures. The model names are the built-in architectures provided by [fast.ai v0.7](https://github.com/fastai/fastai/tree/master/old). 

I got these numbers when I was trying different architectures for a classification problem, so I took it down just for reference. Although he last few layers of the original models are cut out to adapt the problem, the total number of parameters are not affected much. 

Note that although we can take this as a reference for how large the architecture is, the memory usage does not only depend on the number of parameters, but also the network structures. 

Check [this notebook](cnn_arch_n_params.ipynb) for how these numbers are calculated. 