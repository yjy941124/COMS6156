# COMS6156
This repository contains the preliminary experiment code of my mid-term papere: Comparative Analysis of Deep Learning Software Packages". It is divided into three subfolders, each contains the code for Caffe, TensorFlow, and Torch, respectively. In this Readme file, I will explain in detail how do I perform the experiment.

The experiment is performed over [ImageNet Dataset](http://image-net.org/download), using [VGG 19 Model](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Because the size of ImageNet Dataset, I won't be able to upload the entire dataset to the repo. Please see the above link for instructions on how to download the original image/url. 

### Caffe

Fortunately, the original VGG 19 model is initially trained over Caffe, so the author released the original training script. I used this script to train the ImageNet Dataset. This script is taken directly through [here](https://gist.github.com/graphific/16e3b1e3b13e555cb67a) with slight modification to the input source path. Note I don't have any additional code, because this script is directly invoked through shell script. Specifically,

``` 
$ caffe train -solver VGG_ILSVRC_19_layers_train_val.prototxt -output log.txt
```
The experiment is designed to calculate the processing time. This is automatically logged in log.txt. After one test, I will manually change the batch-size inside the .prototxt file and repeat the experiment.

### TensorFlow

I found a tensorflow implementation of VGG 19 [here](https://github.com/machrisaa/tensorflow-vgg). I'm using the training version of the implementation. For this one, I simply need to call test.py to perform the experiment. Similarly, the batchsize is manually changed in vgg19_trainable.py.

### Torch

There is an official implementation of VGG 19 from Facebook Research available [here](https://github.com/xhzhao/imagenet-CPU.torch). The opt.lua is the configuration and train.lua is the actual training code. In order to change the batch size, I manually change line 34 of opt.lua for each experiment. I also modified main.lua to monitor the execution time, and dataset.lua to point to my own data directory.


