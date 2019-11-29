#! /bin/bash

cd ~/caffe
./build/tools/caffe.bin test \
-model ~/Desktop/tmp/mnist-2/test.prototxt \
-weights ~/Desktop/tmp/mnist-2/my_lenet_iter_100000.caffemodel \
-iterations 100
