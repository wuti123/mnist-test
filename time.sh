#! /bin/bash

# cpu版
#cd ~/caffe
#./build/tools/caffe.bin time -model /home/frank/Desktop/tmp/mnist-2/train.prototxt

# gpu版
cd ~/caffe
./build/tools/caffe.bin time -model /home/frank/Desktop/tmp/mnist-2/train.prototxt -gpu 0
