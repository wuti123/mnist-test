# -*- coding:utf-8 -*-
# 有中文注释的话需要utf-8编码
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe

caffe_root = '/home/yang/caffe/'  # 设置Caffe环境的根目录
#sys.path.insert(0, caffe_root + 'python')  # 添加系统环境变量

my_project_root = "/home/frank/Desktop/tmp/mnist-2/";
MODEL_FILE = my_project_root + "deploy.prototxt";  # Lenet网络的定义文件
PRETRAINED = my_project_root + "my_lenet_iter_100000.caffemodel"; # 网络模型参数
IMAGE_FILE = my_project_root + "image" + "/test_2.bmp"  # 测试图片路径

input_image = caffe.io.load_image(IMAGE_FILE, color=False)

net = caffe.Classifier(MODEL_FILE, PRETRAINED)

prediction = net.predict([input_image], oversample=False)
print prediction
caffe.set_mode_cpu()
print 'predicted class:', prediction[0].argmax()
