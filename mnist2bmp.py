# 将二进制图片数据转换成bmp格式，用于测试网络准确度

import numpy as np
import struct
import matplotlib.pyplot as plt
# import Image
from PIL import Image, ImageFont

filename = 't10k-images-idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()

index = 0
s = struct.Struct('>IIII');
# magic,numImages,numRows,numColumns=strcut.unpack_from('>IIII',buf,index)
magic, numImages, numRows, numColumns = s.unpack_from(buf, index)
index += struct.calcsize('>IIII')

for image in range(0, numImages):
    im = struct.unpack_from('>784B', buf, index)  # 28*28=784
    index += struct.calcsize('>784B')

    im = np.array(im, dtype='uint8')
    im = im.reshape(28, 28)

    im = Image.fromarray(im)
    im.save('/home/frank/Desktop/tmp/mnist-2/2/test_%s.bmp' % image, 'bmp')
