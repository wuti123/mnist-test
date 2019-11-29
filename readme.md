# python2

1. 运行create_net.py
python create_net.py
生成train.prototxt test.prototxt两个网络模型

2. 运行create_sovler.py
python create_sovler.py
生成my_solver.prototxt（设置了超参数），用于生成求解器 

3. 运行脚本train.sh
./train.sh
训练网络，迭代次数100000次，生成两个权值文件my_lenet_iter_50000.caffemodel,my_lenet_iter_100000.caffemodel,和两个状态文件。

4. 运行脚本test.sh
./test.sh
用mnist_test_lmdb测试数据集测试网络mAP

5. 运行create_deploy.py
python create_deploy.py
生成deploy.prototxt, 与train.prototxt test.prototxt模型相比，没有了loss\accuracy层，且输入变成了由一张图片输入。

6. 运行mnist2bmp.py
python mnist2bmp.py
将t10k-images-idx3-ubyte二进制测试图片转换成bmp图片格式(经测试，用png图片测试会出错，检测不准，不知道为什么)

7. image文件夹里的图片是从转换出的bmp图片中拷贝的，可以用自己制作的图片，但要保证格式是bmp
运行test_own_picture.py
python test_own_picture.py
实现用训练好的网络，一张一张的对image文件夹里的图片进行检测，(需要对图片进行预处理：改变维度，归一化等)

test_new.py是仅仅对一张图片的检测，没有遍历文件夹

8. 对于test_own_picture.py：
net = caffe.Net(deploy_proto, caffe_model, caffe.TEST);    # 加载model和deploy
out = net.forward();     # 执行测试

   对于test_new.py:
net = caffe.Classifier(deploy_proto, caffe_model);   # 加载model和deploy
prediction = net.predict([img], oversample = False); # 执行测试
