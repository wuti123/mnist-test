# _*_ coding:UTF-8
import caffe
import numpy as np
import cv2
import os

def test(my_project_root, deploy_proto):
    caffe_model = my_project_root + "my_lenet_iter_100000.caffemodel";
    img_dir = my_project_root + "image";


    label_filename = my_project_root + "labels.txt"

    net = caffe.Net(deploy_proto, caffe_model, caffe.TEST);    # 加载model和deploy
    #net = caffe.Classifier(deploy_proto, caffe_model);
    # 图片预处理设置
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,1,28,28)
    transformer.set_transpose('data', (2,0,1));     # 改变维度的顺序，由原始图片（28,28,1）变为(1, 28, 28)
    transformer.set_raw_scale('data', 255);         # 缩放到[0,255]之间
    #transformer.set_channel_swap('data', (2,1,0));  # 交换通道，将图片由RGB变为BGR

    for filename in os.listdir(img_dir):
        img_name = img_dir + "/" + filename;
        img = caffe.io.load_image(img_name, 0);         # 加载图片（灰度图 ）
        net.blobs['data'].data[...] = transformer.preprocess('data', img);  # 执行上面设置的图片预处理操作，并将图片载入到blob中

        out = net.forward();     # 执行测试
        #prediction = net.predict([img], oversample = False);
        #print prediction
        #print 'predicted class:', prediction[0].argmax()

        labels = np.loadtxt(label_filename, str, delimiter='\t');   # 读取类别名称文件
        prob = net.blobs['prob'].data[0].flatten();                 # 取出最后一层(Softmax)属于某个类别的概率
        order = prob.argsort()[-1];                                 # 将概率值排序，取出最大值所在的序号
        print '图片数字为：', img_name, labels[order];                # 将该序号转换成对应的类别名称，并打印
        #cv2.waitKey(0)


if __name__ == '__main__':
    my_project_root = "/home/frank/Desktop/tmp/mnist-2/";
    deploy_proto = my_project_root + "deploy.prototxt";
    test(my_project_root, deploy_proto);