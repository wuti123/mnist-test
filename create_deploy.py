# _*_ coding: UTF-8
# 生成deploy.prototxt, 用于用自己的RGB图片进行预测, 不要训练/测试模型的数据层、分类准确层、损失层，新增prob层

import caffe
#import cv2

def create_deploy():
    # 网络规范
    net = caffe.NetSpec();
    # 第一层Data层
    #net.data, net.label = caffe.layers.Data(source = lmdb, backend = caffe.params.Data.LMDB, batch_size = batch_size, ntop = 2,
     #                                       transform_param = dict(scale = 0.00390625));
    # 第二层Convolution视觉层
    net.conv1 = caffe.layers.Convolution(bottom = 'data', num_output = 20, kernel_size = 5, pad = 1, stride = 1, weight_filler = dict(type = 'xavier'),
                                         bias_filler = dict(type = 'constant'));
    net.relu1 = caffe.layers.ReLU(net.conv1, in_place = True);
    net.pool1 = caffe.layers.Pooling(net.relu1, pool = caffe.params.Pooling.MAX, kernel_size = 3, stride = 2);
    net.conv1_1 = caffe.layers.Convolution(bottom = 'data', num_output = 20, kernel_size = 3, stride = 1, weight_filler = dict(type = 'xavier'),
                                         bias_filler = dict(type = 'constant'));
    net.relu1_1 = caffe.layers.ReLU(net.conv1_1, in_place = True);
    net.pool1_1 = caffe.layers.Pooling(net.relu1_1, pool = caffe.params.Pooling.MAX, kernel_size = 3, stride = 2);
    net.eltwise = caffe.layers.Eltwise(net.pool1, net.pool1_1);
    net.relu  = caffe.layers.ReLU(net.eltwise, in_place = True);
    #net.conv2 = caffe.layers.Convolution(net.pool1, num_output = 32, kernel_size = 3, stride = 1, pad = 1, weight_filler = dict(type = 'xavier'),
     #                                    bias_filler = dict(type = 'constant'));
    net.conv2 = caffe.layers.Convolution(net.eltwise, num_output = 32, kernel_size = 3, stride = 1, pad = 1, weight_filler = dict(type = 'xavier'),
                                         bias_filler = dict(type = 'constant'));
    net.relu2 = caffe.layers.ReLU(net.conv2, in_place = True);
    net.pool2 = caffe.layers.Pooling(net.relu2, pool = caffe.params.Pooling.MAX, kernel_size = 3, stride = 2);
    # 全连接层
    net.ip3   = caffe.layers.InnerProduct(net.pool2, num_output = 1024, weight_filler = dict(type = 'xavier'),
                                          bias_filler = dict(type = 'constant'));
    net.relu3 = caffe.layers.ReLU(net.ip3, in_place = True);
    # dropout层
    net.drop3 = caffe.layers.Dropout(net.relu3, in_place = True);
    net.ip4   = caffe.layers.InnerProduct(net.drop3, num_output = 10, weight_filler = dict(type = 'xavier'),
                                          bias_filler = dict(type = 'constant'));
    # softmax层
    #net.loss  = caffe.layers.SoftmaxWithLoss(net.ip4, net.label)
    # 训练的prototxt文件不包括Accuracy层， 测试的时候需要
    #net.accuracy = caffe.layers.Accuracy(net.ip4, net.label);

    net.prob = caffe.layers.Softmax(net.ip4);

    return net.to_proto();

def write_net(deploy_proto):
    with open(deploy_proto, 'w') as f:
        # 写入第一层数据描述
        f.write('input: "data"\n');
        f.write('input_dim: 1\n');
        f.write('input_dim: 1\n');
        f.write('input_dim: 28\n');
        f.write('input_dim: 28\n');
        f.write(str(create_deploy()));

if __name__ == '__main__':
    my_project_root = "/home/frank/Desktop/tmp/mnist-2/"
    deploy_proto = my_project_root + "deploy.prototxt"
    write_net(deploy_proto);