# _*_ coding:UTF-8
import caffe


def create_net(lmdb, mean_file, batch_size, include_acc=False):
    # 网络规范
    net = caffe.NetSpec();
    # 第一层Data层
    net.data, net.label = caffe.layers.Data(source = lmdb, backend = caffe.params.Data.LMDB, batch_size = batch_size, ntop = 2,
                                            transform_param = dict(scale = 0.00390625));
    # 第二层Convolution视觉层
    net.conv1 = caffe.layers.Convolution(net.data, num_output = 20, kernel_size = 5, pad = 1, stride = 1, weight_filler = dict(type = 'xavier'),
                                         bias_filler = dict(type = 'constant'));
    net.relu1 = caffe.layers.ReLU(net.conv1, in_place = True);
    net.pool1 = caffe.layers.Pooling(net.relu1, pool = caffe.params.Pooling.MAX, kernel_size = 3, stride = 2);
    net.conv1_1 = caffe.layers.Convolution(net.data, num_output = 20, kernel_size = 3, stride = 1, weight_filler = dict(type = 'xavier'),
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
    net.loss  = caffe.layers.SoftmaxWithLoss(net.ip4, net.label)
    # 训练的prototxt文件不包括Accuracy层， 测试的时候需要
    if include_acc:
        net.accuracy = caffe.layers.Accuracy(net.ip4, net.label);
        return str(net.to_proto());

    return str(net.to_proto());


def write_net():
    caffe_root = "/home/frank/Desktop/tmp/mnist-2/";
    train_lmdb = caffe_root + "mnist_train_lmdb";  # train.lmdb文件的位置
    test_lmdb  = caffe_root + "mnist_test_lmdb";
    mean_file  = caffe_root + "mean.binaryproto";  # 均值文件的位置
    train_proto = caffe_root + "train.prototxt";
    test_proto  = caffe_root + "test.prototxt";
    # 写入train.prototxt文件
    with open(train_proto, 'w') as f:
        f.write(create_net(train_lmdb, mean_file, batch_size = 64));

    # 写入test.prototxt文件
    with open(test_proto, 'w') as f:
        f.write(create_net(test_lmdb, mean_file, batch_size = 100, include_acc = True));

if __name__ == '__main__':
    write_net();