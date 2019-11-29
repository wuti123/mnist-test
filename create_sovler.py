# _*_coding: UTF-8
import caffe

def write_sovler():
    my_project_root = "/home/frank/Desktop/tmp/mnist-2/";
    sovler_string = caffe.proto.caffe_pb2.SolverParameter();    # sovler存储
    solver_file   = my_project_root + "my_solver.prototxt"            # sovler文件保存位置
    sovler_string.train_net = my_project_root + "train.prototxt";     # train.prototxt位置指定
    sovler_string.test_net.append(my_project_root + "test.prototxt"); # test.prototxt位置指定
    sovler_string.test_iter.append(100);
    sovler_string.test_interval = 500;
    sovler_string.base_lr = 0.001;
    sovler_string.momentum = 0.9;
    sovler_string.weight_decay = 0.004;
    sovler_string.lr_policy = "fixed"
    sovler_string.display = 100;
    sovler_string.max_iter = 100000;
    sovler_string.snapshot = 50000;
    sovler_string.snapshot_format = 1;        # 临时模型的保存格式，0代表HDF5, 1代表BIN
    sovler_string.snapshot_prefix = '/home/frank/Desktop/tmp/mnist-2/my_lenet'  # 模型前缀
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU  # 优化模式

    with open(solver_file, 'w') as f:
        f.write(str(sovler_string));

if __name__ == '__main__':
    write_sovler();