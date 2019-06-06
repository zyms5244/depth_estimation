import os
import os.path
class SDFCNConfig(object):
    """docstring for SDFCNConfig"""
    env = 'relative_l1_e2'
    model = 'ResDUCNet'
    half_width = False

    # data_root = '/media/ans/Share/nyudepthv2/'
    trainval_data_root = '/media/ans/Share/nyudepthv2/'
    # test_data_root = os.path.join(os.getcwd(), 'data', 'nyu_depth_v2_copy.mat')
    test_data_root = '/media/ans/Share/nyu_depth_v2_labeled.mat'
    resume_from_file = False
    resume_file = '/home/ans/PycharmProjects/Depth_in_The_Wild/resduc_c7.pth.tar'
    checkpoint = 'checkpoint.pth.tar'
    checkpoint_postfix = ''
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint', env)

    weights_file = os.path.join(os.getcwd(), 'models', 'NYU_ResNet-UpProj.npy')
    resnet50_file = os.path.join(os.getcwd(), 'models', 'resnet50-19c8e357.pth')
    # weights_file = 'checkpoint.pth.tar_1080ti_e-3_epoch_27_'

    batch_size = 8 # batch size
    use_gpu = True # use GPU or not
    num_workers = 8 # how many workers for loading data
    print_freq = 20 # print info every N batch
    test_batch_size = 1

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    print_freq = 20

    num_epochs = 20
    step = 8
    lr = 1e-4 # initial learning rate
    lr_decay = 0.1 # when val_loss increase, lr = lr*lr_decay


    weight_decay = 1e-4

    sample_num = None


