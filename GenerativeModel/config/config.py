import os

HOME = os.environ['HOME']

net_config = {
    'net_name': 'cifar10',  # 'dcgan','celebA','cifar10',
    'last_activate': 'tanh',
    'train_x': '(-1,1)',
}
train_config = {
    # 常改参数
    'dataset_root': HOME + '/datasets/cifar-10-batches-py/',  # 训练集根目录

    # 'ScatNets-cifar-10' means we use scat net to get the train date, 'FrscatNets-cifar-10' means to use fractional
    # scatNets, note here we just use the filename to distinguish the data from different feature extract method nad
    # different data_sets
    'dataset': 'FrScatNets-cifar-10',

    'model_name': '_' + net_config['net_name'] + '_' + net_config['last_activate'] + net_config['train_x'],
    # 备注存储模型的文件夹
    'cuda': True,  # 无GPU时需要更改
    'CUDA_VISIBLE_DEVICES': 0,  # default
    'batch_size': 128,
    'num_workers': 4,  # win下设置为0

    'nb_channels_first_layer': 32,
    # 'train_attribute': '65536',
    # 'test_attribute': '2048_after_65536',
    'dim': 512,
    # 'embedding_attribute': 'ScatJ4_projected{}_1norm'.format(512),
    'exper_path': '/home/qiuxiang/experiments',
    # give the number of images that U want to save in function "generate_from_model" after train
    'image_generate': 9,
    'train-images_filename': 'train-images',
    'test-images_filename': 'test-images',
    'train-norm_filename': 'train_norm',
    'test-norm_filename': 'test_norm',
}

data_config = {
    'npy': True,
    'pca_method': 2,  # 1表示三个通道拉长后PCA压缩，2表示三通道累加后在做PCA压缩，3表示单通道做PCA压缩后叠加在一起送入3*512维
    'pca_outdim': 512,  # PCA输出维度

    # note it may be wrong here to use pca on a batch instead of the whole datasets
    # 'pca_fixed':8192,#PCA的batch，，，8192 or 1024

    'train_data_dir': train_config['dataset_root'] + '/train-images',
    'train_scat_dir': train_config['dataset_root'] + train_config['dataset'] + '/train_scat',
    'train_norm_dir': train_config['dataset_root'] + train_config['dataset'] + '/train_norm',
    'test_data_dir': train_config['dataset_root'] + '/test-images',
    'test_scat_dir': train_config['dataset_root'] + train_config['dataset'] + '/test_scat',
    'test_norm_dir': train_config['dataset_root'] + train_config['dataset'] + '/test_norm',

    'scat_J': 3,  # 散射网络J 3 for cifar-10, 4 for celeba
    'scat_shape': (32, 32),  # 散射时输入图像大小,for cifar-10 , it is (32,32), (128,128)for celebA
}

dcgan_config = {

}


def print_shape():
    import numpy as np
    OutPath = '/home/jains/datasets/huang_zhong_ren/huangzhongren32/2048_after_65536_ScatJ2_projected512_1norm_s2ks33ch16*4/'
    listn = os.listdir(OutPath)
    filepath = OutPath + listn[0]
    print(np.load(filepath).shape)
    return
# print_shape()

# print(df['train_psnr'][300:].max(),df['test_psnr'][300:].max(),df['train_ssim'][300:].max(),df['test_ssim'][300:].max())
