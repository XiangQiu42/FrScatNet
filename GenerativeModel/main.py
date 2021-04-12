# coding=utf-8

"""
Original Author: ZangJin

Revised version: 2021/03/16
Revised by: QiuXiang
"""
# Packages for Step 1
from utilss.extract_feature import Extract
from config.config import data_config

# Packages for step 2
from GSN import GSN
from FrGSN import FrGSN
from config.config import train_config
import os

"""
The first step: We need to unpickle the cifar-10 datasets to images
                and then extract features from images
"""

# setting the parameters
param = {

    'train_img_dir': data_config['train_data_dir'],
    'train_scat_dir': data_config['train_scat_dir'],
    'train_norm_dir': data_config['train_norm_dir'],
    'test_img_dir': data_config['test_data_dir'],
    'test_scat_dir': data_config['test_scat_dir'],
    'test_norm_dir': data_config['test_norm_dir'],

    #上面内容不需要修改，下面内容需要根据实际情况自己作修改
    'source_img_dir': '',
    'img_shape': (32,32),
    'cuda': True,
    'scat_J': 3,  #layers of the ScatNets
    'scat_batchsize': 256,

    'loss_data_dir': '/home/qiuxiang/expriments/Loss/'
}
# #
ext = Extract(param=param)
#
# # make the related dirs
# ext.mkdirs()
#
# # preprocess the datasets
# # ext.preProcess_celebA_datasets() # For celebA dataSets
# # ext.cifar10_unpickle() # For cifar-10 dataSets
#
# # # # Extracting features from the image
# # ext.scat_data()
#
# # # # we need to use PCA or FMFta method to compress the features to a implicit vector Z
# # ext.pca_data(param['train_scat_dir'], param['train_norm_dir'], param['test_scat_dir'], param['test_norm_dir'])
#
# alpha_1 = [0.1, 0.4, 0.7, 1.3, 1.6, 1.9, 1, 1, 1, 1, 1, 1]
# alpha_2 = [1, 1, 1, 1, 1, 1, 0.1, 0.4, 0.7, 1.3, 1.6, 1.9]

alpha_1 = [0.5, 1, 0.4, 1]
alpha_2 = [1, 0.5, 1, 1.6]
# ext.fr_scat_data(alpha_1=alpha_1, alpha_2=alpha_2)
# #
# ext.pca_data_fr(param['train_scat_dir'], param['train_norm_dir'], param['test_scat_dir'], param['test_norm_dir'],
#                 alpha_1=alpha_1, alpha_2=alpha_2)


"""
Finishing step 1
Ok, right here ,we have finished what a ScatNet/FrScatNet(Encoder)  want to do. 
The Next Question is: Given the implicit vector Z, can we use a Neural network to generate the image?
And that is what we need to do in the second step

Step 2: Use the Neural NetWork to generate image
"""

# Test if U can use cuda to speed ur networks
if train_config['cuda']:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(train_config['CUDA_VISIBLE_DEVICES'])
    print("You have config cuda device {} ...".format(train_config['CUDA_VISIBLE_DEVICES']))

# path1 = "/home/jains/Project-Zhang/generative-scattering-networks-master/experiments/gsn_hf/" \
#         "celebA_65536_2048_after_65536_ScatJ4_projected512_1norm_ncfl32_NormL1/models/"
# path2 = "/home/jains/Project-Zhang/generative-scattering-networks-master/experiments/gsn_hf/" \
#         "celebA_65536_2048_after_65536_ScatJ4_projected512_1norm_ncfl32_NormL1/models/"
# gsn = GSN(train_config)
# # gsn.compute_errors(1)
# # gsn.fusion_model(path1=path1,epoch1=1,path2=path2,epoch2=2)
# gsn.train(epoch_to_restore=0, epoch_train=350)

# gsn.save_originals()
# gsn.generate_from_model(348)
# gsn.compute_errors(340)
# gsn.analyze_model_epoch(348)
# gsn.compute_avg_errors(340,350)

# Ok, right here ,we have used our model to generate images.
# The Next Question is: Is out model preform well ??
# gsn.display_model_RESULT()

# dir1 = "/home/qiuxiang/experiments/ScatNets-cifar-10_32_NormL1_cifar10_tanh(-1,1)"
# dir2 = "/home/qiuxiang/experiments/FrScatNets-cifar-10_32_NormL1_cifar10_tanh(-1,1)"
# gsn.compare_scatNets(dir_scat=dir1, dir_fr_scat=dir2)

"""
to Compare different fractional parameter, use the following code 
"""
FrGSN = FrGSN(parameters=train_config,alpha_1=alpha_1, alpha_2=alpha_2)
# FrGSN.train(epoch_train=100)
FrGSN.compare_result(scat_dir='/home/qiuxiang/experiments/ScatNets-cifar-10_32_NormL1_cifar10_tanh(-1,1)/')

"""
Finishing Step 2

"""


# import pandas as pd
# import numpy as np
# df = pd.read_csv('/Users/jains/Desktop/data_record/cifar-10-1/cfar-10_fr_train_fr_test_frscat_5_J3_MPCA16_tanh(-1,1)_16_3channels_in(-1,1)MPCA16_loss.csv')
# train_psnr = np.array(df['train_psnr'])
# test_psnr = np.array(df['test_psnr'])
# train_ssim = np.array(df['train_ssim'])
# test_ssim = np.array(df['test_ssim'])
# max_psnr_idx = np.argmax(train_psnr)
# print(max_psnr_idx)
# print(df.iloc[max_psnr_idx:max_psnr_idx+1])
# print("%.5f"%df['train_psnr'].max(),"%.5f"%df['test_psnr'].max(),"%.5f"%df['train_ssim'].max(),"%.5f"%df['test_ssim'].max())
# print(np.argmax(test_psnr),np.max(test_psnr))
# print(np.argmax(train_ssim),np.max(train_ssim))
# print(np.argmax(test_ssim),np.max(test_ssim))
# df = df[:]
# print("%.5f"%df['train_psnr'].max(),"%.5f"%df['test_psnr'].max(),"%.5f"%df['train_ssim'].max(),"%.5f"%df['test_ssim'].max())

# def cal_increase():
#     df = pd.read_excel('/Users/jains/Desktop/data_record.xlsx',sheet_name=6)
#     Train_PSNR = np.array(df['Test SSIM'])
#     len = Train_PSNR.__len__()
#     Increased1 = []
#     print(Train_PSNR)
#     for idx in range(1,len):
#         # if(Train_PSNR[idx]<Train_PSNR[0]):
#         Increased1.append( round((Train_PSNR[idx]-Train_PSNR[0])/Train_PSNR[0]*100,1))
#         # else:
#         #     Increased1.append( round((Train_PSNR[idx]-Train_PSNR[0])/Train_PSNR[idx]*100,1))
#     print(np.array(Increased1))
#
#
#     return None
# # cal_increase()
#
#
#
# def fusion_image():
#     from skimage.measure import compare_ssim, compare_nrmse, compare_psnr, compare_mse
#     from PIL import Image
#
#     img_root = '/Users/jains/Desktop/celebA-recon/'
#     org_test = np.array(Image.open(img_root + '5/originals_test.png'))
#     org_train = np.array(Image.open(img_root + '5/originals_train.png'))
#
#     gen_test = np.array(Image.open( img_root+'3/epoch_350_test.png'))
#     gen_train = np.array(Image.open( img_root+'3/epoch_350_train.png'))
#     gen_test1 = np.array(Image.open(img_root + '5/epoch_345_test.png'))
#     gen_train1 = np.array(Image.open(img_root + '5/epoch_345_train.png'))
#
#     Image.fromarray(np.uint8(0.5*gen_test+0.5*gen_test1)).save(img_root+'test.png')
#     Image.fromarray(np.uint8(0.5*gen_train+0.5*gen_train1)).save(img_root+'train.png')
#     print('%.5f' % compare_psnr(org_train, np.uint8(0.5 * gen_train + 0.5 * gen_train1)),
#           '%.5f' % compare_psnr(org_test, np.uint8(0.5 * gen_test + 0.5 * gen_test1)),
#           '%.5f' % compare_ssim(org_train, np.uint8(0.5 * gen_train + 0.5 * gen_train1), multichannel=True),
#           '%.5f' % compare_ssim(org_test, np.uint8(0.5 * gen_test + 0.5 * gen_test1), multichannel=True))
# # fusion_image()
#
#
# def fusion_image_save():
#     from PIL import Image
#     test1 = np.array((Image.open('/Users/jains/Desktop/未命名文件夹/epoch_336_test.png')))
#     test2 = np.array((Image.open('/Users/jains/Desktop/未命名文件夹/epoch_343_test.png')))
#     train1 = np.array((Image.open('/Users/jains/Desktop/未命名文件夹/epoch_336_train.png')))
#     train2 = np.array((Image.open('/Users/jains/Desktop/未命名文件夹/epoch_343_train.png')))
#
#     Image.fromarray(np.uint8(0.5*test1+0.5*test2)).save('/Users/jains/Desktop/未命名文件夹/fusion_test.jpg')
#     Image.fromarray(np.uint8(0.5 * train1 + 0.5 * train2)).save('/Users/jains/Desktop/未命名文件夹/fusion_train.jpg')
# fusion_image_save()




# import numpy as np
# list1 = [1,2,3,4,5,6,7,8,9,10,11,12]
# arr = np.array(list1)
# out = np.empty(4,int)
# for idx in range(int(len(arr)/3)):
#     tmp = arr[idx*3:(idx+1)*3]
#     out[idx] = tmp.sum()
# print(out)