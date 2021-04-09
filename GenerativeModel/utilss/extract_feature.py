# -*- coding=utf-8 -*-

"""
Author: zhangjing
Date and time: 2/02/19 - 17:58

Revised version : QiuXiang
Date: 2021/03/16
"""

import random
import shutil
import torch
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import scipy.io as scio
import matplotlib.pyplot as plt

import face_recognition
import gc

import sys
sys.path.append('..')

from kymatio.torch import Fr_Scattering2D,Scattering2D
from config.config import train_config, data_config

# from utilss.tools import get_nb_files

class Extract:
    def __init__(self, param):
        self.train_img_dir = param['train_img_dir']
        self.train_scat_dir = param['train_scat_dir']
        # self.train_pro_dir = param['train_pro_dir']  um... Is it useful???
        self.test_img_dir = param['test_img_dir']
        self.test_scat_dir = param['test_scat_dir']
        # self.test_pro_dir = param['test_pro_dir']

        # scat param
        self.source_img_dir = param['source_img_dir']
        self.img_shape = param['img_shape']
        self.cuda = param['cuda']
        self.scat_J = param['scat_J']
        self.scat_batchsize = param['scat_batchsize']

        # loss 综合分析参数
        self.loss_data_dir = param['loss_data_dir']
        print("\nlet's start extracting features...\n")

    def normalize(self, vector):
        norm = np.sqrt(np.sum(vector ** 2))
        return vector / norm

    def resize_face(self):
        source_files = './datasets/diracs/1024/'
        filename_list = os.listdir(source_files)

        for i in range(len(filename_list)):
            filename = filename_list[i]
            imgName = os.path.join(source_files, os.path.basename(filename_list[i]))
            if (os.path.splitext(imgName)[1] != '.jpg'): continue
            img = Image.open(imgName).resize((128, 128))

            img.save('./datasets/celebA/256/%s' % filename)
        return

    # Note here: since there may be images that the face can't be recognized, hence we need to use 'num' to ensure the
    # that we have selected enough images
    def cut_celeba_face(self, image_dir, destination_dir, num_needed):

        file_list = os.listdir(image_dir)
        HEIGHT, WIDTH = self.img_shape
        print("***************Start resizing {2} images to {0}x{1}***************".format(HEIGHT, WIDTH, num_needed))

        for i in range(0, len(file_list)):

            COUNT_NUM = len(os.listdir(destination_dir))
            if COUNT_NUM >= num_needed:
                break
            imgName = os.path.join(image_dir, os.path.basename(file_list[i]))
            fileName = os.path.basename(file_list[i])

            if (os.path.splitext(imgName)[1] != '.jpg'):
                print("Find Images not as jpg!")
                continue
            image = face_recognition.load_image_file(imgName)  # recognition the face
            face_locations = face_recognition.face_locations(image)  # return face locations in the image
            for face_location in face_locations:

                top, right, bottom, left = face_location  # detail of locations

                x = (top + bottom) / 2  # the center of faces
                y = (right + left) / 2

                # compute details of face locations
                top = int(x - HEIGHT / 2)
                bottom = int(x + HEIGHT / 2)
                left = int(y - WIDTH / 2)
                right = int(y + WIDTH / 2)

                # let faces' width equal higth
                if (top < 0) or (bottom > image.shape[0]) or (left < 0) or (right > image.shape[1]):
                    top, right, bottom, left = face_location
                    width = right - left
                    height = bottom - top
                    if (width > height):
                        right -= (width - height)
                    elif (height > width):
                        bottom -= (height - width)

                # cut face from original images
                face_image = image[top:bottom, left:right]

                # translate into PIL data
                pil_image = Image.fromarray(face_image)
                pil_image = pil_image.resize((128, 128))  # resize to the fixed dim images
                pil_image.save(destination_dir + '/' + fileName)  # save the faces

                if (COUNT_NUM) % 1000 == 0:
                    print("In processing images: {}".format(COUNT_NUM))

        print("***************Finish clipping {1} images in {0}***************".format(image_dir, COUNT_NUM))

    def choose_img(self, src_dir, out_dir, num=1):
        filename_list = os.listdir(src_dir)  # read the directory files's name

        for i in range(70001, 86385):
            imgName = os.path.join(src_dir, os.path.basename(filename_list[i]))
            if (os.path.splitext(imgName)[1] != '.jpg'): continue
            img = Image.open(imgName)

            img.save(out_dir + filename_list[i])
            print("step is %d k", (i - 70000) / 1000)
        return 0

    # this function to choose images from cel datasets and then divided
    # it into train set and test set, further more, we need to resize it to (128x128)
    def preProcess_celebA_datasets(self):

        images_train_selected_dir = '/home/qiuxiang/datasets/celebA/train_origianl'
        images_test_selected_dir = '/home/qiuxiang/datasets/celebA/test_origianl'
        if not os.path.exists(images_train_selected_dir):
            os.makedirs(images_train_selected_dir)
        if not os.path.exists(images_test_selected_dir):
            os.makedirs(images_test_selected_dir)

        print("~~~~~~~~~~~~~~~~~~~~~Start making train-set and test-set~~~~~~~~~~~~~~~~~~~~~~~~")
        datasets_sources = '/home/qiuxiang/datasets/celebA/img_align_celeba/'
        filename_list = os.listdir(datasets_sources)  # read the directory files's name
        filename_list.sort()
        print("There are {} images in  the original datasets".format(len(filename_list)))

        # so at here, we want to choose 65536( = 2^16) and 16384 images as train sets and test sets
        num_train = 65536
        num_test = 16384

        # random select some images from datasets and move it to the destination,for the case that
        # can't be recognized ,we need to select  more images
        sample = random.sample(filename_list, num_train + 10000)

        for name in sample:
            shutil.move(datasets_sources + name, images_train_selected_dir + '/' + name)

        filename_list = os.listdir(datasets_sources)  # read the directory files's name
        filename_list.sort()
        print("Finishing make train set,now there are {} images in  the original datasets,"
              "\nlet's start make test set".format(len(filename_list)))
        sample2 = random.sample(filename_list, num_test + 5000)
        for name in sample2:
            shutil.move(datasets_sources + name, images_test_selected_dir + '/' + name)
        print("~~~~~~~~~~~~~~~~~~~~~Finish making train-set and test-set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.cut_celeba_face(image_dir=images_train_selected_dir, destination_dir=self.train_img_dir,
                             num_needed=num_train)
        self.cut_celeba_face(image_dir=images_test_selected_dir, destination_dir=self.test_img_dir, num_needed=num_test)

    # We neet to use PCA on the whole datasets(include train-set and test-set),
    def pca_data(self, data_dir_train, out_dir_train, data_dir_test, out_dir_test):

        if not os.path.exists(out_dir_train):
            os.makedirs(out_dir_train)
        if not os.path.exists(out_dir_test):
            os.makedirs(out_dir_test)

        # This function is to read data from "file_dir" and store it at a numpy Object for return
        def read_data(file_dir, file_num, file_list):
            all_data = []
            for i in range(file_num):
                if (data_config['npy']):
                    tmp = np.load(file_dir + '/' + file_list[i])  # 读取npy文件
                else:
                    '''read matlab scat data .mat format'''
                    # name = data_dir + filename_list[i]
                    # t = scio.loadmat(name)
                    tmp = scio.loadmat(file_dir + '/' + file_list[i])['all_buf']  # 读取.mat文件
                '''read origin image'''
                # tmp = (np.ascontiguousarray(Image.open(data_dir+'/'+filename_list[i]), dtype=np.uint8).
                #     transpose((2, 0, 1)) / 127.5) - 1.0
                if (data_config['pca_method'] == 2):
                    tmp = tmp.reshape(len(tmp), -1)
                    tmp = np.sum(tmp, axis=0)
                all_data.append(tmp)

            return np.array(all_data)

        # This function is to read data from "file_dir" but with a batches size
        # so that we don't need large memory to store the data!!!
        def read_data_batches(file_dir_train, file_num_train, file_list_train,file_dir_test, file_num_test,
                              file_list_test, batch_size = 1024):

            for i in range(0,file_num_test+file_num_train,batch_size):

                all_data = []
                if i < file_num_train:
                    print("start read file: {} -> {} in train set".format(i + 1, i + batch_size))
                    for j in range(batch_size):
                        if (data_config['npy']):
                            tmp = np.load(file_dir_train + '/' + file_list_train[i + j])  # 读取npy文件
                        else:
                            '''read matlab scat data .mat format'''
                            # name = data_dir + filename_list[i]
                            # t = scio.loadmat(name)
                            tmp = scio.loadmat(file_dir_train + '/' + file_list_train[i + j])['all_buf']  # 读取.mat文件

                        '''read origin image'''
                        # tmp = (np.ascontiguousarray(Image.open(data_dir+'/'+filename_list[i]), dtype=np.uint8).
                        #     transpose((2, 0, 1)) / 127.5) - 1.0
                        if (data_config['pca_method'] == 2):
                            tmp = tmp.reshape(len(tmp), -1)
                            tmp = np.sum(tmp, axis=0)
                        all_data.append(tmp)

                else:
                    print("start read file: {} -> {} in test set".format(i + 1- num_train, i + batch_size - num_train))
                    for j in range(batch_size):
                        if (data_config['npy']):
                            tmp = np.load(file_dir_test + '/' + file_list_test[i + j - num_train])  # 读取npy文件
                        else:
                            '''read matlab scat data .mat format'''
                            # name = data_dir + filename_list[i]
                            # t = scio.loadmat(name)
                            tmp = scio.loadmat(file_dir_test + '/' + file_list_test[i + j -num_train])['all_buf']  # 读取.mat文件

                        '''read origin image'''
                        # tmp = (np.ascontiguousarray(Image.open(data_dir+'/'+filename_list[i]), dtype=np.uint8).
                        #     transpose((2, 0, 1)) / 127.5) - 1.0
                        if (data_config['pca_method'] == 2):
                            tmp = tmp.reshape(len(tmp), -1)
                            tmp = np.sum(tmp, axis=0)
                        all_data.append(tmp)

                yield np.array(all_data)

        # read data from train_set and test_set
        train_list = os.listdir(data_dir_train)
        test_list = os.listdir(data_dir_test)
        train_list.sort()
        test_list.sort()
        num_train = len(train_list)
        num_test = len(test_list)

        # for the case that the matrix is too large that is memory expensive, we use IncrementalPCA rather than PCA,
        # and here for celebA: 128x128(larger than 10000) will generate too much features, and hence will use too
        # much memory
        if self.img_shape[0] * self.img_shape[1] > 10000:
            # Note here: the batch_size must be 2^n since we have not care about the situation when the last batch is
            # not a full batch
            batch_size = 2048
            print("We choose IncrementalPCA since the data is too large")
            pca = IncrementalPCA(n_components=data_config['pca_outdim'])

            print('-'*20+"Start fitting the Incremental PCA"+'-'*20)
            data_iter = read_data_batches(data_dir_train,num_train,train_list,data_dir_test,num_test,test_list,
                                          batch_size=batch_size)
            for X_batch in data_iter:
                # print("The Batch's shape is",X_batch.shape)
                pca.partial_fit(X_batch)
            print('-' * 20 + "Finish fitting the Incremental PCA, let's start make dimensionality reduction" + '-' * 20)

            print('~' * 20 + "Start using Incremental PCA to make dimensionality reduction" + '~' * 20)
            count = 0
            for X_batch in read_data_batches(data_dir_train,num_train,train_list,data_dir_test,num_test,test_list,batch_size=batch_size):
                X_result = pca.transform(X_batch)

                print("Saving the reductive features: {0}".format(count+1))

                for j in range(len(X_result)):
                    if (count < num_train):
                        str_temp = train_list[j+count].split('.')
                        np.save(out_dir_train + '/' + str_temp[0] + '.npy', X_result[j])
                    else:
                        str_temp = test_list[j+count-num_train].split('.')
                        np.save(out_dir_test + '/' + str_temp[0] + '.npy', X_result[j])
                count += X_result.shape[0]

            print('~' * 20 + "Finish dimensionality reduction" + '~' * 20)

        else:
            print("We choose PCA for our dimensionality reduction")
            pca = PCA(n_components=data_config['pca_outdim'], copy=True)

            all_data_train = read_data(data_dir_train, num_train, train_list)
            print("The train data's shape is :", all_data_train.shape)
            all_data_test = read_data(data_dir_test, num_test, test_list)
            print("The test data's shape is :", all_data_test.shape)

            # merge the data to all_data
            all_data = np.concatenate((all_data_train, all_data_test), axis=0)
            # then, release the memory
            del all_data_train, all_data_test
            gc.collect()

            print("After merge, the whole data's shape is :", all_data.shape)

            X = all_data - (all_data.sum(axis=0) / len(all_data))  # 去均值,that is, make the features zero-centered
            X = pca.fit_transform(X)

            print("shape of all data after PCA:", X.shape)  # (50000,512) if we use cifar-10 in train set

            # store the features after the dimensionality reduction
            for j in range(len(all_data)):
                # store the train data
                if j < num_train:
                    str_temp = train_list[j].split('.')
                    np.save(out_dir_train + '/' + str_temp[0] + '.npy', X[j])
                # store the test data
                else:
                    str_temp = test_list[j - num_train].split('.')
                    np.save(out_dir_test + '/' + str_temp[0] + '.npy', X[j])



    def scat_data(self):

        print("Note here we choose to use scatNet to extract features.")

        scat_J = self.scat_J
        img_shape = self.img_shape
        batch_size = self.scat_batchsize

        def _scat_data1(scat_img_dir, scat_out_dir):

            filename_list = os.listdir(scat_img_dir)  # read the directory files's name
            filename_list.sort()
            count = len(filename_list)
            if (self.cuda):
                scat = Scattering2D(J=scat_J, shape=img_shape).cuda()  # scattering transform
            else:
                scat = Scattering2D(J=scat_J, shape=img_shape)

            batch_image = []
            for count_idx in range(0, count):
                imgDir = os.path.join(scat_img_dir, os.path.basename(filename_list[count_idx]))
                img = np.float32((np.array(Image.open(imgDir)) / 127.5 - 1.0)).transpose(2, 0, 1)  # 读取彩色图像，3通道做散射操作
                # img = np.float16((np.array(Image.open(imgDir).convert('L'))/127.5 - 1.0))#灰度式式取取
                batch_image.append(img)
                if ((count_idx + 1) % batch_size == 0 or count_idx == count - 1):
                    print("In processing images: {}".format(count_idx + 1))

                    if (self.cuda):
                        batch_image = torch.from_numpy(np.array(batch_image)).cuda()
                        batch_scat = scat.forward(batch_image)
                        batch_scat = batch_scat.cpu()
                    else:
                        batch_image = torch.from_numpy(np.array(batch_image))
                        batch_scat = scat.forward(batch_image)

                    for c in range(len(batch_image)):
                        img_scat = batch_scat[c]
                        str1 = filename_list[c + (int(count_idx / batch_size)) * batch_size].split('.')
                        np.save(scat_out_dir + '/' + str1[0] + '.npy', img_scat)
                    batch_image = []

            print("Scattering transform over for {} -> {}".format(scat_img_dir, scat_out_dir))
            return

        _scat_data1(self.train_img_dir, self.train_scat_dir)
        _scat_data1(self.test_img_dir, self.test_scat_dir)

    def fr_scat_data(self, alpha_1 = 0.4, alpha_2 = 1):

        print("Note here we choose to use fractional scatNet to extract features.")
        print("Where alpha_1 = {0}, alpha_2 = {1}, okay, Let's go!".format(alpha_1, alpha_2))

        scat_J = self.scat_J
        img_shape = self.img_shape
        batch_size = self.scat_batchsize

        def _fr_scat_data1(scat_img_dir, scat_out_dir):

            filename_list = os.listdir(scat_img_dir)  # read the directory files's name
            filename_list.sort()
            count = len(filename_list)

            if (self.cuda):
                scat = Fr_Scattering2D(J=scat_J, shape=img_shape, alpha_1=alpha_1, alpha_2=alpha_2).cuda()
            else:
                scat = Fr_Scattering2D(J=scat_J, shape=img_shape, alpha_1=alpha_1, alpha_2=alpha_2)

            batch_image = []
            for count_idx in range(0, count):
                imgDir = os.path.join(scat_img_dir, os.path.basename(filename_list[count_idx]))
                img = np.float32((np.array(Image.open(imgDir)) / 127.5 - 1.0)).transpose(2, 0, 1)  # 读取彩色图像，3通道做散射操作
                # img = np.float16((np.array(Image.open(imgDir).convert('L'))/127.5 - 1.0))#灰度式式取取
                batch_image.append(img)
                if ((count_idx + 1) % batch_size == 0 or count_idx == count - 1):
                    print("In processing images: {}".format(count_idx + 1))

                    if (self.cuda):
                        batch_image = torch.from_numpy(np.array(batch_image)).cuda()
                        batch_scat = scat.forward(batch_image)
                        batch_scat = batch_scat.cpu()
                    else:
                        batch_image = torch.from_numpy(np.array(batch_image))
                        batch_scat = scat.forward(batch_image)

                    for c in range(len(batch_image)):
                        img_scat = batch_scat[c]
                        str1 = filename_list[c + (int(count_idx / batch_size)) * batch_size].split('.')
                        np.save(scat_out_dir + '/' + str1[0] + '.npy', img_scat)
                    batch_image = []

            print("Fractional Scattering transform over for {} -> {}".format(scat_img_dir, scat_out_dir))
            return

        _fr_scat_data1(self.train_img_dir, self.train_scat_dir)
        _fr_scat_data1(self.test_img_dir, self.test_scat_dir)


    def nopca_reshape(input_dir, output_dir):
        filename_list = os.listdir(input_dir)  # read the directory files's name
        filename_list.sort()  # 对读取的文件名进行排序
        for i in range(len(filename_list)):
            '''read scat data .npy format'''
            if (data_config['npy']):
                tmp = np.load(input_dir + '/' + filename_list[i])  # 读取npy文件
            else:
                '''read matlab scat data .mat format'''
                name = input_dir + filename_list[i]
                t = scio.loadmat(name)
                tmp = scio.loadmat(input_dir + filename_list[i])['all_buf']  # 读取.mat文件
            tmp = tmp.reshape(-1)
            str = output_dir + '/' + filename_list[i]
            np.save(str, tmp)
            if (i % 100 == 0):
                print("step is {}".format(i))

    #   This function is to unpicke the datesets
    #   Here we use Cifar-10 as our datasets, and it is stored at "/home/iuxiang/datasets/cifar-10-batches-py/"
    def cifar10_unpickle(self, file='/home/qiuxiang/datasets/cifar-10-batches-py/'):

        # create dirs to store the unpicked images
        if (os.path.exists(file + "train-images") == False):
            os.makedirs(file + "train-images")
        if (os.path.exists(file + "test-images") == False):
            os.makedirs(file + "test-images")

        import pickle
        for idx in range(1, 6):
            filename = file + 'data_batch_' + str(idx)
            with open(filename, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            images = dict[b'data']
            images = images.reshape(10000, 3, 32, 32)
            for i in range(10000):
                img0 = Image.fromarray(images[i][0])
                img1 = Image.fromarray(images[i][1])
                img2 = Image.fromarray(images[i][2])
                img = Image.merge('RGB', (img0, img1, img2))
                img.save(file + 'train-images/' + str((idx - 1) * 10000) + str(i) + '.jpg')

        filename = file + 'test_batch'
        with open(filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        images = dict[b'data']
        images = images.reshape(10000, 3, 32, 32)
        for i in range(10000):
            img0 = Image.fromarray(images[i][0])
            img1 = Image.fromarray(images[i][1])
            img2 = Image.fromarray(images[i][2])
            img = Image.merge('RGB', (img0, img1, img2))
            img.save(file + 'test-images/' + str(i) + '.jpg')

    # For this function, we need to create some dirs to store the data  extract from the images
    def mkdirs(self):
        if (not os.path.exists(data_config['train_data_dir'])):
            os.makedirs(data_config['train_data_dir'])
        if (not os.path.exists(data_config['test_data_dir'])):
            os.makedirs(data_config['test_data_dir'])
        if (not os.path.exists(data_config['test_scat_dir'])):
            os.makedirs(data_config['test_scat_dir'])
        if (not os.path.exists(data_config['test_norm_dir'])):
            os.makedirs(data_config['test_norm_dir'])
        if (not os.path.exists(data_config['train_scat_dir'])):
            os.makedirs(data_config['train_scat_dir'])
        if (not os.path.exists(data_config['train_norm_dir'])):
            os.makedirs(data_config['train_norm_dir'])
        return

    def judge_name(self, dir_x, dir_scat):
        filename_list = os.listdir(dir_x)  # read the directory files's name
        filename_list.sort()  # 对读取的文件名进行排序
        scatname_list = os.listdir(dir_scat)  # read the directory files's name
        scatname_list.sort()  # 对读取的文件名进行排序
        for idx in range(len(scatname_list)):
            tmp1 = filename_list[idx].split('.')[0]
            tmp2 = scatname_list[idx].split('.')[0]
            if (tmp2 != tmp1):
                print("----------------")
            assert tmp1 == tmp2
            if (idx % 100 == 0):
                print("----{}".format(idx))

    '''128大小黄种人图像转为32大小黄种人图像全部操作'''

    def huangzhongren_to_32(self):
        data_root = "/home/jains/datasets/huang_zhong_ren/celebA32/"  # 数据根目录

        def resize128to32():  # 128大小图像resize为32大小突袭党
            image128dir = "/home/jains/datasets/huang_zhong_ren/celebA/65536/"
            image32dir = "/home/jains/datasets/huang_zhong_ren/celebA/65536bak/"
            imagename_list = os.listdir(image128dir)
            for idx in range(len(imagename_list)):
                img_tmp = Image.open(os.path.join(image128dir, imagename_list[idx])).resize((32, 32))
                img_tmp.save(image32dir + imagename_list[idx])

        # resize128to32()

        test_img_dir = "2048_after_65536/"
        train_img_dir = "65536/"

        def make_directory():  # 创建存储数据的文件夹
            if (not os.path.exists(data_root + "2048_after_65536_ScatJ4/")):
                os.makedirs(data_root + "2048_after_65536_ScatJ4/")
            if (not os.path.exists(data_root + "65536_ScatJ4/")):
                os.makedirs(data_root + "65536_ScatJ4/")
            if (not os.path.exists(data_root + "65536_ScatJ4_projected512_1norm/")):
                os.makedirs(data_root + "65536_ScatJ4_projected512_1norm/")
            if (not os.path.exists(data_root + "2048_after_65536_ScatJ4_projected512_1norm/")):
                os.makedirs(data_root + "2048_after_65536_ScatJ4_projected512_1norm/")

        make_directory()  # 创建存储数据的文件夹

        def make_data():
            # scat_data1(data_root+test_img_dir, data_root+"2048_after_65536_ScatJ4/", (32,32), 4)
            self.pca_data(data_root + "2048_after_65536_ScatJ4/",
                          data_root + "2048_after_65536_ScatJ4_projected512_1norm/")
            # scat_data1(data_root + train_img_dir, data_root + "65536_ScatJ4/", (32, 32), 4)
            self.pca_data(data_root + "65536_ScatJ4/", data_root + "65536_ScatJ4_projected512_1norm/")

        make_data()  # 计算所有数据：散射数据、pca数据
        return

        '''可视化最后激活层前特征值的范围'''

        def plot_arrange():
            max_value = []
            min_value = []
            data_file_dir = "/Users/jains/Desktop/max_min_record.txt"
            max_row = 304853
            x = range(max_row)
            with open(data_file_dir) as f:
                line = f.readline()
                idx = 0
                while line and idx < max_row:
                    max_value.append(float(line.split(' ')[0]))
                    min_value.append(float(line.split(' ')[1].split('\n')[0]))
                    line = f.readline()
                    idx += 1
            plt.plot(x, max_value, label='max--')
            plt.plot(x, min_value, label='min--')
            plt.xlabel('iterations')
            plt.ylabel('values')
            plt.legend()
            plt.show()
            return
        # plot_arrange()#函数调用

    def FusionFeature(self, inputFeaturesDir, outputFeatureDir="./fusion_data/"):
        _dirNameList = os.listdir(inputFeaturesDir)
        _dirNameList.sort()  # 代表文件夹名字
        _fileNameList = []  # 表示某一个文件夹下所有文件名，不包含路径
        _fileLen = get_nb_files(inputFeaturesDir + _dirNameList[0] + '/')  # 某一种特征的文件夹中文件个数

        for i in range(len(_dirNameList)):
            assert _fileLen == get_nb_files(inputFeaturesDir + _dirNameList[i] + '/')  # 确保每个特征文件夹下的文件个数一致
            _fileNameList.append(os.listdir(inputFeaturesDir + _dirNameList[i] + '/'))
            _fileNameList.sort()

        _features = []  # 用来保存多个相同文件名的特征
        for i in range(len(_fileNameList[0])):
            _fileNameTmp = _fileNameList[0][i]  # 以第一个文件夹内的特征名作为标准对比后续读取的文件名是否一致
            for j in range(len(_dirNameList)):
                assert _fileNameTmp == _fileNameList[j][i]
                _tmpFileName = inputFeaturesDir + _dirNameList[j] + '/' + _fileNameList[j][i]
                _tmpData = np.load(_tmpFileName)
                _features.append(_tmpData)
            # 求平均值
            print("step {}".format(i / 1000))
            _featureArray = np.array(_features).mean(axis=0)
            _features.clear()  # _feature必须清空，不然持续添加
            #     保存特征平均值
            _outFileName = outputFeatureDir + _fileNameList[0][i]
            np.save(_outFileName, _featureArray)

        # print(_fileNameList[0])

    # inputFeaturesDir = "/home/jains/datasets/gsndatasets/fusion_feature/"
    # outputFeatureDir = "/home/jains/datasets/gsndatasets/fusion_data/"
    # FusionFeature(inputFeaturesDir=inputFeaturesDir,outputFeatureDir=outputFeatureDir)

    # 对图像旋转
    def ImageRotation(self, inputImageDir, outputImageDir, angle):
        _fileNameList = os.listdir(inputImageDir)  # 文件名
        _fileNameList.sort()
        for i in range(len(_fileNameList)):
            img = Image.open(inputImageDir + _fileNameList[i])
            img = img.rotate(angle)
            img.save(outputImageDir + _fileNameList[i])
            print("step is {}".format(i / 100))

    # inputImageDir = "/home/jains/datasets/gsndatasets/celebA/65536/"
    # outputImageDir = "/home/jains/datasets/gsndatasets/celebA_R270/65536/"
    # ImageRotation(inputImageDir,outputImageDir,270)
