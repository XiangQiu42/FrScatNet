import numpy as np
import os
import torch
from torch.autograd import Variable
from PIL import Image
def get_nb_files(input_dir):
    list_files = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    return len(list_files)

import pandas as pd

def read_run_data(data_dir=""):
    filename = data_dir + data_dir.split('/')[-2]+'_loss.csv'
    print(filename)
    if(os.path.exists(filename)):
        df = pd.read_csv(filename,dtype={'epoch':np.int8})
        df.set_index(df['epoch'], inplace=True)
    else:
        df = pd.DataFrame(columns=['epoch','loss','train_psnr','test_psnr','train_ssim','test_ssim'])
        df.set_index(df['epoch'], inplace=True)
        df.to_csv(filename,index=False)
    return df

def save_run_data(data_dir = "",df=None):
    filename = data_dir + data_dir.split('/')[-2]+'_loss.csv'
    if(df is not None):
        df.to_csv(filename,index=False)
    return

def test_csv_file():
    df = read_run_data('./')
    df.loc[0] = [0, 1, 2, 3, 4, 5]
    df.loc[1] = [1, 2, 2, 2, 2, 2]
    print(df)
    df.set_index(df['epoch'], inplace=True)
    df.loc[10] = [10, 0, 12, 0, 13, 0]
    print(df)
    save_run_data("./", df=df)
    return None
def cal_avg(dir=""):
    df = read_run_data(dir)
    psnr_avg = 0
    ssim_avg = 0
    psnr_max = 0
    ssim_max = 0
    return psnr_avg,psnr_max,ssim_avg,ssim_max

def cal_PSNR_SSIM(g,batch,batch_size):
    import numpy as np
    from skimage.measure import compare_ssim, compare_nrmse, compare_psnr, compare_mse
    psnr=0
    ssim=0

    z = Variable(batch['x']).type(torch.FloatTensor)
    g.eval()
    g_x = g.forward(z)

    g_x = np.uint8((g_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
    x = np.uint8(((batch['x'].cpu().numpy().transpose((0, 2, 3, 1)))+1)*127.5)
    for ii in range (batch_size):
        psnr+=compare_psnr(g_x[ii],x[ii])
        ssim+=compare_ssim(g_x[ii],x[ii],multichannel=True)
    ssim/=batch_size
    psnr/=batch_size
    return psnr,ssim

#cal_avg('/Users/jains/Desktop/data_record/cifar-10/feature_map/frscat_1_J3_57.csv')


    '''可视化分析数据'''
def analysis(loss_data_dir):
    '''可视化分析损失函数'''
    import matplotlib.pyplot as plt
    def plot_loss_picture(df_list, min_x, labels):
        plt.figure(1)

        losses = []
        testloss = []
        train_psnrs = []
        train_ssims = []
        test_psnrs = []
        test_ssims = []
        x = range(0, min_x)
        for idx in range(len(df_list)):
            losses.append(df_list[idx]['loss'][:min_x])
            # testloss.append(df_list[idx]['test_loss'][:min_x])
            train_psnrs.append(df_list[idx]['train_psnr'][:min_x])
            test_psnrs.append(df_list[idx]['test_psnr'][:min_x])
            train_ssims.append(df_list[idx]['train_ssim'][:min_x])
            test_ssims.append(df_list[idx]['test_ssim'][:min_x])

        # 对比loss
        # plt.subplot(311)
        # plt.title('Loss')
        # for idx in range(len(df_list)):
        #     plt.plot(x, losses[idx], label='Train loss')#labels[idx])
        #     plt.plot(x, testloss[idx], label='Test loss')  # labels[idx])
        # plt.legend()
        # plt.ylabel('Loss')
        # plt.xlabel('epoch')
        # plt.show()

        # 对比psnr
        plt.subplot(211)
        plt.title('PSNR Score')
        for idx in range(len(df_list)):
            # plt.plot(x, train_psnrs[idx], label=labels[idx] + '_train')
            # plt.plot(x, test_psnrs[idx], label=labels[idx] + '_test')
            plt.plot(x, train_psnrs[idx], label='train')  # labels[idx] + '_train')
            plt.plot(x, test_psnrs[idx], label='test')  # labels[idx] + '_test')
        plt.legend()
        plt.ylabel('PSNR')
        plt.xlabel('epoch')
        # plt.show()

        # 对比ssim
        plt.subplot(212)
        plt.title('SSIM Score')
        for idx in range(len(df_list)):
            # plt.plot(x, train_ssims[idx], label=labels[idx] + '_train')
            # plt.plot(x, test_ssims[idx], label=labels[idx] + '_test')
            plt.plot(x, train_ssims[idx], label='train')  # labels[idx] + '_train')
            plt.plot(x, test_ssims[idx], label='test')  # labels[idx] + '_test')
        plt.legend()
        plt.ylabel('SSIM')
        plt.xlabel('epoch')
        plt.show()
        return

    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    # plt.subplot(212)
    # plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    # plt.show()

    def analysis_loss(loss_data_dir):
        # loss_data_dir = "/Users/jains/Desktop/data_record/MPCA/cfar-10/"
        file_names = os.listdir(loss_data_dir)
        df = []
        min_x = 2000
        for count in range(len(file_names)):
            if (file_names[count].split('.')[-1] == 'csv' and os.path.exists(loss_data_dir + file_names[count])):
                df_tmp = pd.read_csv(loss_data_dir + file_names[count])
                if (df_tmp.shape[0] < min_x):
                    min_x = df_tmp.shape[0]
                df.append(df_tmp)
        plot_loss_picture(df, min_x, [x.split('.')[0] for x in file_names if (x.split('.')[1] == 'csv')])
        return None
    analysis_loss(loss_data_dir)  # 损失函数和重构指标函数调用

# analysis('/Users/jains/Desktop/data_record/contrast/')


# 对提取的特征聚类分析
def cluster():
    import matplotlib.pyplot as plt
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    data_root = "/Users/jains/datasets/"
    data_dir = "cfar-10/test/"
    filename_list = os.listdir(data_root + data_dir)
    all_data = []
    label_data = []
    for idx in range(10000):
        label_data.append(int(filename_list[idx].split('_')[0]))

        # 直接打开原图
        # data_tmp = (np.array(Image.open(data_root + data_dir + filename_list[idx]), 'f') / 127.5 - 1.0).reshape(
        #     -1, )

        # scat3，三通道值累加
        # tmp = np.load(data_root+data_dir+filename_list[idx])
        # tmp = tmp.reshape(len(tmp), -1)
        # data_tmp = np.sum(tmp, axis=0)

        # scat
        data_tmp = np.load(data_root+data_dir+filename_list[idx]).reshape(-1,)
        all_data.append(data_tmp)
    all_data = np.array(all_data)

    # 去均值
    all_data = all_data - all_data.sum(axis=0) / len(all_data)

    # 标准化
    norms = np.sqrt(np.sum(all_data ** 2, axis=1))
    norms = np.expand_dims(norms, axis=1)
    norms = np.repeat(norms, all_data.shape[1], axis=1)
    all_data /= norms

    # PCA压缩聚类显示
    global max_score, iddx
    max_score = 0
    # for idx in range(2,1000):
    reduce_alldata = PCA(n_components=2, copy=True).fit_transform(all_data)
    y_pred = KMeans(n_clusters=10, random_state=9).fit_predict(reduce_alldata)
    plt.scatter(reduce_alldata[:, 0], reduce_alldata[:, 1], c=y_pred)
    plt.show()
    from sklearn import metrics
    print(metrics.calinski_harabaz_score(reduce_alldata, y_pred))
    return
    # 评分
    # from sklearn import metrics
    # score = metrics.calinski_harabaz_score(reduce_alldata, y_pred)
    # print(idx,score)
    # if score > max_score:
    #     max_score= score
    #     iddx = idx
    #     print(max_score,iddx)

# cluster()
