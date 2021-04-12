# coding=utf-8


"""
Author: zhangjing
Date and time: 2/02/19 - 17:58
"""

import os
import math
from config.config import net_config
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from EmbeddingsImagesDataset import EmbeddingsImagesDataset
from utils import create_folder, normalize, create_name_experiment
import utilss.tools as tool
import matplotlib.pyplot as plt

if net_config['net_name'] == 'dcgan':
    from generator_architecture import Generator128_mpca as Generator, weights_init

    print("the net architecture is for {} ".format(net_config['net_name']))
elif net_config['net_name'] == 'celebA':
    from generator_architecture import Generator128_res2 as Generator, weights_init

    print("the net architecture is for {} ".format(net_config['net_name']))
elif net_config['net_name'] == 'cifar10':
    from generator_architecture import Generator32_3 as Generator, weights_init

    print("the net architecture is for {} ".format(net_config['net_name']))


class GSN:
    def __init__(self, parameters):
        dir_datasets = os.path.expanduser(parameters['dataset_root'])
        dir_experiments = os.path.expanduser(parameters['exper_path'])

        dataset = parameters['dataset']
        # train_attribute = parameters['train_attribute']
        # test_attribute = parameters['test_attribute']
        # embedding_attribute = parameters['embedding_attribute']

        self.dim = parameters['dim']
        self.nb_channels_first_layer = parameters['nb_channels_first_layer']

        name_experiment = create_name_experiment(parameters, 'NormL1')

        # self.dir_x_train = os.path.join(dir_datasets, dataset, '{0}'.format(train_attribute))
        # self.dir_x_test = os.path.join(dir_datasets, dataset, '{0}'.format(test_attribute))
        # self.dir_z_train = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(train_attribute, embedding_attribute))
        # self.dir_z_test = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(test_attribute, embedding_attribute))

        # give the dir where the original images and implicit vector Z stored at  
        self.dir_x_train = os.path.join(dir_datasets, parameters['train-images_filename'])
        self.dir_x_test = os.path.join(dir_datasets, parameters['test-images_filename'])
        self.dir_z_train = os.path.join(dir_datasets, dataset, parameters['train-norm_filename'])
        self.dir_z_test = os.path.join(dir_datasets, dataset, parameters['test-norm_filename'])

        self.images_generate = parameters['image_generate']

        self.dir_experiment = os.path.join(dir_experiments, name_experiment) + parameters['model_name']
        self.dir_models = os.path.join(self.dir_experiment, 'models')
        self.dir_logs = os.path.join(self.dir_experiment, 'logs')

        if not os.path.exists(self.dir_models):
            print('Name experiment: {}'.format(name_experiment))
        create_folder(self.dir_models)

        create_folder(self.dir_logs)

        self.batch_size = parameters['batch_size']
        self.nb_epochs_to_save = 1
        self.cuda = parameters['cuda']
        self.cfg = parameters

        self.last_activate = net_config['last_activate']  # 最后一层网络激活层
        # self.Generator = Generator
        print("***********activate layer is {} ********".format(self.last_activate))
        print("train data is in {} --- test data is in {} --- network config is {}\n".format(
            self.dir_x_train.split('/')[-1],
            self.dir_x_test.split('/')[-1],
            parameters['model_name']))

    def train(self, epoch_to_restore=0, epoch_train=50):
        print("------------train start------------------")

        '''训练开始'''
        g = Generator(self.nb_channels_first_layer, self.dim, last_activate=self.last_activate)

        # 读取csv文件记录所有训练数据
        from utilss.tools import read_run_data, save_run_data
        df = read_run_data(self.dir_experiment + '/')

        # 判断是否加载历史训练模型
        if epoch_to_restore > 0:
            filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch_to_restore))
            g.load_state_dict(torch.load(filename_model))
        else:
            g.apply(weights_init)

        # 判断是否使用cuda
        if (self.cuda):
            g.cuda()
        # 训练函数
        g.train()

        # 加载训练集dataloader
        dataset = EmbeddingsImagesDataset(self.dir_z_train, self.dir_x_train)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=self.cfg['num_workers'],
                                pin_memory=True)

        # '''查看网络结构图'''
        # batch = next(iter(dataloader))
        # z = batch['z'].type(torch.FloatTensor).cuda()
        # with SummaryWriter(comment='Net1')as w:
        #    w.add_graph(g, (z,))

        # validation set's dataloader
        fixed_dataloader = DataLoader(dataset, self.batch_size)  # 用作验证集的数据
        fixed_batch = next(iter(fixed_dataloader))  # 所有值 /127.5 - 1-------------iter获取容器的迭代器,next表示下一个

        # 测试集dataloader
        testdataset = EmbeddingsImagesDataset(self.dir_z_test, self.dir_x_test)
        testdataloader = DataLoader(dataset=testdataset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.cfg['num_workers'], pin_memory=True)
        testfixed_batch = next(iter(testdataloader))

        # 定义损失函数和优化器
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(g.parameters())

        # 定义tensorboard记录
        writer = SummaryWriter(self.dir_logs)

        # 开始训练
        try:
            epoch = epoch_to_restore
            while epoch < epoch_train:
                g.train()
                for _ in range(self.nb_epochs_to_save):
                    epoch += 1
                    print("epoch is : ", epoch)
                    for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                        g.zero_grad()
                        if (self.cuda):
                            x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                            z = Variable(current_batch['z']).type(torch.FloatTensor).cuda()
                        else:
                            x = Variable(current_batch['x']).type(torch.FloatTensor)
                            z = Variable(current_batch['z']).type(torch.FloatTensor)
                        # 前向传播，计算损失，反向传播
                        g_z = g.forward(z)
                        loss = criterion(g_z, x)
                        loss.backward()
                        optimizer.step()

                    # 计算每一个epoch的loss
                    print("----------------{}-------------".format(loss))
                    writer.add_scalar('train_loss', loss, epoch)

                # 计算PSNR，SSIM
                from skimage.measure import compare_ssim, compare_nrmse, compare_psnr, compare_mse
                if (epoch % 1 == 0):
                    test_psnr = 0
                    test_ssim = 0
                    val_psnr = 0
                    val_ssim = 0
                    if (self.cuda):
                        test_z = Variable(testfixed_batch['z']).type(torch.FloatTensor).cuda()
                        val_z = Variable(fixed_batch['z']).type(torch.FloatTensor).cuda()
                    else:
                        test_z = Variable(testfixed_batch['z']).type(torch.FloatTensor)
                        val_z = Variable(fixed_batch['z']).type(torch.FloatTensor).cuda()
                    g.eval()
                    gtest_x = g.forward(test_z)
                    if (net_config['train_x'] == '(-1,1)'):
                        gtest_x = np.uint8((gtest_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                        test_x = np.uint8(((testfixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1))) + 1) * 127.5)
                    elif (net_config['train_x'] == '(0,1)'):
                        gtest_x = np.uint8((gtest_x.data.cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                        test_x = np.uint8(((testfixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)))) * 255)
                    for ii in range(self.batch_size):
                        test_psnr += compare_psnr(gtest_x[ii], test_x[ii])
                        test_ssim += compare_ssim(gtest_x[ii], test_x[ii], multichannel=True)
                    test_ssim /= self.batch_size
                    test_psnr /= self.batch_size

                    gval_x = g.forward(val_z)
                    if (net_config['train_x'] == '(-1,1)'):
                        gval_x = np.uint8((gval_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                        val_x = np.uint8((fixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                    elif (net_config['train_x'] == '(0,1)'):
                        gval_x = np.uint8((gval_x.data.cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                        val_x = np.uint8((fixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                    for ii in range(self.batch_size):
                        val_psnr += compare_psnr(gval_x[ii], val_x[ii])
                        val_ssim += compare_ssim(gval_x[ii], val_x[ii], multichannel=True)
                    val_psnr /= self.batch_size
                    val_ssim /= self.batch_size

                    # 记录所有指标：loss，psnr和ssim, And save it as a csv file
                    df.loc[epoch] = [epoch, loss.cpu().data.numpy(), val_psnr, test_psnr, val_ssim, test_ssim]
                    save_run_data(self.dir_experiment + '/', df=df)

                    # 将结果添加到tensorboard中
                    writer.add_scalars('psnr', {'val_psnr': val_psnr, 'test_psnr': test_psnr}, epoch)
                    writer.add_scalars('ssim', {'val_ssim': val_ssim, 'test_psnr': test_ssim}, epoch)

                '''把验证集中的图片保存下来'''
                # images_tmp = np.uint8((make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose(
                #     (1, 2, 0)) + 1) * 127.5)  # 把图片数据转换到0-255之间
                # writer.add_image('generations', images_tmp, epoch)#把每个epoch生成的图片显示导tensorboard
                # Image.fromarray(images_tmp).save(self.dir_valimg + '/' + str(epoch) + '.jpg')  # 把每个epoch生成图片保存到本地

                '''添加指标得分到tensorboard'''
                # writer.add_scalars('mse_score', {'train_mse': mse, 'test_mse': test_mse}, epoch)
                # writer.add_scalars('psnr_score', {'train_psnr': psnr, 'test_psnr': test_psnr},
                #                    epoch)  # 把每个epoch评价指标在tensorboard显示
                # writer.add_scalars('ssim_score', {'train_ssim': ssim, 'test_ssim': test_ssim}, epoch)

                filename = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
                torch.save(g.state_dict(), filename)

        finally:
            print('*********************************Closing Writer**********************************.')
            writer.close()

    def save_originals(self):

        def _save_originals(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            # the second parameter indicate the picture we want to show is format of 3x3(if self.images_generate = 9)
            fixed_dataloader = DataLoader(dataset, self.images_generate)
            fixed_batch = next(iter(fixed_dataloader))

            temp = make_grid(fixed_batch['x'], nrow=int(math.sqrt(self.images_generate))).numpy().transpose((1, 2, 0))

            filename_images = os.path.join(self.dir_experiment, 'originals_{}.jpg'.format(train_test))
            Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

        _save_originals(self.dir_z_train, self.dir_x_train, 'train')
        _save_originals(self.dir_z_test, self.dir_x_test, 'test')

    def compute_avg_errors(self, epoch_start, epoch_end):
        from skimage.measure import compare_ssim, compare_nrmse, compare_psnr, compare_mse
        def compute_errors(epoch):
            filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
            print("model file name is {}".format(filename_model))
            g = Generator(self.nb_channels_first_layer, self.dim)
            g.cuda()
            g.load_state_dict(torch.load(filename_model))
            g.eval()
            criterion = torch.nn.MSELoss()

            def _compute_error(dir_z, dir_x, train_test):
                fileCount = len(os.listdir(dir_x + "/"))
                dataset = EmbeddingsImagesDataset(dir_z, dir_x)
                dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)
                error = 0
                ssim = 0
                psnr = 0
                for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                    x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                    z = Variable(current_batch['z']).type(torch.FloatTensor).cuda()
                    g_z = g.forward(z)
                    error += criterion(g_z, x).data.cpu().numpy()

                    for idx_ii in range(g_z.data.cpu().numpy().shape[0]):
                        out = g_z.data.cpu().numpy()[idx_ii].transpose((1, 2, 0))
                        x_out = x.data.cpu().numpy()[idx_ii].transpose((1, 2, 0))
                        ssim += compare_ssim(x_out, out, multichannel=True)  # ji suan dan zhang ssim
                        psnr += compare_psnr(x_out, out)

                error /= len(dataloader)
                ssim /= fileCount
                psnr /= fileCount  # pingjun zhi
                print('Error for {}: {}'.format(train_test, error))
                return error, psnr, ssim

            return _compute_error(self.dir_z_train, self.dir_x_train, 'train')

        avg_error = 0
        avg_psnr = 0
        avg_ssim = 0
        min_MSELoss = 1000
        max_psnr = 0
        max_ssim = 0
        min_MseEpoch = 0
        max_PsnrEpoch = 0
        max_SsimEpoch = 0

        for idx in range(epoch_start, epoch_end):
            print("Now epoch is {}".format(idx))
            mse, psnr, ssim = compute_errors(idx + 1)
            if mse < min_MSELoss:
                min_MSELoss = mse
                min_MseEpoch = idx
            avg_error += mse
            if psnr > max_psnr:
                max_psnr = psnr
                max_PsnrEpoch = idx
            avg_psnr += psnr
            if ssim > max_ssim:
                max_ssim = ssim
                max_SsimEpoch = idx
            avg_ssim += ssim

        print(
            "epoch{} to epoch{} average MSEloss is {},min MSEloss is {} and epoch is {}".format(epoch_start, epoch_end,
                                                                                                avg_error / (
                                                                                                        epoch_end - epoch_start + 1),
                                                                                                min_MSELoss,
                                                                                                min_MseEpoch))
        print("epoch{} to epoch{} average PSNR is {},max PSNR is {} and epoch is {}".format(epoch_start, epoch_end,
                                                                                            avg_psnr / (
                                                                                                    epoch_end - epoch_start + 1),
                                                                                            max_psnr, max_PsnrEpoch))
        print("epoch{} to epoch{} average SSIM is {},max SSIM is {} and epoch is {}".format(epoch_start, epoch_end,
                                                                                            avg_ssim / (
                                                                                                    epoch_end - epoch_start + 1),
                                                                                            max_ssim, max_SsimEpoch))

    def compute_errors(self, epoch):
        filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
        g = Generator(self.nb_channels_first_layer, self.dim)
        g.cuda()

        g.load_state_dict(torch.load(filename_model))
        g.eval()
        # //0.236449167408864
        #         0.09840121258639556
        #         0.09840121260458545
        #         0.0984012124772562
        criterion = torch.nn.MSELoss()

        def _compute_error(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

            error = 0

            for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                z = Variable(current_batch['z']).type(torch.FloatTensor).cuda()
                g_z = g.forward(z)

                error += criterion(g_z, x).data.cpu().numpy()

            error /= len(dataloader)

            print('Error for {}: {}'.format(train_test, error))

        print("Display the error of train and test:")
        _compute_error(self.dir_z_train, self.dir_x_train, 'train')
        _compute_error(self.dir_z_test, self.dir_x_test, 'test')

    def generate_from_model(self, epoch):
        filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
        g = Generator(self.nb_channels_first_layer, self.dim)
        g.load_state_dict(torch.load(filename_model))
        g.cuda()
        g.eval()

        def _generate_from_model(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            fixed_dataloader = DataLoader(dataset, self.images_generate, shuffle=True)
            fixed_batch = next(iter(fixed_dataloader))

            # save original images
            images_originals_filename = os.path.join(self.dir_experiment, '{}_originals.jpg'.format(train_test))
            temp_x = make_grid(fixed_batch['x'], nrow=int(math.sqrt(self.images_generate))).numpy().transpose((1, 2, 0))
            Image.fromarray(np.uint8((temp_x + 1) * 127.5)).save(images_originals_filename)

            z = Variable(fixed_batch['z']).type(torch.FloatTensor).cuda()
            g_z = g.forward(z)
            filename_images = os.path.join(self.dir_experiment, 'epoch_{}_{}.jpg'.format(epoch, train_test))
            temp = make_grid(g_z.data[:self.images_generate],
                             nrow=int(math.sqrt(self.images_generate))).cpu().numpy().transpose((1, 2, 0))
            Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

        print("Save images generated from epoch:{}".format(epoch))
        _generate_from_model(self.dir_z_train, self.dir_x_train, 'train')
        _generate_from_model(self.dir_z_test, self.dir_x_test, 'test')
        print("Finish Saving images generated from epoch:{}".format(epoch))

        def _generate_path(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            fixed_dataloader = DataLoader(dataset, 2, shuffle=True)
            fixed_batch = next(iter(fixed_dataloader))

            z0 = fixed_batch['z'][[0]].numpy()
            z1 = fixed_batch['z'][[1]].numpy()

            batch_z = np.copy(z0)

            nb_samples = 100

            interval = np.linspace(0, 1, nb_samples)
            for t in interval:
                if t > 0:
                    zt = normalize((1 - t) * z0 + t * z1)
                    batch_z = np.vstack((batch_z, zt))

            z = Variable(torch.from_numpy(batch_z)).type(torch.FloatTensor).cuda()
            g_z = g.forward(z)

            # filename_images = os.path.join(self.dir_experiment, 'path_epoch_{}_{}.png'.format(epoch, train_test))
            # temp = make_grid(g_z.data, nrow=nb_samples).cpu().numpy().transpose((1, 2, 0))
            # Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

            g_z = g_z.data.cpu().numpy().transpose((0, 2, 3, 1))

            folder_to_save = os.path.join(self.dir_experiment, 'epoch_{}_{}_path'.format(epoch, train_test))
            create_folder(folder_to_save)

            for idx in range(nb_samples):
                filename_image = os.path.join(folder_to_save, '{}.png'.format(idx))
                Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)

        # _generate_path(self.dir_z_train, self.dir_x_train, 'train')
        # _generate_path(self.dir_z_test, self.dir_x_test, 'test')

        def _generate_random():
            nb_samples = self.images_generate
            z = np.random.randn(nb_samples, self.dim)
            print("z-min is {}, z-max is {}".format(z.min(), z.max()))
            norms = np.sqrt(np.sum(z ** 2, axis=1))
            norms = np.expand_dims(norms, axis=1)
            norms = np.repeat(norms, self.dim, axis=1)
            z /= norms
            print("After normalization, z-min is {}, z-max is {}".format(z.min(), z.max()))
            z = Variable(torch.from_numpy(z)).type(torch.FloatTensor).cuda()
            g_z = g.forward(z)
            filename_images = os.path.join(self.dir_experiment, 'epoch_{}_random.png'.format(epoch))
            temp = make_grid(g_z.data[:self.images_generate],
                             nrow=int(math.sqrt(self.images_generate))).cpu().numpy().transpose((1, 2, 0))
            Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

        print("\nGenerate images from random numbers")
        _generate_random()
        print("Finishing Generating images from random numbers")

    # def cacul_model(self,start_epoch,end_epoch):

    # this function is to a analyse one epoch during train
    def analyze_model_epoch(self, epoch):
        filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
        g = Generator(self.nb_channels_first_layer, self.dim)
        g.cuda()
        g.load_state_dict(torch.load(filename_model))
        g.eval()

        nb_samples = 50
        batch_z = np.zeros((nb_samples, 32 * self.nb_channels_first_layer, 4, 4))
        # batch_z = np.maximum(5*np.random.randn(nb_samples, 32 * self.nb_channels_first_layer, 4, 4), 0)
        # batch_z = 5 * np.random.randn(nb_samples, 32 * self.nb_channels_first_layer, 4, 4)

        for i in range(4):
            for j in range(4):
                batch_z[:, :, i, j] = create_path(nb_samples)
        # batch_z[:, :, 0, 0] = create_path(nb_samples)
        # batch_z[:, :, 0, 1] = create_path(nb_samples)
        # batch_z[:, :, 1, 0] = create_path(nb_samples)
        # batch_z[:, :, 1, 1] = create_path(nb_samples)
        batch_z = np.maximum(batch_z, 0)

        z = Variable(torch.from_numpy(batch_z)).type(torch.FloatTensor).cuda()
        temp = g.main._modules['4'].forward(z)
        for i in range(5, 10):
            temp = g.main._modules['{}'.format(i)].forward(temp)

        g_z = temp.data.cpu().numpy().transpose((0, 2, 3, 1))

        folder_to_save = os.path.join(self.dir_experiment, 'epoch_{}_path_after_linear_only00_path'.format(epoch))
        create_folder(folder_to_save)

        for idx in range(nb_samples):
            filename_image = os.path.join(folder_to_save, '{}.png'.format(idx))
            Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)

    def compare_scatNets(self, dir_scat, dir_fr_scat):
        """ compare the different results based on different feature extraction method

                Parameters
                ----------
                dir_scat : string
                    file dir where loss and PSNR or SSIM stored(ScatNet)
                dir_fr_scat: string
                    file dir where loss and PSNR or SSIM stored(FrScatNet)
        """
        df_scat = tool.read_run_data(dir_scat + '/')
        df_fr_scat = tool.read_run_data(dir_fr_scat + '/')
        Max_epoch = max(df_scat.shape[0], df_fr_scat.shape[0])
        X = range(0, Max_epoch)

        losses = {'scat': [], 'frScat': []}
        testloss = {'scat': [], 'frScat': []}
        train_psnrs = {'scat': [], 'frScat': []}
        train_ssims = {'scat': [], 'frScat': []}
        test_psnrs = {'scat': [], 'frScat': []}
        test_ssims = {'scat': [], 'frScat': []}

        losses['scat'].append(df_scat['loss'])
        # testloss['scat''].append(df_scat['test_loss'])
        train_psnrs['scat'].append(df_scat['train_psnr'])
        test_psnrs['scat'].append(df_scat['test_psnr'])
        train_ssims['scat'].append(df_scat['train_ssim'])
        test_ssims['scat'].append(df_scat['test_ssim'])

        losses['frScat'].append(df_fr_scat['loss'])
        # testloss['frScat'''].append(df_fr__scat['test_loss'])
        train_psnrs['frScat'].append(df_fr_scat['train_psnr'])
        test_psnrs['frScat'].append(df_fr_scat['test_psnr'])
        train_ssims['frScat'].append(df_fr_scat['train_ssim'])
        test_ssims['frScat'].append(df_fr_scat['test_ssim'])

        plt.figure(1)
        # 对比 psnr
        plt.subplot(121)
        plt.title('PSNR Score')

        for key in train_psnrs.keys():
            plt.plot(X, train_psnrs[key][0], label=key + '_train')
        for key in test_psnrs.keys():
            plt.plot(X, test_psnrs[key][0], label=key + '_train')

        plt.legend()
        plt.ylabel('PSNR')
        plt.xlabel('epoch')
        # plt.show()

        # 对比ssim
        plt.subplot(122)
        plt.title('SSIM Score')
        for key in train_ssims.keys():
            plt.plot(X, train_ssims[key][0], label=key + '_train')
        for key in test_ssims.keys():
            plt.plot(X, test_ssims[key][0], label=key + '_train')
        plt.legend()
        plt.ylabel('SSIM')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(self.dir_experiment, 'PNSR&SSIM_compare.jpg'))
        plt.show()

    # This function try to analyse our model, which include the loss curve,
    # the compare of PNSR score and so on...
    # note that there is another similar function in utilss.tool
    def display_model_RESULT(self):
        df = tool.read_run_data(self.dir_experiment + '/')

        Max_epoch = df.shape[0]
        plt.figure(1)
        losses = []
        testloss = []
        train_psnrs = []
        train_ssims = []
        test_psnrs = []
        test_ssims = []
        x = range(0, Max_epoch)

        losses.append(df['loss'])
        # testloss.append(df['test_loss'])
        train_psnrs.append(df['train_psnr'])
        test_psnrs.append(df['test_psnr'])
        train_ssims.append(df['train_ssim'])
        test_ssims.append(df['test_ssim'])

        # 对比 loss

        plt.title('Loss')
        plt.plot(x, losses[0], label='Train loss')  # labels[idx])
        # for idx in range(len(df_list)):
        #     plt.plot(x, losses[idx], label='Train loss')#labels[idx])
        #     plt.plot(x, testloss[idx], label='Test loss')  # labels[idx])
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(self.dir_experiment, 'loss.jpg'))
        plt.show()

        plt.figure(2)
        # 对比 psnr
        plt.subplot(211)
        plt.title('PSNR Score')
        plt.plot(x, train_psnrs[0], label='train')  # labels[idx] + '_train')
        plt.plot(x, test_psnrs[0], label='test')  # labels[idx] + '_test')
        # for idx in range(len(df_list)):
        #     # plt.plot(x, train_psnrs[idx], label=labels[idx] + '_train')
        #     # plt.plot(x, test_psnrs[idx], label=labels[idx] + '_test')
        #     plt.plot(x, train_psnrs[idx], label='train')  # labels[idx] + '_train')
        #     plt.plot(x, test_psnrs[idx], label='test')  # labels[idx] + '_test')
        plt.legend()
        plt.ylabel('PSNR')
        plt.xlabel('epoch')
        # plt.show()

        # 对比ssim
        plt.subplot(212)
        plt.title('SSIM Score')
        plt.plot(x, train_ssims[0], label='train')  # labels[idx] + '_train')
        plt.plot(x, test_ssims[0], label='test')  # labels[idx] + '_test')
        # for idx in range(len(df_list)):
        #     # plt.plot(x, train_ssims[idx], label=labels[idx] + '_train')
        #     # plt.plot(x, test_ssims[idx], label=labels[idx] + '_test')
        #     plt.plot(x, train_ssims[idx], label='train')  # labels[idx] + '_train')
        #     plt.plot(x, test_ssims[idx], label='test')  # labels[idx] + '_test')
        plt.legend()
        plt.ylabel('SSIM')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(self.dir_experiment, 'PNSR&SSIM.jpg'))
        plt.show()

    def fusion_image(self, path1, epoch1, path2, epoch2):

        filename_model1 = os.path.join(path1, 'epoch_{}.pth'.format(epoch1))
        filename_model2 = os.path.join(path2, 'epoch_{}.pth'.format(epoch2))

        mode1s_1 = torch.load(filename_model1)
        mode1s_2 = torch.load(filename_model2)

        g1 = Generator(self.nb_channels_first_layer, self.dim)
        g2 = Generator(self.nb_channels_first_layer, self.dim)
        g1.cuda()
        g2.cuda()
        g1.load_state_dict(mode1s_1)
        g2.load_state_dict(mode1s_2)
        g1.eval()
        g2.eval()

        criterion = torch.nn.MSELoss()

        def _compute_error(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            dataloader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

            error = 0
            error1 = 0
            error2 = 0
            for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                z = Variable(current_batch['z']).type(torch.FloatTensor).cuda()
                g1_z = g1.forward(z)
                g2_z = g2.forward(z)

                error1 += criterion(g1_z, x).data.cpu().numpy()
                error2 += criterion(g2_z, x).data.cpu().numpy()
                error += criterion((g1_z + g2_z) / 2.0, x).data.cpu().numpy()

            error1 /= len(dataloader)
            error2 /= len(dataloader)
            error /= len(dataloader)
            psnr1 = 10 * np.log10(255 * 255 / error1) / 3.0
            psnr2 = 10 * np.log10(255 * 255 / error2) / 3.0
            psnr = 10 * np.log10(255 * 255 / error) / 3.0
            print('MSE1 and PSNR1 for {}: {},{}'.format(train_test, error1, psnr1))
            print('MSE2 and PSNR2 for {}: {},{}'.format(train_test, error2, psnr2))
            print('MSE_f and PSNR_f for {}: {},{}'.format(train_test, error, psnr))

        _compute_error(self.dir_z_train, self.dir_x_train, 'train')


def create_path(nb_samples):
    z0 = 5 * np.random.randn(1, 32 * 32)
    z1 = 5 * np.random.randn(1, 32 * 32)

    # z0 = np.zeros((1, 32 * 32))
    # z1 = np.zeros((1, 32 * 32))

    # z0[0, 0] = -20
    # z1[0, 0] = 20

    batch_z = np.copy(z0)

    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    return batch_z
