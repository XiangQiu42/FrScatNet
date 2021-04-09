#coding=utf-8
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from utils import create_folder, normalize,create_name_experiment
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from PIL import Image


HOME= os.environ['HOME']
net_config = {
    'last_activate': 'tanh',
    'train_x':'(-1,1)',
    'first_channels': 50,
}
train_config = {
    #常改参数/home/jains/datasets/huang_zhong_ren/
    'dataset_root':HOME+'/datasets/cfar-10/',#训练集根目录
    'dataset':'scat',#训练集
    'cuda':True,#无GPU时需要更改
    'CUDA_VISIBLE_DEVICES':1,
    'batch_size':50,
    'num_workers': 4,  # win下设置为0

    'model_name':'_'+net_config['last_activate']+net_config['train_x']+'_'+str(net_config['first_channels'])+
                 '_3channels'+'_in(-1,1)'+'m2-24',#备注存储模型的文件夹

    'train_attribute': '65536',
    'test_attribute': '2048_after_65536',
    'embedding_attribute': 'ScatJ3',
    'exper_path': './experiments',
}

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

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)
def weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

from torch.utils.data import Dataset
from utils import get_nb_files

from skimage.measure import compare_ssim, compare_psnr, compare_mse
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return compare_ssim(x,y)


class EmbeddingsImagesDataset(Dataset):
    def __init__(self, dir_z, dir_x, nb_channels=3):
        assert get_nb_files(dir_z) == get_nb_files(dir_x),"data infomation: {}".format(dir_z)
        assert nb_channels in [1, 3]
        self.firstchannel = net_config['first_channels']
        self.nb_files = get_nb_files(dir_z)
        self.nb_channels = nb_channels

        self.dir_z = dir_z
        self.dir_x = dir_x
        self.filename = os.listdir(dir_x)

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        # filename = os.path.join(self.dir_z, '{}.npy'.format(idx))
        filename = os.path.join(self.dir_z,self.filename[idx].split('.')[0])+'.npy'
        #z = scio.loadmat(filename)['outdata']  # 读取.mat文件
        z = np.load(filename)
        z = z.sum(axis=0)
        z = z[0:50].reshape(50,z.shape[1],z.shape[2])
        # z = z.reshape((self.firstchannel,z.shape[2],z.shape[3]))#直接加载scat数据

        filename = os.path.join(self.dir_x,self.filename[idx].split('.')[0])+'.jpg'
        if self.nb_channels == 3:
            if(net_config['train_x']=='(-1,1)'):
                x = (np.ascontiguousarray(Image.open(filename), dtype=np.uint8).transpose((2, 0, 1)) / 127.5) - 1.0
            elif(net_config['train_x']=='(0,1)'):
                x = (np.ascontiguousarray(Image.open(filename), dtype=np.uint8).transpose((2, 0, 1)) / 255)
        else:
            x = np.expand_dims(np.ascontiguousarray(Image.open(filename), dtype=np.uint8), axis=-1)
            if (net_config['train_x'] == '(-1,1)'):
                x = (x.transpose((2, 0, 1)) / 127.5) - 1.0
            elif (net_config['train_x'] == '(0,1)'):
                x = (x.transpose((2, 0, 1)) / 255)

        sample = {'z': z, 'x': x}
        return sample

class Generator(nn.Module):
    def __init__(self, first_channels,size_first_layer=2):
        super(Generator, self).__init__()

        self.ConvBlock0 = nn.Sequential(
            nn.BatchNorm2d(first_channels,eps=0.001, momentum=0.9),
        )

        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(first_channels, 450, 3, bias=False),
            nn.BatchNorm2d(450, eps=0.001, momentum=0.9),
            nn.Conv2d(450, 300, 3, bias=False),
            nn.BatchNorm2d(300, eps=0.001, momentum=0.9),

        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(300,214, 3, bias=False),
            nn.BatchNorm2d(214, eps=0.001, momentum=0.9),
            nn.Conv2d(214, 128, 3, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9),

        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(128,96, 3, bias=False),
            nn.BatchNorm2d(96, eps=0.001, momentum=0.9),
            nn.Conv2d(96, 64, 3, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9),

        )

        # self.ConvBlock4 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(64, 32, 3, bias=False),
        #     nn.BatchNorm2d(32, eps=0.001, momentum=0.9),
        #
        # )
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9),
            nn.Conv2d(32, 3, 3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),

        )


    def forward(self, input_tensor):

        x0 = self.ConvBlock0(input_tensor)
        x0 = F.relu(x0,inplace=True)

        x1  = self.ConvBlock1(x0)
        x1 = F.relu(x1)

        x2 = self.ConvBlock2(x1)
        x2 = F.relu(x2)

        x3 = self.ConvBlock3(x2)
        x3 = F.relu(x3)

        # x4 = self.ConvBlock4(x3)
        # x4 = F.relu(x4)

        output = self.out_layer(x3)
        if(net_config['last_activate']=='tanh'):
            output = F.tanh(output)
        elif(net_config['last_activate'] == 'sigmoid'):
            output = F.sigmoid(output)
        else:
            print("最后一激活层设置错误...请检查")
            exit(0)
        return output


class Generator1(nn.Module):
    def __init__(self, first_channels,size_first_layer=2):
        super(Generator1, self).__init__()

        # self.ConvBlock_1 = nn.Sequential(
        #     nn.Conv2d(first_channels,256,3,bias=False),
        #     nn.BatchNorm2d(256,eps=0.001, momentum=0.9),
        # )
        self.ConvBlock0 = nn.Sequential(
            nn.BatchNorm2d(first_channels, eps=0.001, momentum=0.9),
            nn.Conv2d(first_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.9),
        )

        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(256, 192, 3, bias=False),
            nn.BatchNorm2d(192, eps=0.001, momentum=0.9),
            nn.Conv2d(192, 128, 3, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9),

        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(128,96, 3, bias=False),
            nn.BatchNorm2d(96, eps=0.001, momentum=0.9),
            nn.Conv2d(96, 64, 3, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9),

        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(64,48, 3, bias=False),
            nn.BatchNorm2d(48, eps=0.001, momentum=0.9),
            nn.Conv2d(48, 32, 3, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9),

        )

        #self.ConvBlock4 = nn.Sequential(
          #  nn.Upsample(scale_factor=2, mode='bilinear'),
          #  nn.ReflectionPad2d(1),
         #   nn.Conv2d(32, 16, 3, bias=False),
         #   nn.BatchNorm2d(16, eps=0.001, momentum=0.9),
        #)
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 24, 3, bias=False),
            nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
            nn.Conv2d(24, 3, 3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),

        )


    def forward(self, input_tensor):

        # x_1 = self.ConvBlock_1(input_tensor)
        # x_1 = F.relu(x_1)

        x0 = self.ConvBlock0(input_tensor)
        x0 = F.relu(x0)

        x1  = self.ConvBlock1(x0)
        x1 = F.relu(x1)

        x2 = self.ConvBlock2(x1)
        x2 = F.relu(x2)

        x3 = self.ConvBlock3(x2)
        x3 = F.relu(x3)

        # x4 = self.ConvBlock4(x3)
        # x4 = F.relu(x4)

        output = self.out_layer(x3)
        if(net_config['last_activate']=='tanh'):
            output = F.tanh(output)
        elif(net_config['last_activate'] == 'sigmoid'):
            output = F.sigmoid(output)
        else:
            print("最后一激活层设置错误...请检查")
            exit(0)
        return output

class Generator2(nn.Module):
    def __init__(self, first_channels,size_first_layer=2):
        super(Generator2, self).__init__()
        self.ConvBlock_5 = nn.Sequential(
            nn.Conv2d(3, 256, 7, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.9)
        )
        self.ConvBlock_4 = nn.Sequential(
            nn.Conv2d(256, 192, 7, bias=False),
            nn.BatchNorm2d(192, eps=0.001, momentum=0.9)
        )
        self.ConvBlock_3 = nn.Sequential(
            nn.Conv2d(192, 128, 7, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9)
        )
        self.ConvBlock_2 = nn.Sequential(
            nn.Conv2d(128, 64, 7, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        )
        self.ConvBlock_1 = nn.Sequential(
            nn.Conv2d(64, 48, 3, bias=False),
            nn.BatchNorm2d(48, eps=0.001, momentum=0.9)
        )
        self.ConvBlock_0 = nn.Sequential(
            nn.Conv2d(48,25,3,bias=False),
            nn.BatchNorm2d(25,eps=0.001,momentum=0.9)
        )


        self.ConvBlock0 = nn.Sequential(
            # nn.BatchNorm2d(25, eps=0.001, momentum=0.9),
            nn.Conv2d(25, 256, 1, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.9),
        )

        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(256, 192, 3, bias=False),
            nn.BatchNorm2d(192, eps=0.001, momentum=0.9),
            nn.Conv2d(192, 128, 3, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9),

        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(128,96, 3, bias=False),
            nn.BatchNorm2d(96, eps=0.001, momentum=0.9),
            nn.Conv2d(96, 64, 3, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9),

        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(64,48, 3, bias=False),
            nn.BatchNorm2d(48, eps=0.001, momentum=0.9),
            nn.Conv2d(48, 32, 3, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9),
            # nn.PixelShuffle(2)

        )

        #self.ConvBlock4 = nn.Sequential(
          #  nn.Upsample(scale_factor=2, mode='bilinear'),
          #  nn.ReflectionPad2d(1),
         #   nn.Conv2d(32, 16, 3, bias=False),
         #   nn.BatchNorm2d(16, eps=0.001, momentum=0.9),
        #)
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 24, 3, bias=False),
            nn.Conv2d(24, 3, 3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),

        )


    def forward(self, input_tensor):

        # x_1 = self.ConvBlock_1(input_tensor)
        # x_1 = F.relu(x_1)
        x_5 = F.relu(self.ConvBlock_5(input_tensor))
        x_4 = F.relu(self.ConvBlock_4(x_5))
        x_3 = F.relu(self.ConvBlock_3(x_4))
        x_2 = F.relu(self.ConvBlock_2(x_3))
        x_1 = F.relu(self.ConvBlock_1(x_2))
        x_0 = F.relu(self.ConvBlock_0(x_1))

        x0 = self.ConvBlock0(x_0)
        x0 = F.relu(x0)

        x1  = self.ConvBlock1(x0)
        x1 = F.relu(x1)

        x2 = self.ConvBlock2(x1)
        x2 = F.relu(x2)

        x3 = self.ConvBlock3(x2)
        x3 = F.relu(x3)

        # x4 = self.ConvBlock4(x3)
        # x4 = F.relu(x4)

        output = self.out_layer(x3)
        if(net_config['last_activate']=='tanh'):
            output = F.tanh(output)
        elif(net_config['last_activate'] == 'sigmoid'):
            output = F.sigmoid(output)
        else:
            print("最后一激活层设置错误...请检查")
            exit(0)
        return output

class GSN:
    def __init__(self):
        dir_datasets = train_config['dataset_root']
        dir_experiments = os.path.expanduser('./experiments')
        dataset = train_config['dataset']
        train_attribute = train_config['train_attribute']
        test_attribute = train_config['test_attribute']
        embedding_attribute = train_config['embedding_attribute']
        self.nb_channels_first_layer = net_config['first_channels']

        name_experiment='{}_{}_{}_{}'.format(dataset,
                                                       train_attribute,
                                                     test_attribute,
                                                       embedding_attribute)

        self.dir_x_train = os.path.join(dir_datasets, dataset, '{0}'.format(train_attribute))
        self.dir_x_test = os.path.join(dir_datasets, dataset, '{0}'.format(test_attribute))
        self.dir_z_train = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(train_attribute, embedding_attribute))
        self.dir_z_test = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(test_attribute, embedding_attribute))

        self.dir_experiment = os.path.join(dir_experiments, 'gsn_hf', name_experiment)+train_config['model_name']
        self.dir_models = os.path.join(self.dir_experiment, 'models')
        self.dir_logs = os.path.join(self.dir_experiment, 'logs')
        create_folder(self.dir_models)
        create_folder(self.dir_logs)

        self.batch_size = train_config['batch_size']
        self.nb_epochs_to_save = 1
        self.cuda = train_config['cuda']


        self.last_activate = net_config['last_activate']#最后一层网络激活层
        # self.Generator = Generator
        print("***********activate layer is {} ********".format(self.last_activate))
        print("train data is {} --- test data is {} --- network config is{}".format(self.dir_x_train.split('/')[-1],self.dir_x_test.split('/')[-1],
              train_config['model_name']))


    def train(self, epoch_to_restore=0):
        print("------------train start------------------")

        '''训练开始'''
        g = Generator1(self.nb_channels_first_layer)

        #读取csv文件记录数据
        from utilss.tools import read_run_data,save_run_data
        df = read_run_data(self.dir_experiment+'/')

        if epoch_to_restore > 0:
            filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch_to_restore))
            g.load_state_dict(torch.load(filename_model))
        else:
            g.apply(weights_init)

        if(self.cuda):
            torch.cuda.set_device(train_config['CUDA_VISIBLE_DEVICES'])
            g.cuda()
        g.train()

        dataset = EmbeddingsImagesDataset(self.dir_z_train, self.dir_x_train)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=train_config['num_workers'], pin_memory=True)

        # '''查看网络结构图'''
        #batch = next(iter(dataloader))
        #z = batch['z'].type(torch.FloatTensor).cuda()
        #with SummaryWriter(comment='Net1')as w:
         #    w.add_graph(g, (z,))

        fixed_dataloader = DataLoader(dataset, self.batch_size)  # 用作验证集的数据
        fixed_batch = next(iter(fixed_dataloader))  # 所有值 /127.5 - 1-------------iter获取容器的迭代器,next表示下一个


        testdataset = EmbeddingsImagesDataset(self.dir_z_test,self.dir_x_test)
        testdataloader = DataLoader(dataset=testdataset,batch_size=self.batch_size,shuffle=True,
                                    num_workers=train_config['num_workers'],pin_memory=True)
        testfixed_batch = next(iter(testdataloader))

        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(g.parameters())
        writer = SummaryWriter(self.dir_logs)

        try:
            epoch = epoch_to_restore
            while epoch<350:
                g.train()
                for _ in range(self.nb_epochs_to_save):
                    epoch += 1
                    print("epoch is : ",epoch)
                    for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                        g.zero_grad()
                        if(self.cuda):
                            x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                            z = Variable(current_batch['z']).type(torch.FloatTensor).cuda()
                        else:
                            x = Variable(current_batch['x']).type(torch.FloatTensor)
                            z = Variable(current_batch['z']).type(torch.FloatTensor)

                        g_z = g.forward(z)
                        loss = criterion(g_z, x)
                        loss.backward()
                        optimizer.step()

                    print("----------------{}-------------".format(loss))
                    writer.add_scalar('train_loss', loss, epoch)


                from skimage.measure import compare_ssim, compare_nrmse, compare_psnr, compare_mse
                if(epoch%1 == 0):
                    test_psnr=0
                    test_ssim=0
                    val_psnr = 0
                    val_ssim = 0
                    if (self.cuda):
                        test_z = Variable(testfixed_batch['z']).type(torch.FloatTensor).cuda()
                        val_z = Variable(fixed_batch['z']).type(torch.FloatTensor).cuda()
                    else:
                        test_z = Variable(testfixed_batch['z']).type(torch.FloatTensor)
                        val_z = Variable(fixed_batch['z']).type(torch.FloatTensor)
                    g.eval()
                    gtest_x = g.forward(test_z)
                    if(net_config['train_x']=='(-1,1)'):
                        gtest_x = np.uint8((gtest_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                        test_x = np.uint8(((testfixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)))+1)*127.5)
                    elif(net_config['train_x'] == '(0,1)'):
                        gtest_x = np.uint8((gtest_x.data.cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                        test_x = np.uint8(((testfixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)))) * 255)
                    for ii in range (self.batch_size):
                        test_psnr+=compare_psnr(gtest_x[ii],test_x[ii])
                        test_ssim+=compare_ssim(gtest_x[ii],test_x[ii],multichannel=True)
                    test_ssim/=self.batch_size
                    test_psnr/=self.batch_size


                    gval_x = g.forward(val_z)
                    if (net_config['train_x'] == '(-1,1)'):
                        gval_x = np.uint8((gval_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                        val_x = np.uint8((fixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                    elif (net_config['train_x'] == '(0,1)'):
                        gval_x = np.uint8((gval_x.data.cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                        val_x = np.uint8((fixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                    for ii in range (self.batch_size):
                        val_psnr+=compare_psnr(gval_x[ii],val_x[ii])
                        val_ssim+=compare_ssim(gval_x[ii],val_x[ii],multichannel=True)
                    val_psnr/=self.batch_size
                    val_ssim/=self.batch_size

                    #记录所有指标：loss，psnr和ssim
                    df.loc[epoch] = [epoch,loss.cpu().data.numpy(), val_psnr,test_psnr,val_ssim,test_ssim]
                    save_run_deata(self.dir_experiment+'/',df=df)

                    writer.add_scalars('psnr',  {'val_psnr': val_psnr, 'test_psnr': test_psnr}, epoch)
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
            print('[*] Closing Writer.')
            writer.close()

    def save_originals(self):
        def _save_originals(dir_z, dir_x, train_test):
            dataset = EmbeddingsImagesDataset(dir_z, dir_x)
            fixed_dataloader = DataLoader(dataset, 16)
            fixed_batch = next(iter(fixed_dataloader))

            temp = make_grid(fixed_batch['x'], nrow=4).numpy().transpose((1, 2, 0))

            filename_images = os.path.join(self.dir_experiment, 'originals_{}.png'.format(train_test))
            Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

        _save_originals(self.dir_z_train, self.dir_x_train, 'train')
        _save_originals(self.dir_z_test, self.dir_x_test, 'test')

    def cal_fusion_image():
        model1_dir = ''
        model2_dir = ''

        g = Generator1(self.nb_channels_first_layer)

        # 读取csv文件记录数据
        from utilss.tools import read_run_data, save_run_data
        df = read_run_data(self.dir_experiment + '/')

        if epoch_to_restore > 0:
            filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch_to_restore))
            g.load_state_dict(torch.load(filename_model))
        else:
            g.apply(weights_init)

        if (self.cuda):
            torch.cuda.set_device(train_config['CUDA_VISIBLE_DEVICES'])
            g.cuda()
        test_psnr = 0
        test_ssim = 0
        val_psnr = 0
        val_ssim = 0
        if (self.cuda):
            test_z = Variable(testfixed_batch['z']).type(torch.FloatTensor).cuda()
            val_z = Variable(fixed_batch['z']).type(torch.FloatTensor).cuda()
        else:
            test_z = Variable(testfixed_batch['z']).type(torch.FloatTensor)
            val_z = Variable(fixed_batch['z']).type(torch.FloatTensor)
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

    def fusion_image(self):
        filename_model1 = ''
        filename_model2 = ''
        test_psnr = 0
        test_ssim = 0
        val_psnr = 0
        val_ssim = 0

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
            psnr1 = 10*np.log10(255*255 / error1) / 3.0
            psnr2 = 10 * np.log10(255 * 255 / error2) / 3.0
            psnr = 10 * np.log10(255 * 255 / error) / 3.0
            print('MSE1 and PSNR1 for {}: {},{}'.format(train_test, error1,psnr1))
            print('MSE2 and PSNR2 for {}: {},{}'.format(train_test, error2, psnr2))
            print('MSE_f and PSNR_f for {}: {},{}'.format(train_test, error, psnr))


        _compute_error(self.dir_z_train, self.dir_x_train, 'train')
    def train1(self, epoch_to_restore=0):
        print("------------train start------------------")

        '''训练开始'''
        g = Generator2(self.nb_channels_first_layer)

        #读取csv文件记录数据
        from utilss.tools import read_run_data,save_run_data
        df = read_run_data(self.dir_experiment+'/')

        if epoch_to_restore > 0:
            filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch_to_restore))
            g.load_state_dict(torch.load(filename_model))
        else:
            g.apply(weights_init)

        if(self.cuda):
            torch.cuda.set_device(train_config['CUDA_VISIBLE_DEVICES'])
            g.cuda()
        g.train()

        dataset = EmbeddingsImagesDataset(self.dir_z_train, self.dir_x_train)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=train_config['num_workers'], pin_memory=True)

        # '''查看网络结构图'''
        #batch = next(iter(dataloader))
        #z = batch['z'].type(torch.FloatTensor).cuda()
        #with SummaryWriter(comment='Net1')as w:
         #    w.add_graph(g, (z,))

        fixed_dataloader = DataLoader(dataset, self.batch_size)  # 用作验证集的数据
        fixed_batch = next(iter(fixed_dataloader))  # 所有值 /127.5 - 1-------------iter获取容器的迭代器,next表示下一个


        testdataset = EmbeddingsImagesDataset(self.dir_z_test,self.dir_x_test)
        testdataloader = DataLoader(dataset=testdataset,batch_size=self.batch_size,shuffle=True,
                                    num_workers=train_config['num_workers'],pin_memory=True)
        testfixed_batch = next(iter(testdataloader))

        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(g.parameters())
        writer = SummaryWriter(self.dir_logs)

        try:
            epoch = epoch_to_restore
            while epoch<350:
                g.train()
                for _ in range(self.nb_epochs_to_save):
                    epoch += 1
                    print("epoch is : ",epoch)
                    for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                        g.zero_grad()
                        if(self.cuda):
                            x = Variable(current_batch['x']).type(torch.FloatTensor).cuda()
                        else:
                            x = Variable(current_batch['x']).type(torch.FloatTensor)

                        g_z = g.forward(x)
                        loss = criterion(g_z, x)
                        loss.backward()
                        optimizer.step()

                    print("----------------{}-------------".format(loss))
                    writer.add_scalar('train_loss', loss, epoch)


                from skimage.measure import compare_ssim, compare_nrmse, compare_psnr, compare_mse
                if(epoch%1 == 0):
                    test_psnr=0
                    test_ssim=0
                    val_psnr = 0
                    val_ssim = 0
                    if (self.cuda):
                        test_z = Variable(testfixed_batch['x']).type(torch.FloatTensor).cuda()
                        val_z = Variable(fixed_batch['x']).type(torch.FloatTensor).cuda()
                    else:
                        test_z = Variable(testfixed_batch['x']).type(torch.FloatTensor)
                        val_z = Variable(fixed_batch['x']).type(torch.FloatTensor)
                    g.eval()
                    gtest_x = g.forward(test_z)
                    if(net_config['train_x']=='(-1,1)'):
                        gtest_x = np.uint8((gtest_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                        test_x = np.uint8(((testfixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)))+1)*127.5)
                    elif(net_config['train_x'] == '(0,1)'):
                        gtest_x = np.uint8((gtest_x.data.cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                        test_x = np.uint8(((testfixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)))) * 255)
                    for ii in range (self.batch_size):
                        test_psnr+=compare_psnr(gtest_x[ii],test_x[ii])
                        test_ssim+=compare_ssim(gtest_x[ii],test_x[ii],multichannel=True)
                    test_ssim/=self.batch_size
                    test_psnr/=self.batch_size
#

                    gval_x = g.forward(val_z)
                    if (net_config['train_x'] == '(-1,1)'):
                        gval_x = np.uint8((gval_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                        val_x = np.uint8((fixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
                    elif (net_config['train_x'] == '(0,1)'):
                        gval_x = np.uint8((gval_x.data.cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                        val_x = np.uint8((fixed_batch['x'].cpu().numpy().transpose((0, 2, 3, 1))) * 255)
                    for ii in range (self.batch_size):
                        val_psnr+=compare_psnr(gval_x[ii],val_x[ii])
                        val_ssim+=compare_ssim(gval_x[ii],val_x[ii],multichannel=True)
                    val_psnr/=self.batch_size
                    val_ssim/=self.batch_size

                    #记录所有指标：loss，psnr和ssim
                    df.loc[epoch] = [epoch,loss.cpu().data.numpy(), val_psnr,test_psnr,val_ssim,test_ssim]
                    save_run_data(self.dir_experiment+'/',df=df)

                    writer.add_scalars('psnr',  {'val_psnr': val_psnr, 'test_psnr': test_psnr}, epoch)
                    writer.add_scalars('ssim', {'val_ssim': val_ssim, 'test_psnr': test_ssim}, epoch)


                filename = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
                torch.save(g.state_dict(), filename)

        finally:
            print('[*] Closing Writer.')
            writer.close()

    def save_img(self,model_dir):
        from skimage.measure import compare_ssim, compare_nrmse, compare_psnr, compare_mse
        cuda = True
        test_psnr = 0
        test_ssim = 0
        val_psnr = 0
        val_ssim = 0

        g = Generator1(self.nb_channels_first_layer)
        g.load_state_dict(torch.load(model_dir))
        if(cuda):
            g.cuda()
        g.eval()

        train_dataset = EmbeddingsImagesDataset(self.dir_z_train, self.dir_x_train)
        train_dataloader = DataLoader(train_dataset, batch_size=64)  # 用作验证集的数据
        train_batch = next(iter(train_dataloader))  # 所有值 /127.5 - 1-------------iter获取容器的迭代器,next表示下一个

        test_dataset = EmbeddingsImagesDataset(self.dir_z_test, self.dir_x_test)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)
        test_batch = next(iter(test_dataloader))
        if (cuda):
            test_z = Variable(test_batch['z']).type(torch.FloatTensor).cuda()
            train_z = Variable(train_batch['z']).type(torch.FloatTensor).cuda()
        else:
            test_z = Variable(test_batch['z']).type(torch.FloatTensor)
            train_z = Variable(train_batch['z']).type(torch.FloatTensor).cuda()

        gtest_x = g.forward(test_z)
        gtest_x = np.uint8((gtest_x.data.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127.5)
        test_x = np.uint8(((test_batch['x'].cpu().numpy().transpose((0, 2, 3, 1))) + 1) * 127.5)

        for ii in range(64):
            filename = ''+str(ii)+'.jpg'
            Image.fromarray(test_x).save(filename)
        for ii in range(64):
            filename = ''+str(ii)+'.jpg'
            Image.fromarray(gtest_x).save(filename)




GSN().train()