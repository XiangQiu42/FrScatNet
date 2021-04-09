# -- coding: UTF-8 --

import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')
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

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 32
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10('./data', transform=img_transform,download=False)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1).cuda()
        img = Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    psnr = cal_PSNR_SSIM(model,img,)
    print('epoch [{}/{}], loss:{:.4f}, PSNR:{:.4f}, SSIM:{.4f}'
          .format(epoch + 1, num_epochs, loss.data.cpu().numpy()),)
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')