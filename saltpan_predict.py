import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
#from saltpan_UNet_train import U_Net
from PanNet_dataset import PlanetScope
from PanNet_dataset import PlanetMask
from PanNet_dataset import plot_batch
import torchgeo
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.datasets.utils import stack_samples
import matplotlib.pyplot as plt
from PIL import Image

class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3) #check out these numbers and the needed modification
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1)) #concatenate the prior block and horizontal block
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1)) #these are skip connections

        return upconv1

    def contract_block(self,in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    # now create expand blocks (broaden channels back out)
    def expand_block(self,in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand

modpath = './saved_models/run2opt.pth'
UNet = U_Net(in_channels=3,out_channels=2)

UNet.load_state_dict(torch.load(modpath))

train_data = './train_scenes'
img = PlanetScope(train_data)
train_truth = './train_truth'
truth = PlanetMask(train_truth)
img = img & truth

mint, maxt = 1642191395.0, 1642191395.999999
roi = torchgeo.datasets.BoundingBox(minx = 447139.7152,
                                    maxx =509083.5446,
                                    miny = 3464718.6179,
                                    maxy = 3499982.3511,
                                    mint = mint,
                                    maxt = maxt)

sampler = RandomBatchGeoSampler(img,size = 112*3,length=12,batch_size=3,roi = roi)
dataloader = DataLoader(img, batch_sampler=sampler,collate_fn=stack_samples)
import numpy as np
with torch.no_grad():
    UNet.eval()

    for batch in dataloader:
        x = batch["image"]
        x = x[:, 0:3, :, :]
        predmap = UNet(x)
        predmap1 = predmap.argmax(dim=1)
        print(np.unique(predmap1[0]))
        #tot_batch = batch + predmap
        _,axs = plt.subplots(nrows=3,ncols=3)
        #plot_batch(batch,nrows = 3)
        print(axs)
        print(np.unique(batch['mask']))
        batch['mask'][batch['mask'] != 0] = 1
        print(np.unique(batch['mask']))
        b1 = batch["image"][0].permute(1, 2, 0).numpy().astype(int)
        b2 = batch["image"][1].permute(1, 2, 0).numpy().astype(int)
        b3 = batch["image"][2].permute(1, 2, 0).numpy().astype(int)
        t1 = batch["mask"][0].permute(1, 2, 0).numpy().astype(int)
        t2 = batch["mask"][1].permute(1, 2, 0).numpy().astype(int)
        t3 = batch["mask"][2].permute(1, 2, 0).numpy().astype(int)
        axs[0][0].imshow(b1)
        axs[0][1].imshow(b2)
        axs[0][2].imshow(b3)
        axs[1][0].imshow(t1)
        axs[1][1].imshow(t2)
        axs[1][2].imshow(t3)
        axs[2][0].imshow(predmap1[0])
        axs[2][1].imshow(predmap1[1])
        axs[2][2].imshow(predmap1[2])
        # plt.subplots(predmap1[0])
        # plt.subplots(predmap1[1])
        # plt.subplots(predmap1[2])
        plt.show()

