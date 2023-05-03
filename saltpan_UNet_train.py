import torch
from torch import nn
import numpy as np
import torchvision
import torchgeo
from torchgeo.datasets import RasterDataset, unbind_samples,stack_samples
from torchgeo.samplers import RandomBatchGeoSampler
from torch.utils.data import DataLoader
from torchgeo.samplers import Units
from PanNet_dataset import PlanetScope, PlanetMask, ElevationData



#Define the root dirs for scens and masks

train_data = './train_scenes'
elev_data = './elevation_data'
train_truth = './train_truth'

#create geodatasets from
data = PlanetScope(train_data)
elev = ElevationData(elev_data)
print(elev.crs)
print(elev)
truth = PlanetMask(train_truth)
print(truth.all_bands[0])

trainDS = data & elev
trainDS = trainDS & truth

dir(trainDS)
#PUT ROI INTO sampler
mint, maxt = 1642191395.0, 1642191395.999999 #need to figure this for time series !!!
#roi needs to be in ... pixels?
#roi here is based on 35half from GEOAI_proj
roi = torchgeo.datasets.BoundingBox(minx = 447139.7152,
                                    maxx =509083.5446,
                                    miny = 3464718.6179,
                                    maxy = 3499982.3511,
                                    mint = mint,
                                    maxt = maxt)
#split the roi into two for train/val sets with 70/30 split

#define some useful params
batch_size = 30
length = 150
batches = length/batch_size

trainROI,valROI = roi.split(proportion = 0.7,horizontal = False)
print("trainroi: {} valroi: {}".format(trainROI,valROI))
#create train sampler/loiader
##TROUBLESHOOTING WITH SMALL WINDOW
train_sampler = RandomBatchGeoSampler(trainDS, size=512 * 3, length=length,units=Units.CRS,roi = trainROI,batch_size = batch_size) #
train_dataloader = DataLoader(trainDS, batch_sampler=train_sampler, collate_fn=stack_samples) #
#create validation sampler/loader
val_sampler = RandomBatchGeoSampler(trainDS, size=512 * 3, length=length,units=Units.CRS,roi = valROI,batch_size = batch_size) #, length=120
val_dataloader = DataLoader(trainDS, batch_sampler=val_sampler, collate_fn=stack_samples)


#Define the custom class for a U Net architecture (based on Medium article "Creating a very simple U Net model with pytorch for semantic segmentation of satellite images)
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



#begin training!
from saltpan_trainer import train
import optuna
from optuna.trial import TrialState
from torchvision.models.segmentation import deeplabv3_resnet50



#TRYING TO INCREASE PRECISION, INCREASED BG WEIGHTS
class_weights = torch.FloatTensor([0.07,1.0]).cuda() # weights selected based on pixel imbalance between background and classes

#instantiate model
UNet = U_Net(in_channels=5,out_channels=2)
ResNet = deeplabv3_resnet50(weights = None,num_classes = 2)
modname = 'UNet'


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

#use optuna to find best LR

lr = 0.005256269118924815


loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(UNet.parameters(),lr = lr)        #torch.optim.Adam(UNet.parameters(), lr=0.01)
train_prec, val_prec, train_rec, val_rec, train_F1, val_F1, model = train(UNet, train_dataloader, val_dataloader,
                                                                              loss_fn, optimizer, acc_metric,
                                                                              epochs=100, batches=batches, modname= modname )

print("Training Precision: {}".format(train_prec))
print("Validation Precision: {}".format(val_prec))
print("Training Recall: {}".format(train_rec))
print("Validation Recall: {}".format(val_rec))
print("Train F1 (AVERAGE): {}".format(train_F1))
print("Validation F1 (AVERAGE): {}".format(val_F1))







#it's running! need to evaluate the output

##In Progress:
#implement tensorboard (X)
#create script for training function (X)
#create Git repo (maybe once cleaned & working?)  (X)
#use optuna to find best LR (X)
#implement LR scheduler/optimizer  ()
#consider LR burn in/warm up ()
#revisit elevation normalization ()
    #find absolute min/maxes and automate ()
    #plot band histograms and evaluate norm method ()
#bring in ndvi ()