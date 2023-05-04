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



#Define the custom class for a U Net architecture (based on Medium article "Creating a very simple U Net model with pytorch for semantic segmentation of satellite images)

train_data = './train_scenes'
elev_data = './elevation_data'
train_truth = './train_truth'

data = PlanetScope(train_data)
elev = ElevationData(elev_data)
truth = PlanetMask(train_truth)

trainDS = data & elev
trainDS = trainDS & truth


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

trainROI,valROI = roi.split(proportion = 0.7,horizontal = False)
print("trainroi: {} valroi: {}".format(trainROI,valROI))




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
import pandas
import math



#TRYING DIVERGENCE - UPSAMPLING PAN, DOWNSAMPLING BG


#instantiate model
UNet = U_Net(in_channels=5,out_channels=2)
ResNet = deeplabv3_resnet50(weights = None,num_classes = 2)
modname = 'UNet'


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

#use optuna to find best LR
def objective(trial):
    #optimize for learning rate and optimizer type
    lr = trial.suggest_loguniform("lr",1e-5,0.1)
    optim_type = trial.suggest_categorical("optimizer",["Adam","SGD","RMSprop"])

    #optimize for batch size and class weights
    batch_size = trial.suggest_int("batch_size",3,30)
    class_weightsBG = trial.suggest_float("cw_bg",0.0001,.01)
    class_weightsSP = trial.suggest_float("cw_sp",1.0,2.0,log = True)
    class_weights = torch.FloatTensor([class_weightsBG,class_weightsSP]).cuda()


    # create train sampler/loader
    train_sampler = RandomBatchGeoSampler(trainDS, size=512 * 3, length=120, units=Units.CRS, roi=trainROI,
                                          batch_size=batch_size)  #

    # static params
    length = 120
    batches = train_sampler.__len__()#get precise batchnumber this way!
    print(batches)

    train_dataloader = DataLoader(trainDS, batch_sampler=train_sampler, collate_fn=stack_samples)  #
    # create validation sampler/loader
    val_sampler = RandomBatchGeoSampler(trainDS, size=512 * 3, length=120, units=Units.CRS, roi=valROI, batch_size=batch_size)  #
    val_dataloader = DataLoader(trainDS, batch_sampler=val_sampler, collate_fn=stack_samples)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = getattr(torch.optim, optim_type)(UNet.parameters(),lr = lr)        #torch.optim.Adam(UNet.parameters(), lr=0.01)
    train_prec, val_prec, train_rec, val_rec, train_F1, val_F1, model = train(UNet, train_dataloader, val_dataloader,
                                                                              loss_fn, optimizer, acc_metric,
                                                                              epochs=70, batches=batches, modname= modname,batch_size=batch_size)

    print("Training Precision: {}".format(train_prec))
    print("Validation Precision: {}".format(val_prec))
    print("Training Recall: {}".format(train_rec))
    print("Validation Recall: {}".format(val_rec))
    print("Train F1 (AVERAGE): {}".format(train_F1))
    print("Validation F1 (AVERAGE): {}".format(val_F1))

    return train_F1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study Stats:")
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Number of pruned trials: {}".format(len(pruned_trials)))
    print("Number of completed trials: {}".format(len(complete_trials)))

    print("BEST TRIAL:")
    btrial = study.best_trial
    print("Value: {}".format(btrial.value))
    print("Params:")
    for key, value in btrial.params.items():
        print("    {} : {}".format(key,value))

    btrials = study.best_trials
    print("best trials: \n",btrials)

    #save all trials
    df = study.trials_dataframe()
    df.to_csv('optuna_trials.csv')

#it's running! need to evaluate the output

##In Progress:
#implement tensorboard (X)
#create script for training function (X)
#create Git repo (maybe once cleaned & working?)  ()
#use optuna to find best LR
#implement LR scheduler/optimizer  ()