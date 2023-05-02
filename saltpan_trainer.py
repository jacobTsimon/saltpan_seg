import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from torchmetrics import CohenKappa
from torchvision import transforms

#introduce colors for legibility
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



#normalizing for resnet
preprocess = transforms.Compose([

    transforms.Normalize(mean=[28.2, 27.83, 40.16,2.578], std=[26.88, 23.38, 30.44,3.052]),
])
#write put image maxes/mins from QGIS

#RGB
# mins = [0,0,0,-1.8980236]
# maxs = [255,234,228,29.606974]
#4 BAND ANALYTIC
mins = [0,0,0,0,-1.8980236]
maxs = [9027,10810,12319,11044,29.606974]

def F1_score(precision,recall):
    f1 = 2 * (precision * recall)/(precision + recall)
    return f1

def Z1Norm(input,bands):
    t_n = torch.zeros([30,bands,512,512])
    for b in range(bands):
        # imax, imin = input[:,b,:,:].max(), input[:,b,:,:].min()
        # nmin, nmax = 0, 1
        imax,imin = maxs[b],mins[b]
        plus = input[:,b,:,:] #+ abs(imin)

        t_n[:,b,:,:] = (plus - imin) / (imax - imin)  # * (nmax - nmin) + nmin

    return t_n

#test minmax functions from torchgeo docs: https://torchgeo.readthedocs.io/en/latest/tutorials/transforms.html
# import kornia as K
# class MinMaxNormalize(K.IntensityAugmentationBase2D):
#     """Normalize channels to the range [0, 1] using min/max values."""
#
#     def __init__(self, mins: Tensor, maxs: Tensor) -> None:
#         super().__init__(p=1)
#         self.flags = {"mins": mins.view(1, -1, 1, 1), "maxs": maxs.view(1, -1, 1, 1)}
#
#     def apply_transform(
#         self,
#         input: Tensor,
#         params: Dict[str, Tensor],
#         flags: Dict[str, int],
#         transform: Optional[Tensor] = None,
#     ) -> Tensor:
#         return (input - flags["mins"]) / (flags["maxs"] - flags["mins"] + 1e-10)

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1,batches=1,modname = None):
    start = time.time()
    model.cuda()

    kappa_fn = CohenKappa(task='binary').to('cuda')
    train_precision, valid_precision = [], []
    train_recall, valid_recall = [], []
    train_F1, valid_F1 = [], []

    best_F1 = 0.0
    num_skipped = 0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_dl
                Llogdir = 'TBruns/run9opt_train_Loss' #%%
                Plogdir = 'TBruns/run9opt_train_Precision'
                Rlogdir = 'TBruns/run9opt_train_Recall'
                Flogdir = 'TBruns/run9opt_train_F1'
                Slogdir = 'TBruns/run9opt_train_NumSkipped'
                Lwriter = SummaryWriter(log_dir=Llogdir)
                Pwriter = SummaryWriter(log_dir=Plogdir)
                Rwriter = SummaryWriter(log_dir=Rlogdir)
                Fwriter = SummaryWriter(log_dir=Flogdir)
                Swriter = SummaryWriter(log_dir=Slogdir)
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl
                Llogdir = 'TBruns/run9opt_val_Loss' #%%
                Plogdir = 'TBruns/run9opt_val_Precision'
                Rlogdir = 'TBruns/run9opt_val_Recall'
                Flogdir = 'TBruns/run9opt_val_F1'
                Slogdir = 'TBruns/run9opt_val_NumSkipped'
                Lwriter = SummaryWriter(log_dir=Llogdir)
                Pwriter = SummaryWriter(log_dir=Plogdir)
                Rwriter = SummaryWriter(log_dir=Rlogdir)
                Fwriter = SummaryWriter(log_dir=Flogdir)
                Swriter = SummaryWriter(log_dir=Slogdir)

            running_loss = 0.0
            running_acc = 0.0
            running_kappa = 0.0
            running_P = 0.0
            running_R = 0.0
            running_F1 = 0.0

            step = 0

            # iterate over data
            for batch in dataloader:
                x = batch["image"].cuda()

                y = batch["mask"]
                #yu = np.unique(y.numpy())
                #print("before: {}".format(yu))
                y[y != 0] = 1
                #yu = np.unique(y.numpy())
                #print("after: {}".format(yu))
                #
                # y = torch.argmax(y, dim=1)

                ##TRY RERUNNING WITHOUT THIS BLOCK
                if y.sum() == 0:
                    print(bcolors.BOLD + "Step {} empty - skipped".format(step))
                    print(bcolors.HEADER)
                    num_skipped += 1
                    continue
                #try to screen for nans in elevation data
                nanchek = torch.any(x.isnan(),dim = 1)

                if nanchek.long().sum() != 0:
                    print(bcolors.FAIL + "NaN in elevation @ step {} - skipped".format(step))
                    print(bcolors.HEADER)
                    num_skipped += 1
                    continue

                y = y[:,0,:,:] # convert the 4D tensors to 3D (Batch size, X, Y)
                #print(y.sum())

                y = y.cuda()
                #for comparison to ResNet, use only visible bands
                #print(np.unique(x.cpu().detach()))

                #x = x[:, 0:3, :, :]

                #need to do this to get rid of weird Vis product band - RGB ONLY
                # ind = torch.tensor([0,1,2,4])
                # x = torch.index_select(x,1,ind.cuda())

                #x= preprocess(x)
                print("PreNorm shape: {}".format(x.shape))
                x = Z1Norm(x,5).cuda()
                print("PostNorm shape: {}".format(x.shape))

                #print(np.unique(x[:,0,:,:].cpu().detach()))

                #need to ditch 4th band for ResNet
                if modname == 'ResNet':
                    x = x[:,0:3,:,:]

                    x = preprocess(x)

                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)

                    loss = loss_fn(outputs, y) #['out']

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward() #VALIDATE ABOVE CLAIM ? ^
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long()) #['out']

                # stats - whatever is the phase
                #acc = acc_fn(outputs['out'], y)
                #kappa = kappa_fn(preds = outputs['out'], target = y.to('cuda')) #.argmax(dim= 1) FOR UNET ADD BACK IN
                print("outputs shape:",outputs.shape,"outputs unique:",np.unique(outputs[0,:,:,:].detach().cpu()))
                a = torch.reshape(outputs.argmax(dim=1).detach().cpu(),(-1,)) #.argmax(dim=1)
                print("a shape:",a.shape,"a unique:", np.unique(a))
                assert not torch.isnan(a).any()
                assert not torch.isinf(a).any()
                print("y shape", y.shape, "y0 unique:",np.unique(y[0,:,:].detach().cpu()))
                b = torch.reshape(y.cpu(),(-1,))
                print("b shape", b.shape, "b unique:",np.unique(b))

                precision = precision_score(y_pred = a,y_true = b)
                recall = recall_score(y_pred = a,y_true = b)
                f1 = F1_score(precision,recall)
                skf1 = f1_score(y_true = b,y_pred = a)
                #skf12 = f1_score(y.detach().cpu(),outputs.detach().cpu())

                #assert f1 == skf1, print("customF1: {} | skF1: {}".format(f1,skf1))
                #assert skf1 == skf12
                #print(outputs.argmax(dim=1) == y.cuda())
                # cm = confusion_matrix(y.cpu(),outputs.cpu().argmax(dim= 1))
                # print(cm)

               # running_acc += acc #* batch_size
                #running_kappa += kappa
                running_loss += loss #* batch_size
                running_P += precision
                running_R += recall
                running_F1 += skf1
                print(step)

                if step % 2 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}   Precision: {} Recall: {} F1: {} AllocMem (Mb): {}'.format(step, loss, precision,recall,f1,torch.cuda.memory_allocated() / 1024 / 1024))


            epoch_loss = running_loss / batches #len(dataloader.dataset)
            #epoch_acc = running_acc / batches #len(dataloader.dataset)
            #epoch_kappa = running_kappa/ batches
            epoch_prec = running_P/batches
            epoch_rec = running_R/batches
            epoch_F1 = running_F1/batches

            Lwriter.add_scalar("Loss/{}".format(phase),epoch_loss,epoch)
            Pwriter.add_scalar("Precision/{}".format(phase),epoch_prec,epoch)
            Rwriter.add_scalar("Recall/{}".format(phase),epoch_rec,epoch)
            Fwriter.add_scalar("F1/{}".format(phase),epoch_F1,epoch)
            Swriter.add_scalar("NumSkipped/{}".format(phase),num_skipped,epoch)

            num_skipped = 0

            print(bcolors.OKBLUE + '{} epoch {} | epoch Loss: {:.4f} |   epoch Precision: {} | epoch Recall: {} | epoch F1: {}'.format(phase,epoch, epoch_loss, epoch_prec,epoch_rec,epoch_F1))
            print(bcolors.HEADER)

            train_precision.append(epoch_prec) if phase == 'train' else valid_precision.append(epoch_prec)
            train_recall.append(epoch_rec) if phase == 'train' else valid_recall.append(epoch_rec)
            #attempt to exclude nan values while doing optuna optimization
            if phase == 'train' and np.isnan(epoch_F1) == False:
                train_F1.append(epoch_F1)
            if phase == 'valid' and np.isnan(epoch_F1) == False:
                valid_F1.append(epoch_F1)

            #save best model
            if phase == 'valid' and epoch_F1 > best_F1:
                best_F1 = epoch_F1
                print(bcolors.OKGREEN + "New best model F1: {}".format(epoch_F1))
                print(bcolors.HEADER)
                torch.save(model.state_dict(),'./saved_models/run9opt.pth') #%%


    # average F1 scores for optuna (IS THIS VALID??)
    train20 = train_F1[-20:-1]
    print(train20) #this grabs UP TO the last 20 F1 scores (in early trials there are  often less)
    valid20 = valid_F1[-20:-1]
    train_F1 = sum(train20) / len(train20)
    valid_F1 = sum(valid20) / len(valid20)

    time_elapsed = time.time() - start
    print(bcolors.OKGREEN + 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(bcolors.OKGREEN + 'Number of empty batches skipped: {}'.format(num_skipped))
    print(bcolors.OKGREEN + "Best model: {}".format(best_F1))
    print(bcolors.HEADER)

    return train_precision, valid_precision, train_recall, valid_recall,train_F1,valid_F1, model