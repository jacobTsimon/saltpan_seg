import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from torchmetrics import CohenKappa
from torchvision import transforms
from sklearn.metrics import f1_score

#normalizing for resnet
preprocess = transforms.Compose([

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

def F1_score(precision,recall):
    f1 = 2 * (precision * recall)/(precision + recall)
    return f1

def Z1Norm(input,bands,batch_size):
    t_n = torch.zeros([batch_size,bands,512,512])
    for b in range(bands):
        # imax, imin = input[:,b,:,:].max(), input[:,b,:,:].min()
        # nmin, nmax = 0, 1
        imax,imin = maxs[b],mins[b]
        plus = input[:,b,:,:] #+ abs(imin)

        t_n[:,b,:,:] = (plus - imin) / (imax - imin)  # * (nmax - nmin) + nmin

    return t_n

mins = [0,0,0,0,-1.8980236]
maxs = [9027,10810,12319,11044,29.606974]

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1,batches=1,modname = None,batch_size = 3):
    start = time.time()
    model.cuda()

    kappa_fn = CohenKappa(task='binary').to('cuda')
    train_precision, valid_precision = [], []
    train_recall, valid_recall = [], []
    train_F1, valid_F1 = [], []

    best_acc = 0.0
    num_skipped = 0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_dl
                # Llogdir = 'TBruns/run1_train_Loss'
                # Plogdir = 'TBruns/run1_train_Precision'
                # Rlogdir = 'TBruns/run1_train_Recall'
                # Flogdir = 'TBruns/run1_train_F1'
                # Lwriter = SummaryWriter(log_dir=Llogdir)
                # Pwriter = SummaryWriter(log_dir=Plogdir)
                # Rwriter = SummaryWriter(log_dir=Rlogdir)
                # Fwriter = SummaryWriter(log_dir=Flogdir)
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl
                # Llogdir = 'TBruns/run1_val_Loss'
                # Plogdir = 'TBruns/run1_val_Precision'
                # Rlogdir = 'TBruns/run1_val_Recall'
                # Flogdir = 'TBruns/run1_val_F1'
                # Lwriter = SummaryWriter(log_dir=Llogdir)
                # Pwriter = SummaryWriter(log_dir=Plogdir)
                # Rwriter = SummaryWriter(log_dir=Rlogdir)
                # Fwriter = SummaryWriter(log_dir=Flogdir)

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
                if y.sum() == 0:
                    print("Step {} empty - skipped".format(step))
                    num_skipped += 1
                    continue

                # try to screen for nans in elevation data
                nanchek = torch.any(x.isnan(), dim=1)

                if nanchek.long().sum() != 0:
                    print(bcolors.FAIL + "NaN in elevation @ step {} - skipped".format(step))
                    print(bcolors.ENDC)
                    num_skipped += 1
                    continue

                y = y[:,0,:,:] # convert the 4D tensors to 3D (Batch size, X, Y)


                y = y.cuda()

                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    #outU = outputs.detach().cpu().numpy()
                    #print(np.unique(outU))
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

                a = torch.reshape(outputs.argmax(dim=1).detach().cpu(),(-1,)) #.argmax(dim=1)

                b = torch.reshape(y.cpu(),(-1,))

                precision = precision_score(y_pred = a,y_true = b)
                recall = recall_score(y_pred = a,y_true = b)
                f1 = f1_score(y_true = b,y_pred = a)
                #print(outputs.argmax(dim=1) == y.cuda())
                # cm = confusion_matrix(y.cpu(),outputs.cpu().argmax(dim= 1))
                # print(cm)

               # running_acc += acc #* batch_size
                #running_kappa += kappa
                running_loss += loss #* batch_size
                running_P += precision
                running_R += recall
                running_F1 += f1
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

            # Lwriter.add_scalar("Loss/{}".format(phase),epoch_loss,epoch)
            # Pwriter.add_scalar("Precision/{}".format(phase),epoch_prec,epoch)
            # Rwriter.add_scalar("Recall/{}".format(phase),epoch_rec,epoch)
            # Fwriter.add_scalar("F1/{}".format(phase),epoch_F1,epoch)

            print('{} epoch {} | epoch Loss: {:.4f} |   epoch Precision: {} | epoch Recall: {} | epoch F1: {}'.format(phase,epoch, epoch_loss, epoch_prec,epoch_rec,epoch_F1))

            train_precision.append(epoch_prec) if phase == 'train' else valid_precision.append(epoch_prec)
            train_recall.append(epoch_rec) if phase == 'train' else valid_recall.append(epoch_rec)
            #attempt to exclude nan values while doing optuna optimization
            if phase == 'train' and np.isnan(epoch_F1) == False:
                train_F1.append(epoch_F1)
            if phase == 'valid' and np.isnan(epoch_F1) == False:
                valid_F1.append(epoch_F1)


    # average F1 scores for optuna (IS THIS VALID??)
    train20 = train_F1[-20:-1]
    print(train20) #this grabs UP TO the last 20 F1 scores (in early trials there are  often less)
    valid20 = valid_F1[-20:-1]
    train_F1 = sum(train20) / len(train20)
    valid_F1 = sum(valid20) / len(valid20)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Number of empty batches skipped: {}'.format(num_skipped))

    return train_precision, valid_precision, train_recall, valid_recall,train_F1,valid_F1, model