import torchvision
import torch.nn as nn
import torch 
import cv2 , os , numpy as np 
from glob import glob
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms,models,datasets
from random import shuffle  , seed
from Model import Model
from torchsummary import summary


class DataSet(Dataset):

    def __init__(self,folder):
        pan = glob(folder+"/pan/*")
        addhar = glob(folder+"/aadhar/*")
        passport = glob(folder+"/passport/*")
        self.fpath = pan+addhar+passport
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.299,0.224,0.225])
        seed(10)
        shuffle(self.fpath)
        self.label2idx = {"pan":0,
                          "aadhar":1,
                          "passport":2,
                          }
        
        self.targets = [self.label2idx[fname.split("/")[-2]] for fname in self.fpath]

    def __len__(self):return len(self.fpath)
    def __getitem__(self,ix):

        f = self.fpath[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:,:,::-1])
        im = cv2.resize(im,(224,224))
        im = torch.tensor(im/255)
        im = im.permute(2,0,1)
        im = self.normalize(im)
        return im.float().to() , torch.tensor([target]).to()
    
def train_batch(x,y,model,optimizer,loss_fn):
    model.train()
    prediction = model(x)
    #_, predictedy = torch.max(prediction, 1)
    #print(predictedy , y.squeeze(1))
    batch_loss = loss_fn(prediction, y.squeeze(1))
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x,y,model):
    model.eval()
    prediction = model(x)
    _, predicted = torch.max(prediction, 1)
    _, predictedy = torch.max(y, 1)
    is_correct = (predicted ) == predictedy
    return is_correct.cpu().numpy().tolist()

def get_data():
    train = DataSet("../dataset/")
    trnl_dl = DataLoader(train,batch_size =16, shuffle=True,drop_last = True) 
    return trnl_dl , trnl_dl



if __name__:
    train = DataSet("../dataset/")
    model, loss_fn, optimizer = Model()
    trn_dl, val_dl = get_data()
    
    print(train.targets)
    print(train.fpath)
    print(summary(model, torch.zeros(1,3,224,224)))

    train_losses, train_accuracies = [], []
    val_accuracies = []
    for epoch in range(25):
        print(f" epoch {epoch + 1}/25")
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies = []

        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss) 
        train_epoch_loss = np.array(train_epoch_losses).mean()
        print("epoch :", epoch , " loss :", train_epoch_loss)
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        print("epoch :", epoch , " Accuracy :", train_epoch_accuracy)
        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
        

    torch.save(model, "./Model-epoch 3pkl.pth")
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save("./Model-epoch 3.pth") # Save