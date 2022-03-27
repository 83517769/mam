import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
from PIL import Image
import random

class ImageDataset(Dataset):
    def __init__(self, df,label,transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.label = label
        self.transform = transform

    def __len__(self):
        #print('len = {}'.format((self.df.shape[0]*400)))
        return self.df.shape[0]*400

    def __getitem__(self, idx):
#        path = self.df[0][0][idx][0][0]
#        name = self.df[0][1][idx][0][0]
#        box=50
#        img = self.transform(Image.open(path))
        #print('idx={}'.format(idx))
        raw=self.df[idx//400,:,:]
#        x=random.randint(0,227-box)
#        y=random.randint(0,227-box)
#        raw[x:x+50,y:y+50]=0
        #print('raw={}'.format(raw))
        #print('raw.type =%s' % (type(raw)))
        img = Image.fromarray(raw.astype('uint32')).convert('RGB')
        labels = int(self.label[idx//400])
        img = self.transform(img)
        sample = {'img':img,'labels':labels}

        return sample
        
class ImageDataset_test(Dataset):
    def __init__(self, df,label,transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.label = label
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
#        path = self.df[0][0][idx][0][0]
#        name = self.df[0][1][idx][0][0]
        
#        img = self.transform(Image.open(path))
        img = Image.fromarray(self.df[idx,:,:].astype('uint32')).convert('RGB')
        labels = int(self.label[idx])
        img = self.transform(img)
        sample = {'img':img,'labels':labels}
        return sample