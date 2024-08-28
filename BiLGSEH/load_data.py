import random

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import pdb
import torch
import h5py
import scipy.io as sio
import pickle

# torch.multiprocessing.set_sharing_strategy('file_system')

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs):
        self.images = images
        self.texts = texts
        self.labs = labs

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        return img, text, lab, index

    def __len__(self):
        count = len(self.texts)
        return count

def load_dataset(dataset, batch_size):
    '''
        load datasets : mirflickr, mscoco, nus-wide
    '''
    train_loc = 'datasets/' + dataset + '/train.pkl'
    query_loc = 'datasets/' + dataset + '/query.pkl'
    retrieval_loc = 'datasets/' + dataset + '/retrieval.pkl'

    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = torch.tensor(data['label'], dtype=torch.int64)
        train_texts = torch.tensor(data['text'], dtype=torch.float32)
        train_images = torch.tensor(data['image'], dtype=torch.float32)

    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_labels = torch.tensor(data['label'], dtype=torch.int64)
        query_texts = torch.tensor(data['text'], dtype=torch.float32)
        query_images = torch.tensor(data['image'], dtype=torch.float32)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_lables = torch.tensor(data['label'], dtype=torch.int64)
        retrieval_texts = torch.tensor(data['text'], dtype=torch.float32)
        retrieval_images = torch.tensor(data['image'], dtype=torch.float32)

    imgs = {'train': train_images[:4992], 'query': query_images, 'retrieval': retrieval_images}
    texts = {'train': train_texts[:4992], 'query': query_texts, 'retrieval': retrieval_texts}
    labs = {'train': train_labels[:4992], 'query': query_labels, 'retrieval': retrieval_lables}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x]) for x in ['train', 'query', 'retrieval']}
    shuffle = {'train': True, 'query': False, 'retrieval': False}
    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=shuffle[x],
                                num_workers=4) for x in ['train', 'query', 'retrieval']}

    return dataloader, (train_images, train_texts, train_labels)
