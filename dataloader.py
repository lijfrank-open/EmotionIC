import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
import pandas as pd
import numpy as np

def batch_to_device(batch, target_device, used_id = [[0], [], []]):
    assert sum([len(uid) for uid in used_id]) != 0
    assert len(used_id) == 3

    r_features = [batch[:4][i].to(target_device)    for i in used_id[0]]
    x_features = [batch[4:10][i].to(target_device)  for i in used_id[1]]
    o_features = [batch[10:13][i].to(target_device) for i in used_id[2]]
    qmask, umask, label = [d.to(target_device) for d in batch[13:16]]

    return [(r_features + x_features + o_features, qmask, umask), label]

def get_train_hyparameters(datasets_name, classify='emotion'):
    if datasets_name == 'iemocap':
        num_class = 6
        labels = np.arange(num_class).tolist()
        loss_weights = torch.FloatTensor([1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668])       

    elif datasets_name == 'meld':
        num_class = 7 if classify=='emotion' else 3
        labels = np.arange(num_class).tolist()
        loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 0.84847735, 5.42461417, 1.21859721])  

    elif datasets_name == 'dailydialog':
        num_class = 7
        labels = np.arange(num_class).tolist()
        labels.pop(1)
        loss_weights = torch.FloatTensor([2, 0.3, 4, 4, 8, 4, 8])

    elif datasets_name == 'emorynlp':
        num_class = 7 if classify=='emotion' else 3
        labels = np.arange(num_class).tolist()
        loss_weights = torch.FloatTensor([0.5, 1, 1, 0.3, 1, 1, 0.9])  
   
    else:
        exit("Wrong dataset name, which can only be one of ['iemocap', 'meld', 'dailydialog', 'emorynlp']")

    return loss_weights, num_class, labels

def get_Emo_loaders(datasets_name, classify='emotion', batch_size=[32, 32, 32], num_workers=0, pin_memory=False):

    if datasets_name == 'iemocap':
        trainset = IEMOCAPRobertaCometDataset('train')
        validset = IEMOCAPRobertaCometDataset('valid')
        testset = IEMOCAPRobertaCometDataset('test')
    elif datasets_name == 'meld':
        trainset = MELDRobertaCometDataset('train', classify)
        validset = MELDRobertaCometDataset('valid', classify)
        testset = MELDRobertaCometDataset('test', classify)
    elif datasets_name == 'dailydialog':
        trainset = DailyDialogueRobertaCometDataset('train')
        validset = DailyDialogueRobertaCometDataset('valid')
        testset = DailyDialogueRobertaCometDataset('test')
    elif datasets_name == 'emorynlp':
        trainset = EmoryNLPRobertaCometDataset('train', classify)
        validset = EmoryNLPRobertaCometDataset('valid', classify)
        testset = EmoryNLPRobertaCometDataset('test', classify)
    else:
        exit("Wrong dataset name, which can only be one of ['iemocap', 'meld', 'dailydialog', 'emorynlp']")

    train_loader = DataLoader(trainset,
                            batch_size=batch_size,
                            collate_fn=trainset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                            batch_size=batch_size,
                            collate_fn=trainset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                            batch_size=batch_size,
                            collate_fn=testset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, split):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('/home/lijfrank/code/dataset/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.LongTensor(np.array([1 if x=='M' else 0 for x in self.speakers[vid]])),\
               torch.BoolTensor(np.array([1]*len(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(tuple(dat[i])) if i<7 else dat[i].tolist() for i in dat]


class EmoryNLPRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        
        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainId, self.testId, self.validId \
        = pickle.load(open('/home/lijfrank/code/dataset/EMORYNLP_features/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')
        
        sentiment_labels = {}
        for item in self.emotion_labels:
            array = []

            for e in self.emotion_labels[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]
            
        if classify == 'emotion':
            self.labels = self.emotion_labels
        elif classify == 'sentiment':
            self.labels = sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.LongTensor(np.array(np.nonzero(self.speakers[vid])[1])),\
               torch.BoolTensor(np.array([1]*len(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid            

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(tuple(dat[i])) if i<7 else dat[i].tolist() for i in dat]


class DailyDialogueRobertaCometDataset(Dataset):

    def __init__(self, split):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('/home/lijfrank/code/dataset/DAILYDIALOG_features/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.LongTensor(np.array([1 if x=='0' else 0 for x in self.speakers[vid]])),\
               torch.BoolTensor(np.array([1]*len(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(tuple(dat[i])) if i<7 else dat[i].tolist() for i in dat]


class MELDRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping = 
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('/home/lijfrank/code/dataset/MELD_features/meld_features_roberta.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        return  torch.FloatTensor(np.array(self.roberta1[vid])),\
                torch.FloatTensor(np.array(self.roberta2[vid])),\
                torch.FloatTensor(np.array(self.roberta3[vid])),\
                torch.FloatTensor(np.array(self.roberta4[vid])),\
                torch.LongTensor(np.array(np.nonzero(self.speakers[vid])[1])),\
                torch.LongTensor(np.array([1]*len(self.labels[vid]))),\
                torch.LongTensor(np.array(self.labels[vid])),\
                vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(tuple(dat[i])) if i<7 else dat[i].tolist() for i in dat]





if __name__ == '__main__':


    train_loader, valid_loader, test_loader =\
            get_Emo_loaders(datasets_name='meld',
                            batch_size=32,
                            num_workers=8)

    for iter, data in enumerate(train_loader):

        r1, _, _, _, \
        identity_labels, umask, labels = [d for d in data[:-1]]

        print(r1.shape, identity_labels.shape, umask.shape, labels.shape)
        
        if iter == 0:
            break