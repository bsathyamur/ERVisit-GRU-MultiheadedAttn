"""
- Script loads the classes and 
- functions for the dataset loader
"""

import os
import random
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class CustomDataset(Dataset):

    def __init__(self,diagSeqs,memberDemo,ER_Ind):        
        self.diag = diagSeqs
        self.demo = memberDemo
        self.er = ER_Ind        
        
    def __len__(self):        
        """
        Return the number of samples (i.e. patients).
        """
        return len(self.er)
    
    def __getitem__(self, index):    
        
        """
        Generates one sample of data.
        """

        return self.diag[index],self.demo[index]["age"],self.demo[index]["gender"],self.er[index]
        
def collate_fn(data):
    """
    Arguments:
        data: a list of samples fetched from CustomDataset
        
    Outputs:
        x: a tensor of shape (# patients, max # visits, max # diagnosis codes) of type torch.long
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        y: target label of type torch.float
    """

    sequences, age, gender, labels= zip(*data)
        
    numOfPatients = len(sequences)
    
    numVisits = [len(patient) for patient in sequences]
    maxNumOfVisits = max(numVisits)

    maxNumOfDiag = 0

    for i in range(numOfPatients):
        for j in range(len(sequences[i])):
            maxNumOfDiag = max(maxNumOfDiag,len(sequences[i][j]))  
            
    x = torch.zeros([numOfPatients,maxNumOfVisits,maxNumOfDiag], dtype=torch.long)
    masks = torch.zeros([numOfPatients,maxNumOfVisits,maxNumOfDiag], dtype=torch.bool)  

    for i in range(len(sequences)):
        if len(sequences[i]) < maxNumOfVisits:
            diff1 = maxNumOfVisits - len(sequences[i])
            while diff1 > 0:
                sequences[i].append(["*0*"]*maxNumOfDiag)
                diff1-=1

            for j in range(len(sequences[i])):
                if len(sequences[i][j]) < maxNumOfDiag:
                    diff2 = maxNumOfDiag - len(sequences[i][j])

                    while diff2 > 0:
                        sequences[i][j].append("*0*")
                        diff2-=1                       
    
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            for k in range(len(sequences[i][j])):
                if sequences[i][j][k] == "*0*":
                    x[i][j][k] = 0
                else:
                    x[i][j][k] = sequences[i][j][k]
                    masks[i][j][k] = 1

    age_x = torch.zeros([numOfPatients], dtype=torch.long)

    for i in range(len(age)):
        age_x[i] = age[i]    

    gender_x = torch.zeros([numOfPatients], dtype=torch.long)

    for i in range(len(gender)):
        gender_x[i] = gender[i] 

    y = torch.zeros([numOfPatients], dtype=torch.float)
    
    for i in range(len(labels)):
        y[i] = labels[i]
    
    return x, masks,age_x,gender_x,y

def load_data(train_dataset, val_dataset, batch_size, collate_fn):
        
    """
    This function returns the data loader for  train and validation dataset. 
    
    Arguments:
        train dataset: train dataset of type `CustomDataset`
        val dataset: validation dataset of type `CustomDataset`
        collate_fn: collate function
        
    Outputs:
        train_loader, val_loader: train and validation dataloaders
    """

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,collate_fn=collate_fn)
    
    return train_loader, val_loader