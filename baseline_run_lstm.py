"""
- Script loads the dictionary objects saved as pickle files
- and creates the train and test data set as per the
- split ratio configured in the default.yml
- builds the GRU multiheaded attention model
- trains and tests the model for the number of epochs
- configured in the default.yml file
"""
import time
import os
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data.dataset import random_split
from dataset_lstm import *
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import logging
import yaml
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

starttime = time.time()

# Define data path
DATA_PATH = Path("output/")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading the data from the pickle file
diagSeqs = pickle.load(open(DATA_PATH / Path("diagSeqs.pickle"), 'rb'))
memberDemo = pickle.load(open(DATA_PATH / Path("member_demo.pickle"), 'rb'))
ERInd = pickle.load(open(DATA_PATH / Path("ER_Ind.pickle"), 'rb'))

logging.info(f"Total # of patients (diag sequences) loaded {len(diagSeqs)}")
logging.info(f"Total # of patients (member Demographics) loaded {len(memberDemo)}")
logging.info(f"Total # of patients (ER Indicator - target) loaded {len(ERInd)}")

dataset = CustomDataset(diagSeqs,memberDemo,ERInd)
logging.info(f"Total # of records in the dataset  {len(dataset)}")

# retrieve the configuration values
with open('config/default.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

split_ratio = config['split_ratio']

logging.info(f"splitting the dataset for split ratio  {split_ratio}")

split = int(len(dataset)*split_ratio)

lengths = [split, len(dataset) - split]
train_dataset, val_dataset = random_split(dataset, lengths)

logging.info(f"length of train dataset  {len(train_dataset)}")
logging.info(f"length of test dataset  {len(val_dataset)}")

batch_size = config['batch_size']

logging.info(f"splitting data for batch size  {batch_size}")

train_loader, val_loader = load_data(train_dataset, val_dataset, batch_size, collate_fn)

def generateEmbedding(diag_x,diag_mask,embeddingDim):

    # Generate sum Embedding for diagnosis code
    embed_diag = nn.Embedding(540,embeddingDim)
    diagEmbed = embed_diag(diag_x)
    diag_mask = diag_mask.unsqueeze(-1)
    diagEmbed = diagEmbed * diag_mask    
    sum_embed_diag = diagEmbed.sum(dim=2)
    
    return sum_embed_diag

def get_last_visit(hidden_states, masks):
    
    hidden_states = hidden_states.permute(0,2,1)
    
    last_true_visits = torch.sum(masks, dim=-1)
    last_true_visits = last_true_visits.nonzero()
    last_true_visits_np = last_true_visits.numpy()   
    df = pd.DataFrame(last_true_visits_np, columns=["key", "val"])
    maxval=df.groupby("key")["val"].max()
    last_true_visits_np=maxval.to_numpy() 
    
    last_true_visits = torch.tensor(last_true_visits_np)
    last_true_visits = last_true_visits.unsqueeze(-1)
    last_true_visits = last_true_visits.unsqueeze(-1) 
    last_true_visits = last_true_visits.expand(hidden_states.size())
    
    true_h_n = torch.gather(hidden_states, dim=1, index=last_true_visits)[:, -1, :]    
    
    return true_h_n

class LSTM(nn.Module):
    def __init__(self,input_size=100,hidden_size=100,output_size=1,embedding_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(input_size,hidden_size)
        self.hidden = (torch.zeros(1,embedding_dim,hidden_size),torch.zeros(1,embedding_dim,hidden_size))
        self.fc1 = nn.Linear(embedding_dim,output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,embedding_seq,mask):
        embedding_seq = generateEmbedding(embedding_seq,mask,self.embedding_dim)        
        lstm_out,self.hidden = self.lstm(embedding_seq.view(len(embedding_seq),self.embedding_dim,-1),self.hidden)
        lstm_out = lstm_out.view(len(embedding_seq),self.embedding_dim,-1)

        out = get_last_visit(lstm_out,mask)        
        out = self.fc1(out)
        prob_out = self.sigmoid(out)
        prob_out = prob_out.view(-1)
        return prob_out 

embedding_dim = 128
max_claims = 100
model = LSTM(input_size = max_claims,hidden_size=max_claims,embedding_dim=embedding_dim)

logging.info(f"LSTM model created as  \n {model}")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def eval_model(model, val_loader):
    
    val_labels = []
    val_probs = []
    
    model.eval()
    for step, batch in enumerate(val_loader):
        b_x, b_masks, b_age, b_gender, b_labels = batch

        with torch.no_grad():
            
            # Define the model hidden state
            model.hidden = (torch.zeros(1,embedding_dim,max_claims),torch.zeros(1,embedding_dim,max_claims))
            
            b_probs = model(b_x, b_masks)
            val_labels.extend(b_labels.detach().numpy().tolist())
            val_probs.extend(b_probs.detach().numpy().reshape(-1).tolist())
            
            results = np.array(val_probs)>0.5
            
                                
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, results, average='binary')
    roc_auc = roc_auc_score(val_labels, val_probs)
    
    return precision, recall, f1, roc_auc

def train(model, train_loader, val_loader, n_epochs):
    
    model.train()
    
    for epoch in range(n_epochs):
        train_loss = 0
        for batch in train_loader:
            
            b_x, b_masks,b_age,b_gender,b_labels = batch
            
            optimizer.zero_grad()
            
            # Define the model hidden state
            model.hidden = (torch.zeros(1,embedding_dim,max_claims),torch.zeros(1,embedding_dim,max_claims))
            
            b_probs = model(b_x, b_masks)  
            loss = criterion(b_probs,b_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss
            
        train_loss = train_loss / len(train_loader)
        logging.info('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        p, r, f, roc_auc = eval_model(model, val_loader)
        logging.info('Epoch: %d \t Validation f: %.2f, acc: %.2f precision: %.2f recall: %2f'%(epoch+1,f,roc_auc,p,r))
    
# number of epochs to train the model
n_epochs = config['NumEpochs']
train(model, train_loader, val_loader, n_epochs)

logging.info(f"model training and validation completed in {(time.time() - starttime)/60:.4f} minutes.")