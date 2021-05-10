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
from dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import logging
import yaml
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def sum_embeddings_with_mask(x, masks):
    """
    Arguments:
        x: the embeddings of diagnosis sequence of shape (batch_size, # claims, # diagnosis codes, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, # visits, embedding_dim)
    """

    mask = masks.unsqueeze(-1)
    mask_embeddings = x * mask.float()
    sum_embeddings = mask_embeddings.sum(dim=2)
    
    return sum_embeddings

def get_last_visit(hidden_states, masks):
     
    """
    Arguments:
        hidden_states: the hidden states of each visit of shape (batch_size, # visits, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        last_hidden_state: the hidden state for the last true visit of shape (batch_size, embedding_dim)
    """

    sum_masks = masks.sum(axis = 2)
    
    last_true_visits = ((sum_masks > 0).sum(axis = 1)-1).unsqueeze(1)
    
    last_true_visits = last_true_visits.unsqueeze(1).expand(hidden_states.size())
    
    last_hidden_state = torch.gather(hidden_states, dim=1, index=last_true_visits)[:, -1, :]    
    
    return last_hidden_state

class gruMHRNN(nn.Module):
    
    def __init__(self, num_codes, embeddingDim, embeddingDimDemo=128,include_demo=False):

        super().__init__()
        
        # Embedding for the claim diagnosis codes labels
        self.embed_diag = nn.Embedding(num_codes,embeddingDim)

        if include_demo:
            # Embedding for the age demographics
            self.embed_age = nn.Embedding(121,embeddingDimDemo)

            # Embedding for the gender demographics
            self.embed_gender = nn.Embedding(4,embeddingDimDemo)

        # GRU layer
        self.gru = nn.GRU(input_size=embeddingDim,hidden_size=embeddingDim,batch_first = True)
        
        # Multiheaded attention
        self.attn_alpha = nn.Parameter(torch.empty(1))

        self.attn = nn.MultiheadAttention(embeddingDim,8)
        
        # Final Fully connected linear layer
        if include_demo:
            
            # Fully connected linear layer for member demographics
            self.fc1 = nn.Linear(embeddingDimDemo * 2, embeddingDimDemo)            
            
            self.fc2 = nn.Linear(embeddingDim + embeddingDimDemo, 1)
        else:
            self.fc2 = nn.Linear(embeddingDim, 1)
        
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x,masks,age,gender,include_demo=False):

        seq_out = self.embed_diag(x)
        seq_out = sum_embeddings_with_mask(seq_out,masks)
        
        # Pass through the GRU layer
        gru_out,hidden_state = self.gru(seq_out)

        # Apply the masks on the gru output
        last_true_visits = torch.sum(masks, dim=-1)
        last_true_visits = (last_true_visits > 0)
        last_true_visits = last_true_visits.unsqueeze(-1)

        gru_out = gru_out * last_true_visits
        
        # Pass the gru output as query, key, value to the multiheaded attention
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        
        # Get the maximum attention
        rnn_attn_max = F.adaptive_max_pool1d(attn_out.permute(0, 2, 1), 1).squeeze()
    
        # Retrieve the last true visit from the GRU output
        last_true_visit = get_last_visit(gru_out,masks)
        
        # Apply the attention output to the last true visit
        rnn_out = (1 - self.sigmoid(self.attn_alpha)) * last_true_visit + rnn_attn_max * self.sigmoid(self.attn_alpha)
        
        # If static demographics needs to be included in the model
        if include_demo:

            age_out = self.embed_age(age)
            gender_out = self.embed_gender(gender)

            final_demo_embed = torch.cat([age_out,gender_out],dim=1)

            demo_out = self.fc1(final_demo_embed)

            final_output = torch.cat([demo_out,rnn_out],dim=1)

            linear_out = self.fc2(final_output)

        else:
            # Pass the output through the final fully connected layer
            linear_out = self.fc2(rnn_out)
        
        # Pass through the activiation function
        out = self.sigmoid(linear_out)

        prob_out = out.view(-1)
        
        return prob_out

# load the model
num_ccs_codes = config['num_ccs_codes']
embeddingDim = config['embeddingDim']
include_demo = config['include_demo']
embeddingDimDemo = config['embeddingDimDemo']

gruMH_rnn = gruMHRNN(num_codes = num_ccs_codes,embeddingDim = embeddingDim,embeddingDimDemo = embeddingDimDemo,include_demo = include_demo)

logging.info(f"Model include demographics static data is set to  {include_demo}")
logging.info(f"GRU multiheaded model created as  \n {gruMH_rnn}")

lr = config['lr']

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(gruMH_rnn.parameters(), lr = lr)

logging.info(f"optimizer initialized with learning rate  {lr}")

def eval_model(model, val_loader):
    
    val_labels = []
    val_probs = []
    
    model.eval()
    for step, batch in enumerate(val_loader):
        b_x, b_masks, b_age,b_gender, b_labels = batch

        with torch.no_grad():
            
            b_probs = model(b_x, b_masks,b_age,b_gender,include_demo)
            val_labels.extend(b_labels.detach().numpy().tolist())
            val_probs.extend(b_probs.detach().numpy().reshape(-1).tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, np.array(val_probs)>0.5, average='binary')
    roc_auc = roc_auc_score(val_labels, val_probs)
    
    return precision, recall, f1, roc_auc

def train(model, train_loader, val_loader, n_epochs):
    
    model.train()
    
    losses = []
    auc = []
    fscore = []
    precision = []
    recall = []

    for epoch in range(n_epochs):
        train_loss = 0
        for batch in train_loader:
            
            b_x, b_masks,b_age, b_gender, b_labels = batch
            
            optimizer.zero_grad()
            
            b_probs = model(b_x, b_masks,b_age,b_gender,include_demo) 
            loss = criterion(b_probs,b_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss
            
        train_loss = train_loss / len(train_loader)

        logging.info('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))

        losses.append(train_loss)

        p, r, f, roc_auc = eval_model(model, val_loader)
        
        auc.append(roc_auc)
        fscore.append(f)
        precision.append(p)
        recall.append(r)
        
        logging.info('Epoch: %d \t Validation f: %.2f, acc: %.2f precision: %.2f recall: %2f'%(epoch+1,f,roc_auc,p,r))

    return  losses,auc,fscore,precision,recall  
        
# number of epochs to train the model
n_epochs = config['NumEpochs']
losses,auc,fscore,precision,recall = train(gruMH_rnn, train_loader, val_loader, n_epochs)

logging.info('Average AUC: %2f'%(sum(auc)/n_epochs))
logging.info('Average f1-score: %2f'%(sum(fscore)/n_epochs))
logging.info('Average precision: %2f'%(sum(precision)/n_epochs))
logging.info('Average recall: %2f'%(sum(recall)/n_epochs))

def plot_performance(filename,n_epochs):

    """
    Plot the performance of the epoch losses, AUC and fscore
    """

    X_ticks_array=[i for i in range(1,n_epochs+1)]

    # Figure size
    fig = plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Train Loss")
    plt.plot(losses, label="train")
    plt.xticks(X_ticks_array)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend(loc="upper right")
    plt.rcParams.update({"font.size": 10})

    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.title("AUC/f-score")
    plt.plot(auc, label="AUC")
    plt.plot(fscore, label="f-score")
    plt.xticks(X_ticks_array)
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend(loc="lower right")

    fig.tight_layout(pad=1)
    fig.canvas.draw()

    # Save figure
    plt.savefig(filename)

# Plot the performance of the training and validation with the model

PLT_PATH = "plot/"
plot_performance(PLT_PATH + "model-performance.png",n_epochs)

logging.info("model performance saved as file model-performance.png")

logging.info(f"model training and validation completed in {(time.time() - starttime)/60:.4f} minutes.")