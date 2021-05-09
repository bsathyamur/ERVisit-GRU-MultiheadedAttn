"""
- Script loads the dictionary objects saved as pickle files
- and creates the train and test data set as per the
- split ratio configured in the default.yml
- builds the Random Forest model
- trains and tests the model for the number of epochs
- configured in the default.yml file
"""


import pickle
import os
import numpy as np

# Define data path
DATA_PATH = Path("output/")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading the data from the pickle file
diagSeqs = pickle.load(open(DATA_PATH / Path("diagSeqs.pickle"), 'rb'))
memberDemo = pickle.load(open(DATA_PATH / Path("member_demo.pickle"), 'rb'))
ER_Ind = pickle.load(open(DATA_PATH / Path("ER_Ind.pickle"), 'rb'))

logging.info(f"Total # of patients (diag sequences) loaded {len(diagSeqs)}")
logging.info(f"Total # of patients (member Demographics) loaded {len(memberDemo)}")
logging.info(f"Total # of patients (ER Indicator - target) loaded {len(ER_Ind)}")


features = np.full((len(diagSeqs), 100), -1)
labels = np.full(len(diagSeqs), -1)
#print(features)
#print(labels)

for pid in diagSeqs:
    diags = diagSeqs[pid]
    if pid in ER_Ind:
        seq = 0
        print(len(diags))
        for diag in diags:
            #features[pid][seq] = np.array(diag, dtype=object)
            features[pid][seq] = diag[0]
            #features[pid][seq] = diag
            labels[pid] = ER_Ind[pid]
            seq += 1
            
#print(features)
#print(labels)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)
print("Training Complete!")


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
#print(predictions)
predictions[predictions>=0.5] = 1
predictions[predictions<0.5] = 0
#print(predictions)
#print(test_labels)

# Calculate the absolute errors
errors = abs(predictions - test_labels)
print("Number of mis-detected/ errored predictions: ", len(errors[errors>0]))
print("Number of correct predictions: ", len(errors[errors<1]))

# calculate accuracy
accuracy = 100 * (len(errors[errors<1])/ len(errors))
print('Accuracy:', round(accuracy, 2), '%.')


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# calculate prediction
precision = precision_score(test_labels, predictions, average='binary')
print('Precision: %.3f' % precision)

# calculate recall
recall = recall_score(test_labels, predictions, average='binary')
print('Recall: %.3f' % recall)
