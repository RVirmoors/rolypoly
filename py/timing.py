"""
Rolypoly timing model
2020 rvirmoors

TODO methods:
new performance
add row
preprocess dataset
train model
inference

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# Helper libraries
import random
#from tqdm import tqdm

# Deterministic results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)

"""


def inference(featVec):
    return 0


def addRow(featVec, y):
    newRow = np.append(featVec, y)
    #print('saved as: ', newRow)
