import torch, torch.nn as nn
import nn_tilde
from typing import List, Tuple

import data # data helper methods
import model

model = model.Transformer()
# model.load_state_dict(torch.load('model.pt'))
print(model)