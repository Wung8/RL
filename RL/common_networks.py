import numpy as np
import torch 
from torch import nn
from torch import Tensor
from torch.nn import functional as F


# apparently the recommended one
def init_weights(model, gain='relu'):
    for layer in model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(gain))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    for layer in reversed(model):
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                layer.weight.div_(100)
            break

def base_MLP_model(input_space, output_space, headless=False):
    if headless:
        return nn.Sequential(
          nn.Linear(input_space[0],64),
          nn.Mish(),
          nn.Linear(64,64),
          nn.Mish()
        )
        
    model = nn.Sequential(
      nn.Linear(input_space[0],64),
      nn.Mish(),
      nn.Linear(64,64),
      nn.Mish(),
      nn.Linear(64,sum(output_space))
    )
    if sum(output_space) > 1: init_weights(model)
    return model

def mini_CNN_model(input_space, output_space, headless=False):
    n_input_channels = input_space[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
        nn.Mish(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.Mish(),
        nn.Flatten(start_dim=-3),
    )
    with torch.no_grad():
        n_flatten = cnn(torch.zeros((1,*input_space))).shape[1]

    if headless:
        return nn.Sequential(
            cnn,
            nn.Linear(n_flatten, 64),
            nn.Mish()
        )

    model = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, 256),
        nn.Mish(),
        nn.Linear(256, sum(output_space)),
    )
    if sum(output_space) > 1: init_weights(model)
    return model

def base_CNN_model(input_space, output_space, headless=False):
    if all(dim <= 7 for dim in input_space):
        return mini_CNN_model(input_space, output_space, headless)
    
    n_input_channels = input_space[0]
        
    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.Mish(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.Mish(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.Mish(),
        nn.Flatten(start_dim=-3),
    )
    with torch.no_grad():
        n_flatten = cnn(torch.zeros((1,*input_space))).shape[1]

    if headless:
        return nn.Sequential(
            cnn,
            nn.Linear(n_flatten, 64),
            nn.Mish()
        )

    model = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, 256),
        nn.Mish(),
        nn.Linear(256, sum(output_space)),
    )
    if sum(output_space) > 1: init_weights(model)
    return model

class base_LSTM_model(nn.Module):
    def __init__(self, input_space, output_space, n_lstm_layers, hidden_size):
        super(base_LSTM_model, self).__init__()
        # base feature extractors output size (batch_size, 64)
        if len(input_space) == 1:
            self.feature_extractor = base_MLP_model(input_space, -1, headless=True)
        else:
            self.feature_extractor = base_CNN_model(input_space, -1, headless=True)
        self.lstm = nn.LSTM(64, hidden_size, num_layers=n_lstm_layers)
        self.fc = nn.Linear(hidden_size, sum(output_space))

    def forward(self, x, lstm_states=None):
        seq_len, batch_size, *obs_space = x.shape

        # reshape not necessary for FC feature extractor but is for CNN
        x = x.view(batch_size * seq_len, *obs_space)
        x = self.feature_extractor(x)
        x = x.view(seq_len, batch_size, -1)
        if lstm_states is not None:
            x, lstm_states = self.lstm(x, lstm_states)
        else:
            x, lstm_states = self.lstm(x)
        x = self.fc(x)

        return x, lstm_states
##
##class Aggregate():
##    def __init__():
##
##    def forward(self, x: Tensor)
##
##class Aggregated_Memory_LSTM_model(base_LSTM_model):
##    def __init__(self, input_space, output_space, n_lstm_layers, hidden_size):
##        super().__init__(
##            input_space,
##            output_space,
##            n_lstm_layers,
##            hidden_size
##        )
##        self.aggregate = 
##
##    def forward(self, x, lstm_states=None):
##        seq_len, batch_size, *obs_space = x.shape
##
##        # reshape not necessary for FC feature extractor but is for CNN
##        x = x.view(batch_size * seq_len, *obs_space)
##        x = self.feature_extractor(x)
##        x = x.view(seq_len, batch_size, -1)
##        if lstm_states is not None:
##            x, lstm_states = self.lstm(x, lstm_states)
##        else:
##            x, lstm_states = self.lstm(x)
##        x = self.fc(x)
##
##        return x, lstm_states
    


    
    
