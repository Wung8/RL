import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F


# apparently the recommended one
def init_weights(model, gain='relu'):
    for layer in model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(gain))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def base_MLP_model(input_space, output_space):
    model = nn.Sequential(
      nn.Linear(input_space[0],64),
      nn.Mish(),
      nn.Linear(64,64),
      nn.Mish(),
      nn.Linear(64,output_space)
    )
    init_weights(model)
    return model

def base_CNN_model(input_space, output_space):
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
        
    linear = nn.Sequential(
        nn.Linear(n_flatten, 256),
        nn.Mish(),
        nn.Linear(256, output_space),
    )

    model = nn.Sequential(
        cnn,
        linear,
    )
    init_weights(model)
    return model
