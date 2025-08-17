import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
"T": 1,
"width": 128,
"depth": 4,
"out_alpha": 1,  # 1/math.sqrt(width)
"connect_type": "pre"  # pre or post
"""

class Block(nn.Module):
    def __init__(self,width,depth,T,connect_type):
        super(Block, self).__init__()
        self.connect_type = connect_type
        self.scale = math.sqrt(T/depth)
        self.mlp = nn.Linear(width, width, bias=False)
    def forward(self, x):
        if self.connect_type == "pre":
            return x +  self.scale*self.mlp(F.relu(x))
        else:
            return x + self.scale*F.relu(self.mlp(x))

class ResNetMlp(nn.Module):
    def __init__(self, width,depth,T,out_alpha, connect_type,num_classes=10):
        super(ResNetMlp, self).__init__()

        # input layer
        self.input_layer = nn.Linear(3072, width, bias=False)

        # out layer
        self.output_layer = nn.Linear(width, num_classes, bias=False)
        
        # mid layer
        self.mid_layers = nn.ModuleList([Block(width,depth,T,connect_type) for _ in range(depth)])

        # inital
        self.reset_parameters()

        # mid output check
        self.hidden_state = {}
        
    def reset_parameters(self):
        # input layer
        nn.init.kaiming_normal_(self.input_layer.weight, a=1, mode='fan_in')
        # output layer
        nn.init.kaiming_normal_(self.output_layer.weight, a=1, mode='fan_in')

        # mid layer
        for layer in self.mid_layers:
            nn.init.kaiming_normal_(layer.mlp.weight, a=1, mode='fan_in')

    def forward(self, x):
        x = self.input_layer(x)
        for i, midlayer in enumerate(self.mid_layers):
            x = midlayer(x)
            # save output
            self.hidden_state["layer_{}".format(i)] = x.clone().detach().cpu()
            
        x = self.output_layer(x)
        return x,self.hidden_state