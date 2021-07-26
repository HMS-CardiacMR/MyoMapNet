import torch.nn as nn
import torch

net_scale = 2

def save_models(model, name, epoch):
    """
    Save model function - you should sepecify the path in here

    Arguments:
    model: the model to save
    name -- string, name of the model 
    
    """

    torch.save(model.state_dict(), "path" + name + ".model")
    print("Model saved at epoch {}".format(epoch))

class DenseNN(nn.Module):
    """
    Implementation of the MyoMapNet architecture
    
    Arguments:
    in_channels -- a scalar, for number of input channels
    out_channels --  a scalar, for number of output channel   
    x -- a tensor, forward function takes a tensor as input with 4 T1W and 4 IT 
    
    Returns:
    net -- a tensor, the predicted T1 map
    """
    
    def __init__(self, in_channels, out_channels):
        super(DenseNN, self).__init__()

        self.T1Net = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=200*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=200*net_scale, out_features=200*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=200*net_scale, out_features=100*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=100*net_scale, out_features=100*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=100*net_scale, out_features=50*net_scale),
            nn.LeakyReLU(),
            nn.Linear(in_features=50*net_scale, out_features=out_channels),
        )


    def forward(self, x):
        net = self.T1Net(x)
        return net




