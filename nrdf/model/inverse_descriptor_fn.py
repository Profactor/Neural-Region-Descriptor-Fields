import torch
from torch import nn
import torch.nn.functional as F
from nrdf.model.encoder import ResnetBlockFC


class InvDescAllGloblaLocalFn(nn.Module):
    def __init__(self, z_global_dim=128, z_local_dim=256, desc_dim=1953, hidden_size_0=1024, hidden_size=512, return_features=False, out_dim=3):
        super(InvDescAllGloblaLocalFn, self).__init__() 
        self.return_features = return_features
        fc_in_dim = z_global_dim+z_local_dim+desc_dim

        self.fc_in = nn.Linear(fc_in_dim, hidden_size_0)
        self.block0 = ResnetBlockFC(hidden_size_0, size_out=hidden_size)
        self.block1 = ResnetBlockFC(hidden_size, size_out=int(hidden_size/2))
        self.block2 = ResnetBlockFC(int(hidden_size/2))
        self.fc_out = nn.Linear(int(hidden_size/2), out_dim)

    def forward(self, z, f):
        z_global = z['z_global']
        z_local = z['z_local']
        _, T, _ = f.size()

        z_global = z_global.unsqueeze(1).repeat(1, T, 1) 
        z_local = z_local.unsqueeze(1).repeat(1, T, 1) 
        net = torch.cat([z_global, z_local, f], dim=2)

        x1 = self.fc_in(net)
        
        x3 = self.block0(x1)

        x4 = self.block1(x3)

        x5 = self.block2(x4)
        x5 = F.leaky_relu(x5, negative_slope=0.02)      
               
        x6 = self.fc_out(x5)
        x6 = torch.tanh(x6)

        return x6
    