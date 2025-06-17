import torch
from torch import nn
import torch.nn.functional as F
from nrdf.model.encoder import ResnetBlockFC


class GlobalLocalDecoder(nn.Module):
    def __init__(self, dim=3, z_global_dim=128, z_local_dim=128,
                 hidden_size=256, n_blocks=5, o_dim=1,
                 leaky=False, return_features=False,
                 sigmoid=True, acts='all_noin_act_out', data_type='occ'):
        super().__init__()
        self.z_global_dim = z_global_dim
        self.z_local_dim = z_local_dim
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.return_features = return_features
        self.sigmoid = sigmoid
        self.acts = acts
        self.o_dim = o_dim
        self.data_type = data_type

        self.fc_p = nn.Linear(dim, 32)

        self.fc_in = nn.Linear(self.z_local_dim+self.z_global_dim+32, self.hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, self.o_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z_local, z_global, z_sparse=None, **kwargs):
        batch_size, T, D = p.size() 
        z_global = z_global.unsqueeze(1).repeat(1, T, 1)
        z_local = z_local.unsqueeze(1).repeat(1, T, 1)
        acts = []
        acts_inp = []
        acts_first_rn = []
        acts_first_net = []
        acts_inp_first_rn = []

        p = p.float()
        acts_inp.append(p)

        net = self.fc_p(p)
        if z_sparse is None:
            net = torch.cat([net, z_global, z_local], dim=2)
        else:
            net = torch.cat([net, z_global, z_local, z_sparse], dim=2)
        acts.append(net)
        acts_first_net.append(net)

        net = self.fc_in(net)
        acts.append(net)        

        for i in range(self.n_blocks):
            net = self.blocks[i](net)
            acts.append(net)

        last_act = net

        out = self.fc_out(self.actvn(net))
        last_act_out = [last_act, out]
        all_act_out = acts_inp + acts + [out]
        all_noin_act_out = acts + [out]
        
        out = out.squeeze(-1)
        if self.sigmoid:
            out = torch.sigmoid(out)

        if self.return_features:
            if self.acts == 'all':
                acts = torch.cat(acts, dim=-1)
            if self.acts == 'all_and_out':
                acts = torch.cat(all_act_out, dim=-1)                
            elif self.acts == 'all_noin_act_out':
                acts = torch.cat(all_noin_act_out, dim=-1)
            elif self.acts == 'last':
                acts = last_act
            elif self.acts == 'last_and_out':
                acts = torch.cat(last_act_out, dim=-1)
            elif self.acts == 'inp_first_rn':
                acts = torch.cat(acts_inp_first_rn, dim=-1)
            elif self.acts == 'first_net':
                acts = torch.cat(acts_first_net, dim=-1)
            acts = F.normalize(acts, p=2, dim=-1)
            return out, acts
        return out
        