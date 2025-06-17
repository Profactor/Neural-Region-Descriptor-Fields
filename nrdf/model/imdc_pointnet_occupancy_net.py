import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nrdf.model.encoder import ResnetBlockFC
from nrdf.model.encoder import IMDCEncoder, ResnetPointnet
from nrdf.model.decoder import GlobalLocalDecoder


class positional_encoding(object):
    #REF: https://github.com/dsvilarkovic/dynamic_plane_convolutional_onet/blob/main/src/common.py#L420
    def __init__(self, basis_function='sin_cos', L = 3):
        super().__init__()
        self.func = basis_function
        self.L = L
        freq_bands = 2.**(np.linspace(0,self.L-1, self.L))
        self.freq_bands =  freq_bands if self.L==1 else freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = [p]  
        for freq in self.freq_bands:
            out.append(torch.sin(freq * p))
            out.append(torch.cos(freq * p))

        p = torch.cat(out, dim=(len(p.shape)-1))
        return p

def posenc(self, x):
        """
        REF: https://github.com/ispc-lab/NeuralPCI/blob/main/model/NeuralPCI.py#L159
        sinusoidal positional encoding : Nx3 ——> Nx9
        [x] ——> [x, sin(x), cos(x)]
        """
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        x = torch.cat((x, sinx, cosx), dim=-1)
        return x


class IMDCPointOnet(nn.Module):
    def __init__(self,
                 local_latent_dim=256,
                 global_latent_dim = 128,
                 data_type = 'occ',
                 sigmoid=True,
                 o_dim=1,
                 return_features=False,
                 acts='all_noin_act_out',
                 encode_position=False,
                 scaling=10.0):
        super().__init__()

        self.scaling = scaling
        self.return_features = return_features
        self.encode_position = encode_position
        self.data_type = data_type
        self.o_dim = o_dim
        self.position_encoder = positional_encoding()
        enc_in_dim = 9 if self.encode_position else 3

        self.local_encoder = IMDCEncoder(channel=enc_in_dim, z_local_dim=local_latent_dim)
        self.global_encoder = ResnetPointnet(dim=enc_in_dim, z_global_dim=global_latent_dim)
        self.decoder = GlobalLocalDecoder(dim=enc_in_dim, z_global_dim=global_latent_dim, z_local_dim=local_latent_dim, sigmoid=sigmoid, o_dim=self.o_dim,
            return_features=return_features, acts=acts, data_type=self.data_type)
    
   
    def forward(self, input, z=None, for_im_corr=False):
        out_dict = {}
        if self.encode_position:
            enc_in = self.position_encoder(input['point_cloud'])* self.scaling 
        else:
            enc_in = input['point_cloud'] * self.scaling

        if for_im_corr:
            if self.encode_position:
                query_points = self.position_encoder(input['obj_pcd']) * self.scaling 
            else:            
                query_points = input['obj_pcd'] * self.scaling 
        else:
            if self.encode_position:
                query_points = self.position_encoder(input['coords'])* self.scaling 
            else:              
                query_points = input['coords'] * self.scaling
        
        z_local = self.local_encoder(enc_in)
        out_dict['z'] = {'z_local': z_local}

        z_global = self.global_encoder(enc_in)
        out_dict['z']['z_global'] = z_global

        if self.return_features:
            if self.o_dim == 2:
                out_dict['sdf'], out_dict['occ'], out_dict['features'] = self.decoder(query_points, z_local, z_global)
            else:
                out_dict['occ'], out_dict['features'] = self.decoder(query_points, z_local, z_global)
        else:
            if self.o_dim==2:
                out_dict['sdf'], out_dict['occ'] = self.decoder(query_points, z_local, z_global)
            else:
                out_dict['occ'] = self.decoder(query_points, z_local, z_global)

        return out_dict
    
