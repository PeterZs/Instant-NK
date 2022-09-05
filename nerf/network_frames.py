import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from .renderer import NeRFRenderer
import math

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 hidden_deform_dim=512,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 cuda_ray=False,
                 num_frames=2,
                 ):
        super().__init__(cuda_ray)

        self.latent_dim = 32
        # deform network
        self.latent_features = torch.empty((num_frames, self.latent_dim))
        nn.init.normal_(self.latent_features)
        self.latent_features = nn.Parameter(self.latent_features)

        self.num_frames=num_frames
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding)

        deform_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim_deform+self.latent_dim
            else:
                in_dim = hidden_deform_dim

            if l == num_layers - 1:
                out_dim = 3
            else:
                out_dim = hidden_deform_dim

            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)

        self.encoder, self.in_dim = get_encoder(encoding)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    
    def forward(self, x, d, bound,frame_id):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)

        # deform
        x_enc = self.encoder_deform(x, size=bound)
        frame_id = torch.ones(x.size(0), device=x.device) * frame_id
        latent = torch.index_select(self.latent_features, 0, frame_id.squeeze().long())
        h=torch.cat([x_enc, latent], dim=-1)
        for l in range(self.num_layers):
            h = self.deform_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        delta_x=h
        # sigma
        x = self.encoder(x+delta_x, size=bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)

        return sigma, color

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]

        x = self.encoder(x, size=bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = torch.exp(torch.clamp(h[..., 0], -15, 15))
        sigma = F.relu(h[..., 0])

        return sigma
