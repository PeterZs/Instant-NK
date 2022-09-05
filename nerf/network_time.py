import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from encoding import get_encoder
from .renderer import NeRFRenderer
import math
import sys
import time
sys.path.append("/data/new_disk/jyh/animationMesh")
import svr

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=256,
                 hidden_deform_dim=128,	
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 cuda_ray=False,
                 num_frames=2,
                 ):
        super().__init__(cuda_ray)

        self.num_frames=num_frames
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        # self.encoder_deform, self.in_dim_deform = get_encoder(encoding, input_dim = 4,num_levels = 32)
        # deform_net = []
        # for l in range(num_layers):
        #     if l==0:
        #         in_dim = self.in_dim_deform
        #     else:
        #         in_dim = hidden_deform_dim

        #     if l == num_layers - 1:
        #         out_dim = 3
        #     else:
        #         out_dim = hidden_deform_dim

        #     deform_net.append(nn.Linear(in_dim, out_dim))
        
        # self.deform_net = nn.ModuleList(deform_net)

        self.encoder, self.in_dim = get_encoder(encoding, input_dim=4, num_levels=32)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1+ self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))

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
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.color_net = nn.ModuleList(color_net)
        self.pid = 0

    
    def forward(self, x, d, bound,frame_id,knn_field = None, weight_field = None, flag_field = None, res = None, left_top = None,leng = None, ed_nodes = None, ed_motions = None):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]
        # print('st',time.time())
        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)
        # warp
        # print('saving %d'%self.pid)

        if knn_field != None:
            st = time.time()
            x[:,[0,1,2]] = x[:,[2,0,1]]
            x = x * 3
            # x[:,1]*=-1
            # x[:,2]*=-1
            # st = time.time()
            # np.savetxt('orig_%d.txt'%self.pid,x.detach().cpu().numpy())

            # np.savetxt('ed_%d.txt'%self.pid,ed_nodes.detach().cpu().numpy())
            coords_out = svr.warp_vertices(x, ed_nodes, 
                                                ed_motions, 
                                                knn_field,
                                                weight_field, 
                                                flag_field, 
                                                res,left_top,leng)
            torch.cuda.synchronize()
            coords_out = coords_out[0]                                    
            # np.savetxt('warp_%d.txt' % self.pid,coords_out.detach().cpu().numpy())
            # x[coords_out<-2.5] = 0
            self.pid+=1
            # coords_out[coords_out>-2.5] = x
            x = coords_out   
            # print(self.pid)
            # np.savetxt('del_%d.txt' % self.pid, x.detach().cpu().numpy())

            # x[:,1]*=-1
            # x[:,2]*=-1
            x = x/3  
            x[:,[2,0,1]] = x[:,[0,1,2]]
            # print('use explicit warp',time.time()-st)

       
        # deform 
        frame_id = torch.ones((x.size(0),1), device=x.device) * frame_id /self.num_frames
        frame_id=frame_id*2-1# nomalized in [-1, 1]
        x_t=torch.cat([x,frame_id],dim=-1)
        h = self.encoder(x_t, size=bound)
        # h = self.encoder_deform(x_t, size = bound)
        # for l in range(self.num_layers):
        #     h = self.deform_net[l](h)
        #     if l!= self.num_layers - 1:
        #         h = F.relu(h,inplace = True)
        
        # delta_x = torch.sigmoid(h) - 0.5 # in [-1,1]
        # x_d = torch.clamp(x + delta_x,min = -1,max = 1)
        # h = self.encoder(x_d, size=bound)

        # sigma

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
        # print('ed',time.time())
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
