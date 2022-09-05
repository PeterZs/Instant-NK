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
import sys
sys.path.append('/new_disk/jyh/code/apps/SMPLEstimation/')
from lbs import *
class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
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
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding, input_dim = 4,num_levels = 32)
        deform_net = []
        self.smpl_model = mySMPL(device = 'cuda')

        for l in range(num_layers):
            if l==0:
                in_dim = self.in_dim_deform
            else:
                in_dim = hidden_deform_dim

            if l == num_layers - 1:
                out_dim = 3
            else:
                out_dim = hidden_deform_dim

            deform_net.append(nn.Linear(in_dim, out_dim))
        
        self.deform_net = nn.ModuleList(deform_net)

        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, num_levels=32)

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
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
        self.pid = 0

    
    def forward(self, x, d, bound,frame_id, need_warp = True):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]
        # print('st',time.time())
        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)
        tmp = x.clone()
        if need_warp == True:
            x = self.check_warp(tmp.clone())

        h = self.encoder(x, size=bound)
        # deform 
        frame_id = torch.ones((x.size(0),1), device=x.device) * frame_id /self.num_frames
        frame_id=frame_id*2-1# nomalized in [-1, 1]
        x_t=torch.cat([x,frame_id],dim=-1)
        h = self.encoder_deform(x_t, size = bound)
        for l in range(self.num_layers):
            h = self.deform_net[l](h)
            if l!= self.num_layers - 1:
                h = F.relu(h,inplace = True)
        
        delta_x = torch.sigmoid(h) - 0.5 # in [-1,1]
        x_d = torch.clamp(x + delta_x,min = -1,max = 1)
        h = self.encoder(x_d, size=bound)

        # sigma

        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.relu(h[..., 0])
        # sigma[x[:,0]<-0.9]=0
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

    def deform(self,x,frame_id , bound = 1):
        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        h = self.encoder(x, size=bound)
        # deform 
        frame_id = torch.ones((x.size(0),1), device=x.device) * frame_id /self.num_frames
        frame_id=frame_id*2-1# nomalized in [-1, 1]
        x_t=torch.cat([x,frame_id],dim=-1)
        h = self.encoder_deform(x_t, size = bound)
        for l in range(self.num_layers):
            h = self.deform_net[l](h)
            if l!= self.num_layers - 1:
                h = F.relu(h,inplace = True)
        
        delta_x = torch.sigmoid(h) - 0.5 # in [-1,1]
        x_d = torch.clamp(x + delta_x,min = -1,max = 1)
        return x_d
    
    def check_warp(self, x):
        if self.smpl_vertices!= None:
            x[:,[0,1,2]] = x[:,[2,0,1]]
            x = x * 3
            # print(x,smpl_vertices)
            # print(x.shape, smpl_vertices.shape)
            # np.savetxt('results/raw.txt', x.detach().cpu().numpy())
            x_out = self.smpl_model.mylbs_inv_fast(x.unsqueeze(0), self.smpl_vertices, self.body_params, self.pack)
            # np.savetxt('results/can.txt', x_out[0].detach().cpu().numpy())
            x[x_out[0]<-2.4] = -2.9
            # np.savetxt('results/live.txt', x.detach().cpu().numpy())
            # print(x.shape)
            # while True:
            #     print(1)
            x = x_out[0]/3  
            # x = x/3  
            # while True:
            #     print(1)
            x[:,[2,0,1]] = x[:,[0,1,2]]
        elif self.knn_field != None:

            # st = time.time()
            x[:,[0,1,2]] = x[:,[2,0,1]]
            x = x * 2
            coords_out = svr.warp_vertices(x, self.ed_nodes, 
                                                self.ed_motions, 
                                                self.knn_field,
                                                self.weight_field, 
                                                self.flag_field, 
                                                self.res,self.left_top,self.leng)
            torch.cuda.synchronize()
            coords_out = coords_out[0]   
            x = coords_out   
            x = x/2
            x[:,[2,0,1]] = x[:,[0,1,2]]
        elif self.ed_nodes != None:
            x[:,[0,1,2]] = x[:,[2,0,1]]
            x = x * 1.5
            # tmp = x.clone()
            coords_out, R_out = svr.warp_vertices_one_stage(x, self.ed_nodes, 
                                                        self.ed_motions, 
                                                       self.res,self.left_top,self.leng)
            torch.cuda.synchronize()
            # coords_out = coords_out[0]  
            # x[coords_out<-2.5]=-2.9
            # np.savetxt('can_%d.txt'%self.save, coords_out[coords_out>-2.5].reshape(-1,3).detach().cpu().numpy())
            # np.savetxt('live_%d.txt'%self.save, x[x>-2.5].reshape(-1,3).detach().cpu().numpy())
            self.save += 1
            # print('sum:', torch.abs(coords_out[coords_out>-2.5]-x[coords_out>-2.5]).shape)
            x = coords_out
            self.mask = torch.ones([coords_out.shape[0],1]).to(x.device)
            # print( self.mask.shape, coords_out.shape)
            self.mask[coords_out[:,:1]<-self.res[0]+0.25 ] = 0
            # tmp = x.clone() * self.mask
            # np.savetxt('results/mask.txt', tmp.detach().cpu().numpy())
            # if (torch.isnan(x).any() == True):
            #     np.savetxt('after.txt', x.detach().cpu().numpy())
            #     np.savetxt('before.txt', tmp.detach().cpu().numpy())
    
            x = x / 1.5
            x[:,[2,0,1]] = x[:,[0,1,2]]
            self.R = R_out
        return x