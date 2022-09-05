# train:
# OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-11.1  CUDA_VISIBLE_DEVICES=1 python train_nerf.py /new_disk/jyh/datasets/denseView/zhengshuang/output/500700 --workspace zhengshuang1 --fp16 --mode nhr  --bound 1 --num_steps 256 --upsample_steps 256 --st_frame 560 --num_frames 60 --dyna_mode time
# test:
#OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-11.1  CUDA_VISIBLE_DEVICES=3 python test_nerf.py /new_disk/jyh/datasets/denseView/zhengshuang/output/fvv --workspace zhengshuang_svr --mode nhr  --bound 1 --num_steps 256 --upsample_steps 256 --st_frame 560 --num_frames 60 --dyna_mode time

# OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-11.1  CUDA_VISIBLE_DEVICES=3 python test_nerf.py /new_disk/jyh/datasets/denseView/spider2 --workspace spider2_can7  --fp16 --mode nhr  --bound 1 --num_steps 256 --upsample_steps 256 --st_frame 30 --num_frames 35 --num_image 90 --dyna_mode deform


import torch

from nerf.provider import NeRFDataset
from nerf.provider_frames import NeRFDatasetFrames
from nerf.utils import *

import argparse

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096*3)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096*3)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender,nhr)")
    # the default setting for fox.
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")

    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")

    parser.add_argument('--color_dim', type=int, default=64, help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--st_frame',type=int, default=1, help="start frame")
    parser.add_argument('--num_frames',type=int, default=1, help="num of frames used for nhr")
    parser.add_argument('--num_image',type=int, default=1, help="num of frames used for nhr")

    parser.add_argument('--dyna_mode', type=str, default='deform', help="dynamic mode, supports (deform , fourier, time)")
    parser.add_argument('--hashnerf', action='store_true', help="use pure pytorch")	
    parser.add_argument('--freq', action='store_true', help="use freq pytorch")
    parser.add_argument('--test_save_path', type=str, default='result', help="test_save_path")
    opt = parser.parse_args()

    print(opt)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        if opt.num_frames>1:
            if opt.dyna_mode=="fourier":
                from nerf.network_ff_fourrier import NeRFNetwork
            else:
                from nerf.network_ff_frames import NeRFNetwork
        else:
            from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        if opt.num_frames > 1:
            from nerf.network_tcnn_deform import NeRFNetwork
        else:	
            from nerf.network_tcnn import NeRFNetwork	
    elif opt.hashnerf:
        if opt.num_frames > 1:
            from nerf.network_deform_hashnerf import NeRFNetwork
        else:
            from nerf.network_hashnerf import NeRFNetwork
    elif opt.freq:
        if opt.num_frames > 1:
            from nerf.network_deform_nerf import NeRFNetwork
            optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-2, betas = (0.9,
                                                                                     0.99), weight_decay = 1e-6,eps=1e-15)
    else:
        if opt.num_frames > 1:
            if opt.dyna_mode == "fourier":
                from nerf.network_fourier import NeRFNetwork
            elif opt.dyna_mode == "time":
                from nerf.network_time import NeRFNetwork
            else:
                from nerf.network_time_deform import NeRFNetwork
                optimizer = lambda model: torch.optim.Adam([	
                    {'name': 'encoding',	
                     'params': list(model.encoder.parameters()) + list(model.encoder_deform.parameters())},	
                    {'name': 'net',	
                     'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()) + list(	
                         model.deform_net.parameters()),	
                     'weight_decay': 1e-6},	
                ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

                # optimizer = lambda model: torch.optim.Adam([	
                #     {'name': 'encoding',	
                #      'params':  list(model.encoder_deform.parameters())},	
                #     {'name': 'net',	
                #      'params':  list( model.deform_net.parameters()),	
                #      'weight_decay': 1e-6},	
                # ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)
        else:
            from nerf.network import NeRFNetwork

    
    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics",
        num_layers=2,  geo_feat_dim=15, num_layers_color=3, hidden_dim_color=opt.color_dim,
        num_frames=opt.num_frames,
        cuda_ray=opt.cuda_ray,
    )
    print(model)
    criterion = torch.nn.SmoothL1Loss()
    if opt.dyna_mode != "deform":	
        optimizer = lambda model: torch.optim.Adam([	
            {'name': 'encoding', 'params': list(model.encoder.parameters())},	
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},	
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    seed_everything(opt.seed)
    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)


    test_dataset = NeRFDatasetFrames(opt.path, type='test', mode=opt.mode,num_frame=opt.num_frames,st_frame = opt.st_frame)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)


    trainer = Trainer('ngp', vars(opt), model, st_frame = opt.st_frame, num_image = opt.num_image, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')
    
    trainer.test(test_loader,save_path=opt.test_save_path)


# import torch

# from nerf.provider_frames import NeRFDatasetFrames
# from nerf.utils import *

# import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('path', type=str)
#     parser.add_argument('--workspace', type=str, default='workspace')
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--num_rays', type=int, default=4096*3)
#     parser.add_argument('--num_steps', type=int, default=128)
#     parser.add_argument('--upsample_steps', type=int, default=128)
#     parser.add_argument('--max_ray_batch', type=int, default=4096*3)
#     parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
#     parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
#     parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

#     parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender,nhr)")
#     # the default setting for fox.
#     parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
#     parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")

#     parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")

#     parser.add_argument('--color_dim', type=int, default=64, help="use CUDA raymarching instead of pytorch")
#     parser.add_argument('--st_frame',type=int, default=1, help="start frame")
#     parser.add_argument('--num_frames',type=int, default=1, help="num of frames used for nhr")
#     parser.add_argument('--num_image',type=int, default=1, help="num of frames used for nhr")

#     parser.add_argument('--dyna_mode', type=str, default='deform', help="dynamic mode, supports (deform , fourier, time)")
#     parser.add_argument('--hashnerf', action='store_true', help="use pure pytorch")	
#     parser.add_argument('--freq', action='store_true', help="use freq pytorch")
#     parser.add_argument('--test_save_path', type=str, default='result', help="test_save_path")

#     opt = parser.parse_args()

#     print(opt)

#     if opt.ff:
#         assert opt.fp16, "fully-fused mode must be used with fp16 mode"
#         if opt.num_frames>1:
#             if opt.dyna_mode=="fourier":
#                 from nerf.network_ff_fourrier import NeRFNetwork
#             else:
#                 from nerf.network_ff_frames import NeRFNetwork
#         else:
#             from nerf.network_ff import NeRFNetwork
#     elif opt.tcnn:
#         from nerf.network_tcnn import NeRFNetwork
#     else:
#         if opt.num_frames > 1:
#             if opt.dyna_mode == "fourier":
#                 from nerf.network_fourier import NeRFNetwork
#             elif opt.dyna_mode == "time":
#                 from nerf.network_time import NeRFNetwork
#             else:
#                 from nerf.network_time_deform import NeRFNetwork
#         else:
#             from nerf.network import NeRFNetwork
        
#     seed_everything(opt.seed)

#     model = NeRFNetwork(
#         encoding="hashgrid", encoding_dir="sphere_harmonics", 
#         num_layers=2,  geo_feat_dim=15, num_layers_color=3, hidden_dim_color=opt.color_dim,num_frames=opt.num_frames,
#         cuda_ray=opt.cuda_ray,
#     )

#     print(model)
    
#     trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')

#     # save mesh
#     #trainer.save_mesh()
#     test_dataset = NeRFDatasetFrames(opt.path, type='test', mode=opt.mode,num_frame=opt.num_frames,st_frame = opt.st_frame)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
#     trainer.test(test_loader,save_path=opt.test_save_path)

