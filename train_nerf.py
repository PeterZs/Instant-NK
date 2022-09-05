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
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--upsample_steps', type=int, default=256)
    parser.add_argument('--max_ray_batch', type=int, default=4096*3)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    parser.add_argument('--mode', type=str, default='nh', help="dataset mode, supports (colmap, blender,nhr)")

    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")

    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")

    parser.add_argument('--color_dim', type=int, default=64, help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--st_frame',type=int, default=1, help="start frame")
    parser.add_argument('--num_frames',type=int, default=1, help="num of frames used for nhr")
    parser.add_argument('--num_image',type=int, default=1, help="camera number")

    parser.add_argument('--ed_folder',type = str, default = '/new_disk/jyh/animationMesh/eval/sequence/jyh0809' ,help="non-rigid information")
    parser.add_argument('--dyna_mode', type=str, default='deform', help="dynamic mode, supports (deform , fourier, time)")
    parser.add_argument('--hashnerf', action='store_true', help="use pure pytorch")	
    parser.add_argument('--freq', action='store_true', help="use freq pytorch")
    parser.add_argument('--test', action='store_true', help="test")
    parser.add_argument('--smpl', action='store_true', help="test")

    opt = parser.parse_args()

    print(opt)

    
    if opt.dyna_mode == "time":
        from nerf.network_time import NeRFNetwork
        optimizer = lambda model: torch.optim.Adam([	
        {'name': 'encoding', 'params': list(model.encoder.parameters())},	
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)
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

    
    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics",
        num_layers=2,  geo_feat_dim=15, num_layers_color=3, hidden_dim_color=opt.color_dim,
        num_frames=opt.num_frames,
        cuda_ray=opt.cuda_ray,
    )
    print(model)
    criterion = torch.nn.SmoothL1Loss()
    seed_everything(opt.seed)
    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)

    trainer = Trainer('ngp', vars(opt), model, st_frame = opt.st_frame, num_image = opt.num_image, ed_folder = opt.ed_folder, smpl_option = opt.smpl, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=2)
    if opt.test==False:
        train_dataset = NeRFDatasetFrames(opt.path, type='train', mode=opt.mode,num_frame=opt.num_frames,st_frame = opt.st_frame)
        valid_dataset = NeRFDatasetFrames(opt.path, type='val', mode=opt.mode, downscale=2,num_frame=opt.num_frames,st_frame = opt.st_frame)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
        trainer.train(train_loader, valid_loader, 1000)

    else:
        test_dataset = NeRFDatasetFrames(opt.path, type='test', mode=opt.mode,num_frame=opt.num_frames,st_frame = opt.st_frame)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        trainer.test(test_loader,save_path=opt.test_save_path)

    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)

    # test dataset
    #trainer.save_mesh()


