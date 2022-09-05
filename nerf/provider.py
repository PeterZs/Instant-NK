from operator import index
import os
import time
import glob
from turtle import down
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.spatial.transform import Slerp, Rotation

# NeRF dataset
import json


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33,offset=[0.,0.,0.]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
        [0, 0, 0, 1],
    ],dtype=np.float32)
    return new_pose

# def nerf_matrix_to_ngp(pose, scale=0.33,offset=[0.,0.,0.]):
#     # for the fox dataset, 0.33 scales camera radius to ~ 2
#     new_pose = np.array([
#         [pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3] * scale + offset[0]],
#         [pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3] * scale + offset[1]],
#         [pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3] * scale + offset[2]],
#         [0, 0, 0, 1],
#     ],dtype=np.float32)
#     return new_pose

class NeRFDataset(Dataset):
    def __init__(self, path, type='train', mode='colmap', preload=True, downscale=1, scale=0.33, n_test=10):
        super().__init__()
        # path: the json file path.

        self.root_path = path
        self.type = type # train, val, test
        self.mode = mode # colmap, blender, llff, nhr
        self.downscale = downscale
        self.preload = preload # preload data into GPU

        # camera radius scale to make sure camera are inside the bounding box.
        self.scale = scale

        # load nerf-compatible format data.
        if mode == 'colmap' or mode == 'nhr':
            transform_path = os.path.join(path, 'transforms.json')
        elif mode == 'blender':
            transform_path = os.path.join(path, f'transforms_{type}.json')
        else:
            raise NotImplementedError(f'unknown dataset mode: {mode}')

        with open(transform_path, 'r') as f:
            transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])    
        if mode == 'nhr':
            aabb = transform['aabb']
            length = max(0.000001, max(
                max(abs(float(aabb[1][0]) - float(aabb[0][0])), abs(float(aabb[1][1]) - float(aabb[0][1]))),
                abs(float(aabb[1][2]) - float(aabb[0][2]))))
            scale = 1 / length *2
            offset = [((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) * -scale,
                      ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) *
                      -scale, ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5) * -scale]
            print("scale ", scale)
            print("offset ", offset)
        else:
            offset=[0.,0.,0.]

        # for colmap, manually interpolate a test set.
        if (mode == 'colmap' or mode == 'nhr')and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=scale,offset=offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=scale,offset=offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            if mode == 'nhr':
                self.Ks = []
                Ks = np.array(f0['K'], dtype=np.float32)  # [3, 3]
                Ks = Ks / downscale
                Ks[2, 2] = 1.

            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
                if mode=='nhr':
                    self.Ks.append(Ks)
        else:
            # for colmap, manually split a valid set (the first frame).
            if mode == 'colmap'  or mode == 'nhr':
                frames = frames[1:] if type == 'train' else frames[:1]
            
            self.poses = []
            self.images = []
            self.Ks=[]
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if mode == 'blender':
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                if mode=='nhr':
                    pose = nerf_matrix_to_ngp(pose, scale=scale, offset=offset)
                else:
                    pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                if mode == 'nhr':
                    Ks = np.array(f['K'], dtype=np.float32)  # [3, 3]
                    Ks = Ks / downscale
                    Ks[2, 2] = 1.
                    # print(Ks)
                    self.Ks.append(Ks)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = np.stack(self.poses, axis=0)
        self.images = np.stack(self.images, axis=0)
        if mode == 'nhr':
            self.Ks = np.stack(self.Ks, axis=0).astype(np.float32)

        cam_poses = self.poses[:, :3, 3]
        print("cam poses min", np.min(cam_poses, 0))
        print("cam poses max", np.max(cam_poses, 0))

        if preload:
            self.poses = torch.from_numpy(self.poses).cuda()
            self.images = torch.from_numpy(self.images).cuda()

        # load intrinsics
        if mode != 'nhr':
        
            if 'fl_x' in transform or 'fl_y' in transform:
                fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
                fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
            elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
                # blender, assert in radians. already downscaled since we use H/W
                fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
                fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
                if fl_x is None: fl_x = fl_y
                if fl_y is None: fl_y = fl_x
            else:
                raise RuntimeError('cannot read focal!')

            cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
            cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)

            self.intrinsic = np.eye(3, dtype=np.float32)
            self.intrinsic[0, 0] = fl_x
            self.intrinsic[1, 1] = fl_y
            self.intrinsic[0, 2] = cx
            self.intrinsic[1, 2] = cy

            if preload:
                self.intrinsic = torch.from_numpy(self.intrinsic).cuda()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        if self.mode=='nhr':
            results = {
                'pose': self.poses[index],
                'intrinsic': self.Ks[index],
                'index': index,
                #'num_cameras': len(self.poses),
            }
        else:
            results = {
                'pose': self.poses[index],
                'intrinsic': self.intrinsic,
                'index': index,
            }

        if self.type == 'test':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            #results['num_cameras'] = len(self.poses)
            return results
        else:
            results['image'] = self.images[index]
            return results