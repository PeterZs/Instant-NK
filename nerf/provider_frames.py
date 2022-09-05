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

class NeRFDatasetFrames(Dataset):
    def __init__(self, path, type='train', mode='nhr', preload=True, downscale=1, n_test=10, num_frame=2,st_frame = 1):
        super().__init__()
        # path: the json file path.

        self.root_path = path
        self.type = type # train, val, test
        self.mode = mode # colmap, blender, llff, nhr
        self.downscale = downscale
        self.preload = preload # preload data into GPU
        self.num_frame= num_frame

        memoryplace_holder = torch.Tensor(2000, 2000, 300).cuda()
        print('start prepare')
        # load nerf-compatible format data.

        if path.find('json')!=-1:
            self.root_path = self.root_path[:self.root_path.rfind('/')]
            transform_path = path
        else:
            transform_path = os.path.join(path, 'transforms.json')

        with open(transform_path, 'r') as f:
            transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        file_folder = transform["file_folder"]
        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])    

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
        # for colmap, manually interpolate a test set.
        if type == 'test':
            image_idx=0
            self.poses = []
            self.images = []
            self.Ks = []
            self.cam_num = len(frames)
            self.frame_ids=[]
            for f in frames:
                file_path = f['file_path']
                time_stamp= int(image_idx % self.num_frame)
                image_idx+=1
                self.frame_ids.append(time_stamp)

                pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]

                pose = nerf_matrix_to_ngp(pose, scale=scale, offset=offset)

                Ks = np.array(f['K'], dtype=np.float32)  # [3, 3]
                Ks = Ks / downscale
                Ks[2, 2] = 1.
                # print(Ks)
                self.Ks.append(Ks)
                self.poses.append(pose)

            self.poses = np.stack(self.poses, axis=0)
            self.Ks = np.stack(self.Ks, axis=0).astype(np.float32)
            # for i in range(len(self.poses)):
            #     self.poses[i] = np.array( [
            #     [
            #         0.1410253793001175,
            #         7.209580843436925e-08,
            #         -0.9900059700012207,
            #         -3.662359618252415
            #     ],
            #     [
            #         -0.9900059700012207,
            #         -1.6838826866205636e-07,
            #         -0.1410253793001175,
            #         -0.4615644436892796
            #     ],
            #     [
            #         -1.8258003819937585e-07,
            #         1.0,
            #         5.2580183762529487e-08,
            #         -0.19999999176001101
            #     ],
            #     [
            #         0.0,
            #         0.0,
            #         0.0,
            #         1.0
            #     ]
            # ],dtype=np.float32)
            #     self.Ks[i] = np.array([[
            #             2844.4444444444443,
            #             0.0,
            #             1024.0
            #         ],
            #         [
            #             0.0,
            #             2844.4444444444443,
            #             768.0
            #         ],
            #         [
            #             0.0,
            #             0.0,
            #             1.0
            #         ]
            #     ],dtype=np.float32)
            #     self.poses[i] = nerf_matrix_to_ngp(self.poses[i], scale=scale, offset=offset)
        else:
            # for colmap, manually split a valid set (the first frame).
            frames = frames[1:] if type == 'train' else frames[:1]
            
            self.poses = []
            self.images = []
            self.Ks=[]
            self.cam_num=len(frames)
            for time_stamp in range(self.num_frame):
                print(f"reading images from time_stamp {time_stamp}")
                for f in frames:
                    file_path=f['file_path']
                    f_path = os.path.join(self.root_path, file_folder ,"%d/"%(st_frame+time_stamp), file_path)
                   

                    # there are non-exist paths in fox...
                    if not os.path.exists(f_path):
                        print(f_path)
                        continue

                    pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                    pose = nerf_matrix_to_ngp(pose, scale=scale, offset=offset)

                    Ks = np.array(f['K'], dtype=np.float32)  # [3, 3]
                    Ks = Ks / downscale
                    Ks[2, 2] = 1.
                    # print(Ks)
                    self.Ks.append(Ks)

                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    # print(image.shape)
                    # image = image[:,:,:3]
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
            # if type != 'train':
            #     for i in range(len(self.poses)):
            #         self.poses[i] = np.array([
            #             [
            #                 0.751232922077179,
            #                 2.9728743911050515e-08,
            #                 -0.6600371599197388,
            #                 -3.3592144529522017
            #             ],
            #             [
            #                 -0.6600371599197388,
            #                 1.1103404062851041e-07,
            #                 -0.7512328624725342,
            #                 -3.943347520976084
            #             ],
            #             [
            #                 2.9728743911050515e-08,
            #                 0.9999999403953552,
            #                 1.1103404062851041e-07,
            #                 -0.49999986957057035
            #             ],
            #             [
            #                 0.0,
            #                 0.0,
            #                 0.0,
            #                 1.0
            #             ]
            #         ],dtype=np.float32)
            #         self.poses[i] = nerf_matrix_to_ngp(self.poses[i], scale=scale, offset=offset)

            self.poses = np.stack(self.poses, axis=0)
            #self.images = np.stack(self.images, axis=0)
            # cost too long and too much memory
            self.Ks = np.stack(self.Ks, axis=0).astype(np.float32)

            cam_poses = self.poses[:, :3, 3]
            print("cam poses min", np.min(cam_poses, 0))
            print("cam poses max", np.max(cam_poses, 0))

        del memoryplace_holder
        torch.cuda.empty_cache()


    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.Ks[index],
            'index': index,
            'frame_id': index // self.cam_num,
        }

        if self.type == 'test':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            results['frame_id'] = self.frame_ids[index]
            return results
        else:
            results['image'] = self.images[index]
            return results