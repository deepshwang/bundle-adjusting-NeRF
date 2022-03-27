import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
import camera
from util import log, debug


class Dataset(base.Dataset):

    def __init__(self, opt, split="train", subset=None):
        super().__init__(opt, split)
        self.raw_H, self.raw_W = self.opt.data.raw_image_size
        self.root = opt.data.root or "data/blender"
        self.path = "{}/{}".format(self.root, opt.data.scene)
        self.focal = opt.camera.focal
        self.imgfiles = self.get_imagefiles()
        if subset: self.imgfiles = self.imgfiles[:subset]
        self.bg_imgfile = self.path + "/background.jpg" if os.path.exists(self.path+"/background.jpg") else None
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt, self.get_image)
            self.bg_image = self.get_bg_image(opt)

    def prefetch_all_data(self, opt):
        assert (not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    # def get_all_camera_poses(self,opt):
    #     pose_raw_all = [torch.tensor(f["transform_matrix"],dtype=torch.float32) for f in self.list]
    #     pose_canon_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
    #     return pose_canon_all

    def __getitem__(self, idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt, idx)
        image = self.preprocess_image(opt, image, aug=aug)
        if self.bg_image is not None:
            bg_image = self.preprocess_image(opt, self.bg_image, aug=aug)
            obj_mask = torch.where(torch.abs(torch.norm(image - bg_image, dim=0)) > self.opt.scannerf.mask_thresh,
                                   self.opt.scannerf.obj_weight,
                                   self.opt.scannerf.bg_weight)
            obj_mask = obj_mask[None, ...]
        intr = self.get_camera(opt)
        intr= self.preprocess_camera(opt, intr, pose=None, aug=aug)
        if bg_image is not None:
            sample.update(image=image, bg_image=bg_image, obj_mask=obj_mask, intr=intr)
        else:
            sample.update(image=image, intr=intr)
        return sample

    def get_imagefiles(self):
        imgfiles = [os.path.join(self.path, self.split, f) for f in\
                    sorted(os.listdir(os.path.join(self.path, self.split))) if
                    f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        return imgfiles

    def get_image(self, opt, idx):
        image = PIL.Image.fromarray(
            imageio.imread(self.imgfiles[idx]))  # directly using PIL.Image.open() leads to weird corruption....
        return image
    def get_bg_image(self, opt):
        if self.bg_imgfile is not None:
            bg_image = PIL.Image.fromarray(imageio.imread(self.bg_imgfile))
        else:
            bg_image = None
        return bg_image


    def preprocess_image(self, opt, image, aug=None):
        image = super().preprocess_image(opt, image, aug=aug)
        return image

    def get_camera(self, opt):
        intr = torch.tensor([[self.focal, 0, self.raw_W / 2],
                             [0, self.focal, self.raw_H / 2],
                             [0, 0, 1]]).float()
        return intr

    def parse_raw_camera(self, opt, pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1, -1, -1])))
        pose = camera.pose.compose([pose_flip, pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose
