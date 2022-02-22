import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import visdom
import matplotlib.pyplot as plt
from math import log2
from kornia.filters import filter2D

import util, util_vis
from util import log, debug
from . import nerf
from . import base
import camera
import itertools


# ============================ main engine for training and evaluation ============================

class Model(nerf.Model):

    def __init__(self, opt):
        super().__init__(opt)

    def build_networks(self, opt):
        super().build_networks(opt)  # Initialize -> self.graph = model.{opt.model}.Graph()
        self.graph.se3_refine = torch.nn.Embedding(len(self.train_data), 6).to(opt.device)
        torch.nn.init.zeros_(self.graph.se3_refine.weight)

    def setup_optimizer(self, opt):
        """
        Parameters for optimization
            - NeRF model: 2 * (N_obj + 1) number of models
            - Pose parameters: (6 * N_obj * number of frames) number of parameters
            - Latent code: N_obj * dim_latent
        """
        # NeRF network optimizers
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim, opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.se3_refine.parameters(), lr=opt.optim.lr_renderer)])
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
            if opt.optim.lr_end:
                assert (opt.optim.sched.type == "ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end / opt.optim.lr) ** (1. / opt.max_iter)
            kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
            self.sched = scheduler(self.optim, **kwargs)

        # # Neural renderer optimizers
        # optimizer = getattr(torch.optim, opt.optim.algo)
        # self.optim_renderer = optimizer([dict(params=self.graph.se3_refine.parameters(), lr=opt.optim.lr_pose)])
        # # set up scheduler
        # if opt.optim.sched_pose:
        #     scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched_pose.type)
        #     if opt.optim.lr_pose_end:
        #         assert (opt.optim.sched_pose.type == "ExponentialLR")
        #         opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end / opt.optim.lr_pose) ** (1. / opt.max_iter)
        #     kwargs = {k: v for k, v in opt.optim.sched_pose.items() if k != "type"}
        #     self.sched_pose = scheduler(self.optim_renderer, **kwargs)

        # Pose optimizers
        optimizer = getattr(torch.optim, opt.optim.algo)
        self.optim_pose = optimizer([dict(params=self.graph.se3_refine.parameters(), lr=opt.optim.lr_pose)])
        # set up scheduler
        if opt.optim.sched_pose:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched_pose.type)
            if opt.optim.lr_pose_end:
                assert (opt.optim.sched_pose.type == "ExponentialLR")
                opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end / opt.optim.lr_pose) ** (1. / opt.max_iter)
            kwargs = {k: v for k, v in opt.optim.sched_pose.items() if k != "type"}
            self.sched_pose = scheduler(self.optim_pose, **kwargs)

        # Latent code optimizers
        optimizer = getattr(torch.optim, opt.optim.algo)
        self.optim_latent = optimizer([dict(params=self.graph.latent.parameters(), lr=opt.optim.lr_latent)])
        if opt.optim.sched_latent:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched_latent.type)
            if opt.optim.lr_latent_end:
                assert (opt.optim.sched_latent.type == "ExponentialLR")
                opt.optim.sched_latent.gamma = (opt.optim.lr_latent_end / opt.optim.lr_latent) ** (1. / opt.max_iter)
            kwargs = {k: v for k, v in opt.optim.sched_latent.items() if k != "type"}
            self.sched_latent = scheduler(self.optim_latent, **kwargs)

    # WORKING
    def train_iteration(self, opt, var, loader):
        self.optim_pose.zero_grad()
        if opt.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.optim_pose.param_groups[0]["lr_orig"] = self.optim_pose.param_groups[0][
                "lr"]  # cache the original learning rate
            self.optim_pose.param_groups[0]["lr"] *= min(1, self.it / opt.optim.warmup_pose)
        loss = super().train_iteration(opt, var, loader)
        self.optim_pose.step()
        if opt.optim.warmup_pose:
            self.optim_pose.param_groups[0]["lr"] = self.optim_pose.param_groups[0]["lr_orig"]  # reset learning rate
        if opt.optim.sched_pose: self.sched_pose.step()
        self.graph.nerf.progress.data.fill_(self.it / opt.max_iter)
        if opt.nerf.fine_sampling:
            self.graph.nerf_fine.progress.data.fill_(self.it / opt.max_iter)
        return loss

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(opt, var, loss, metric=metric, step=step, split=split)
        if split == "train":
            # log learning rate
            lr = self.optim_pose.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split, "lr_pose"), lr, step)
        # compute pose error
        if split == "train" and opt.data.dataset in ["blender", "llff"]:
            pose, pose_GT = self.get_all_training_poses(opt)
            pose_aligned, _ = self.prealign_cameras(opt, pose, pose_GT)
            error = self.evaluate_camera_alignment(opt, pose_aligned, pose_GT)
            self.tb.add_scalar("{0}/error_R".format(split), error.R.mean(), step)
            self.tb.add_scalar("{0}/error_t".format(split), error.t.mean(), step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        super().visualize(opt, var, step=step, split=split)
        if opt.visdom:
            if split == "val":
                pose, pose_GT = self.get_all_training_poses(opt)
                util_vis.vis_cameras(opt, self.vis, step=step, poses=[pose, pose_GT])

    @torch.no_grad()
    def get_all_training_poses(self, opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        # add synthetic pose perturbation to all training data
        if opt.data.dataset == "blender":
            pose = pose_GT
            if opt.camera.noise:
                pose = camera.pose.compose([self.graph.pose_noise, pose])
        else:
            pose = self.graph.pose_eye
        # add learned pose correction to all training data
        pose_refine = camera.lie.se3_to_SE3(self.graph.se3_refine.weight)  # SE3: (3, 4) matrix
        pose = camera.pose.compose([pose_refine, pose])
        return pose, pose_GT

    @torch.no_grad()
    def prealign_cameras(self, opt, pose, pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = torch.zeros(1, 1, 3, device=opt.device)
        center_pred = camera.cam2world(center, pose)[:, 0]  # [N,3]
        center_GT = camera.cam2world(center, pose_GT)[:, 0]  # [N,3]
        try:
            sim3 = camera.procrustes_analysis(center_GT, center_pred)
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=opt.device))
        # align the camera poses
        center_aligned = (center_pred - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
        R_aligned = pose[..., :3] @ sim3.R.t()
        t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
        pose_aligned = camera.pose(R=R_aligned, t=t_aligned)
        return pose_aligned, sim3

    @torch.no_grad()
    def evaluate_camera_alignment(self, opt, pose_aligned, pose_GT):
        # measure errors in rotation and translation
        R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
        R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
        R_error = camera.rotation_distance(R_aligned, R_GT)
        t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
        error = edict(R=R_error, t=t_error)
        return error

    @torch.no_grad()
    def evaluate_full(self, opt):
        self.graph.eval()
        # evaluate rotation/translation
        pose, pose_GT = self.get_all_training_poses(opt)
        pose_aligned, self.graph.sim3 = self.prealign_cameras(opt, pose, pose_GT)
        error = self.evaluate_camera_alignment(opt, pose_aligned, pose_GT)
        print("--------------------------")
        print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        print("trans: {:10.5f}".format(error.t.mean()))
        print("--------------------------")
        # dump numbers
        quant_fname = "{}/quant_pose.txt".format(opt.output_path)
        with open(quant_fname, "w") as file:
            for i, (err_R, err_t) in enumerate(zip(error.R, error.t)):
                file.write("{} {} {}\n".format(i, err_R.item(), err_t.item()))
        # evaluate novel view synthesis
        super().evaluate_full(opt)

    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self, opt, var):
        # use another se3 Parameter to absorb the remaining pose errors
        var.se3_refine_test = torch.nn.Parameter(torch.zeros(1, 6, device=opt.device))
        optimizer = getattr(torch.optim, opt.optim.algo)
        optim_pose = optimizer([dict(params=[var.se3_refine_test], lr=opt.optim.lr_pose)])
        iterator = tqdm.trange(opt.optim.test_iter, desc="test-time optim.", leave=False, position=1)
        for it in iterator:
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.forward(opt, var, mode="test-optim")
            loss = self.graph.compute_loss(opt, var, mode="test-optim")
            loss = self.summarize_loss(opt, var, loss)
            loss.all.backward()
            optim_pose.step()
            iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return var

    @torch.no_grad()
    def generate_videos_pose(self, opt):
        self.graph.eval()
        fig = plt.figure(figsize=(10, 10) if opt.data.dataset == "blender" else (16, 8))
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path, exist_ok=True)
        ep_list = []
        for ep in range(0, opt.max_iter + 1, opt.freq.ckpt):
            # load checkpoint (0 is random init)
            if ep != 0:
                try:
                    util.restore_checkpoint(opt, self, resume=ep)
                except:
                    continue
            # get the camera poses
            pose, pose_ref = self.get_all_training_poses(opt)
            if opt.data.dataset in ["blender", "llff"]:
                pose_aligned, _ = self.prealign_cameras(opt, pose, pose_ref)
                pose_aligned, pose_ref = pose_aligned.detach().cpu(), pose_ref.detach().cpu()
                dict(
                    blender=util_vis.plot_save_poses_blender,
                    llff=util_vis.plot_save_poses,
                )[opt.data.dataset](opt, fig, pose_aligned, pose_ref=pose_ref, path=cam_path, ep=ep)
            else:
                pose = pose.detach().cpu()
                util_vis.plot_save_poses(opt, fig, pose, pose_ref=None, path=cam_path, ep=ep)
            ep_list.append(ep)
        plt.close()
        # write videos
        print("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname, "w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system(
            "ffmpeg -y -r 30 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname, cam_vid_fname))
        os.remove(list_fname)


# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self, opt):
        super().__init__(opt)
        self.N_obj = opt.scannerf.N_obj
        self.pose_eye = torch.eye(3, 4).to(opt.device)
        self.obj_pose_init = torch.eye(3, 4).to(opt.device)
        # In ScanNeRF, camera pose is set as global pose (eye(4))
        self.world_pose = torch.eye(3, 4).to(opt.device)
        ### Models / trainable parameters ###
        self.scannerf = torch.nn.ModuleList(2 * [nerf.NeRF(opt)] + 2 * self.N_obj * [CondNeRF(opt)])
        # self.renderer = NeuralRenderer()
        self.latent = torch.nn.Embedding(self.N_obj, opt.arch.dim_latent).to(opt.device)


    def forward(self, opt, var, mode=None):
        # var: dataloader output, easydict type
        batch_size = opt.batch_size
        pose = self.get_pose(opt, var, mode=mode)
        # render images
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            # sample random rays for optimization
            var.ray_idx = torch.randperm(opt.H * opt.W, device=opt.device)[:opt.nerf.rand_rays//batch_size]
            ret = self.render(opt, pose, self.latent, intr=var.intr, ray_idx=var.ray_idx, mode=mode) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(opt, pose, self.latent, intr=var.intr,mode=mode) if opt.nerf.rand_rays else \
                  self.render(opt, pose, self.latent, intr=var.intr, mode=mode) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def render(self, opt, pose, latent, intr=None, ray_idx=None, mode=None):
        batch_size = opt.batch_size
        center, ray = camera.get_center_and_ray(opt, self.world_pose.expand(batch_size, -1, -1), intr=intr)  # [B,HW,3]
        while ray.isnan().any():  # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center, ray = camera.get_center_and_ray(opt, self.world_pose.expand(batch_size, -1, -1),
                                                    intr=intr)  # [B,HW,3]
        if ray_idx is not None:
            # consider only subset of rays
            center, ray = center[:, ray_idx], ray[:, ray_idx]  # [B, n_rays, 3]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center, ray = camera.convert_NDC(opt, center, ray, intr=intr)

        # WORKING => condnerf.forward_samples
        # Compositional Rendering with learned-pose adjusted inference
        depth_samples = self.sample_depth(opt, batch_size, num_rays=ray.shape[1])  # [B, n_rays, n_samples, 1]
        composite_rgb_samples = 0.0
        composite_density_samples = 0.0
        composite_rgb_samples_fine = 0.0
        composite_density_samples_fine = 0.0

        for i in range(self.N_obj + 1):
            # Background object: normal NeRF rendering
            if i == 0:
                rgb_samples, density_samples = self.scannerf[2*i].forward_samples(opt, center, ray, depth_samples,
                                                                                mode=mode)

            # Foreground objects: bundle adjusting pose and conditional nerf
            else:
                latent = self.latent.weight[i-1]
                rgb_samples, density_samples = self.scannerf[2*i].forward_samples(opt, center, ray,
                                                                                depth_samples, latent, pose, mode=mode)

            composite_rgb_samples += rgb_samples / (self.N_obj + 1)
            composite_density_samples += density_samples / (self.N_obj + 1)
            prob = self.composite(opt, ray, rgb_samples, density_samples, depth_samples, prob_only=True)

            if opt.nerf.fine_sampling:
                with torch.no_grad():
                    # resample depth acoording to coarse empirical distribution
                    depth_samples_fine = self.sample_depth_from_pdf(opt, pdf=prob[..., 0])  # [B,HW,Nf,1]
                    depth_samples = torch.cat([depth_samples, depth_samples_fine], dim=2)  # [B,HW,N+Nf,1]
                    depth_samples = depth_samples.sort(dim=2).values
                    if i == 0:
                        rgb_samples, density_samples = self.scannerf[2*i + 1].forward_samples(opt, center, ray,
                                                                                            depth_samples,
                                                                                            mode=mode)
                    # Foreground objects: bundle adjusting pose and conditional nerf
                    else:
                        rgb_samples, density_samples = self.scannerf[2*i + 1].forward_samples(opt, center, ray,
                                                                                            depth_samples, latent, pose,
                                                                                            mode=mode)
                    composite_rgb_samples_fine += rgb_samples / (self.N_obj + 1)
                    composite_density_samples_fine += density_samples / (self.N_obj + 1)

        rgb, depth, opacity, prob = self.composite(opt, ray, composite_rgb_samples, composite_density_samples,
                                                   depth_samples)
        rgb_fine, depth_fine, opacity_fine, prob_fine = self.composite(opt, ray, composite_rgb_samples_fine,
                                                                       composite_density_samples_fine,
                                                                       depth_samples)

        return edict(rgb=rgb, depth=depth, opacity=opacity,
                     rgb_fine=rgb_fine, depth_fine=depth_fine, opacity_fine=opacity_fine)  # [B,HW,K]


    def render_by_slices(self, opt, pose, latent, intr=None, mode=None):
        ret_all = edict(rgb=[], depth=[], opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[], depth_fine=[], opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,opt. H*opt.W, opt.nerf.rand_rays):
            ray_idx = torch.arange(c, min(c+opt.nerf.rand_rays, opt.H*opt.W), device=opt.device)
            ret = self.render(opt, pose, latent, intr=intr, ray_idx=ray_idx, mode=mode) # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        return ret_all


    def sample_depth(self, opt, batch_size, num_rays=None):
        depth_min, depth_max = opt.nerf.depth.range
        num_rays = num_rays or opt.H * opt.W
        rand_samples = torch.rand(batch_size, num_rays, opt.nerf.sample_intvs, 1, device=opt.device) \
            if opt.nerf.sample_stratified else 0.5
        rand_samples += torch.arange(opt.nerf.sample_intvs, device=opt.device)[None, None, :,
                        None].float()  # [B,HW,N,1]
        depth_samples = rand_samples / opt.nerf.sample_intvs * (depth_max - depth_min) + depth_min  # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1 / (depth_samples + 1e-8),
        )[opt.nerf.depth.param]
        return depth_samples


    def sample_depth_from_pdf(self, opt, pdf):
        depth_min, depth_max = opt.nerf.depth.range
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1)  # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [B,HW,N+1]
        # take uniform samples
        grid = torch.linspace(0, 1, opt.nerf.sample_intvs_fine + 1, device=opt.device)  # [Nf+1]
        unif = 0.5 * (grid[:-1] + grid[1:]).repeat(*cdf.shape[:-1], 1)  # [B,HW,Nf]
        idx = torch.searchsorted(cdf, unif, right=True)  # [B,HW,Nf] \in {1...N}
        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min, depth_max, opt.nerf.sample_intvs + 1, device=opt.device)  # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1], 1)  # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2, index=(idx - 1).clamp(min=0))  # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2, index=idx.clamp(max=opt.nerf.sample_intvs))  # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2, index=(idx - 1).clamp(min=0))  # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2, index=idx.clamp(max=opt.nerf.sample_intvs))  # [B,HW,Nf]
        # linear interpolation
        t = (unif - cdf_low) / (cdf_high - cdf_low + 1e-8)  # [B,HW,Nf]
        depth_samples = depth_low + t * (depth_high - depth_low)  # [B,HW,Nf]
        return depth_samples[..., None]  # [B,HW,Nf,1]

    def get_pose(self, opt, var, mode=None):
        if mode in ["train", 'val']:
            # add learnable pose correction
            var.se3_refine = self.se3_refine.weight[var.idx]
            # 0th frames' object pose is initialized as fixed hyperparameter
            pose = camera.lie.se3_to_SE3(var.se3_refine)
            # pose[var.idx == 0] = self.obj_pose_init
        elif mode in ["eval", "test-optim"]:
            # align test pose to refined coordinate system (up to sim3)
            sim3 = self.sim3
            center = torch.zeros(1, 1, 3, device=opt.device)
            center = camera.cam2world(center, var.pose)[:, 0]  # [N,3]
            center_aligned = (center - sim3.t0) / sim3.s0 @ sim3.R * sim3.s1 + sim3.t1
            R_aligned = var.pose[..., :3] @ self.sim3.R
            t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
            pose = camera.pose(R=R_aligned, t=t_aligned)
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo:
                pose = camera.pose.compose([var.pose_refine_test, pose])
        else:
            pose = var.pose
        return pose


    def composite(self, opt, ray, rgb_samples, density_samples, depth_samples, prob_only=False):
        ray_length = ray.norm(dim=-1, keepdim=True)  # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0]  # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples, torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)],
                                       dim=2)  # [B,HW,N]
        dist_samples = depth_intv_samples * ray_length  # [B,HW,N]
        sigma_delta = density_samples * dist_samples  # [B,HW,N]
        alpha = 1 - (-sigma_delta).exp_()  # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=2).cumsum(
            dim=2)).exp_()  # [B,HW,N]
        prob = (T * alpha)[..., None]  # [B,HW,N,1]
        if prob_only:
            return prob
        # integrate RGB and depth weighted by probability
        depth = (depth_samples * prob).sum(dim=2)  # [B,HW,1]
        rgb = (rgb_samples * prob).sum(dim=2)  # [B,HW,3]
        opacity = prob.sum(dim=2)  # [B,HW,1]
        if opt.nerf.setbg_opaque:
            rgb = rgb + opt.data.bgcolor * (1 - opacity)
        return rgb, depth, opacity, prob  # [B,HW,K]


class CondNeRF(torch.nn.Module):
    """
    Latent code-conditional Neural Radiance Field
    """

    def __init__(self, opt):
        super().__init__()
        self.progress = torch.nn.Parameter(torch.tensor(0.))  # use Parameter so it could be checkpointed (BARF)

    def define_network(self, opt):
        input_3D_dim = 3 + opt.arch.dim_latent + 6 * opt.arch.posenc.L_3D if opt.arch.posenc \
            else 3 + opt.arch.dim_latent
        input_view_dim = 3 + 3 * opt.arch.posenc.L_pose + 6 * opt.arch.posenc.L_view

        # point-wise feature
        self.mlp_feat = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_feat)
        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            if li == len(L) - 1: k_out += 1
            linear = torch.nn.Linear(k_in, k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt, linear, out="first" if li == len(L) - 1 else None)
            self.mlp_feat.append(linear)
        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_rgb)
        feat_dim = opt.arch.layers_feat[-1]
        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = feat_dim + (input_view_dim if opt.nerf.view_dep else 0)
            linear = torch.nn.Linear(k_in, k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt, linear, out="all" if li == len(L) - 1 else None)
            self.mlp_rgb.append(linear)

    def tensorflow_init_weights(self, opt, linear, out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu")  # sqrt(2)
        if out == "all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out == "first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:], gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight, gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, opt, points_3D, latent, ray_unit=None, mode=None):
        points_enc = self.positional_encoding(opt, points_3D, L=opt.arch.posenc.L_3D)
        points_enc = torch.cat([points_3D, points_enc, latent], dim=-1)  # [B,...,3 + 6L_3D + dim_latent]
        feat = points_enc

        # extract coordinate-based, latent-conditioned features
        for li, layer in enumerate(self.mlp_feat):
            if li in opt.arch.skip: feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
            if li == len(self.mlp_feat) - 1:
                density = feat[..., 0]
                if opt.nerf.density_noise_reg and mode == "train":
                    density += torch.randn_like(density) * opt.nerf.density_noise_reg
                density_activ = getattr(torch_F, opt.arch.density_activ)  # relu_,abs_,sigmoid_,exp_....
                density = density_activ(density)
                feat = feat[..., 1:]
            feat = torch_F.relu(feat)

        # predict RGB values
        if opt.nerf.view_dep:
            assert (ray_unit is not None)
            if opt.arch.posenc:
                ray_enc = self.positional_encoding(opt, ray_unit, L=opt.arch.posenc.L_view)
                ray_enc = torch.cat([ray_unit, ray_enc], dim=-1)  # [B,...,6L+3]
            else:
                ray_enc = ray_unit
            feat = torch.cat([feat, ray_enc], dim=-1)
        for li, layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            if li != len(self.mlp_rgb) - 1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_()  # [B,...,3]
        return rgb, density

    def forward_samples(self, opt, center, ray, depth_samples, latent, pose, mode=None):
        """
        center, ray -> [B, n_rays, 3]
        depth_samples -> [B, n_rays, n_samples, 1]
        pose -> [B, 3, 4]
        latent => [dim_latent]
        """
        # 3D points in inhomogeneous coordinate
        points_3D_samples = camera.get_3D_points_from_depth(opt, center, ray, depth_samples,
                                                            multi_samples=True)  # [B, n_rays, n_samples, 3]
        # 3D points in homogeneous coordinate
        points_3D_samples = camera.to_hom(points_3D_samples)
        # [B, n_rays, n_samples, 4]
        # Rigid transformation of points with learned pose
        points_3D_samples = camera.world2cam(points_3D_samples, pose)

        latent = latent[None, None, None, :].expand_as(points_3D_samples)

        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,HW,3]
            ray_unit_samples = ray_unit[..., None, :].expand_as(points_3D_samples)  # [B, n_rays, n_samples, 3]
            # Rigid transformation of rays with learned pose
            ray_unit_samples = torch.matmul(pose[:, None, :, :-1], ray_unit_samples)
        else:
            ray_unit_samples = None

        rgb_samples, density_samples = self.forward(opt, points_3D_samples, latent, ray_unit=ray_unit_samples,
                                                    mode=mode)  # [B,HW,N],[B,HW,N,3]
        return rgb_samples, density_samples

    def positional_encoding(self, opt, input, L):  # [B,...,N] (BARF-style)
        input_enc = super().positional_encoding(opt, input, L=L)  # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start, end = opt.barf_c2f
            alpha = (self.progress.data - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=opt.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L) * weight).view(*shape)
        return input_enc


# class NeuralRenderer(nn.Module):
#     ''' Neural renderer class
#     Args:
#         n_feat (int): number of features
#         input_dim (int): input dimension; if not equal to n_feat,
#             it is projected to n_feat with a 1x1 convolution
#         out_dim (int): output dimension
#         final_actvn (bool): whether to apply a final activation (sigmoid)
#         min_feat (int): minimum features
#         img_size (int): output image size
#         use_rgb_skip (bool): whether to use RGB skip connections
#         upsample_feat (str): upsampling type for feature upsampling
#         upsample_rgb (str): upsampling type for rgb upsampling
#         use_norm (bool): whether to use normalization
#     '''
#
#     def __init__(
#             self, n_feat=128, input_dim=128, out_dim=3, final_actvn=True,
#             min_feat=32, img_size=64, use_rgb_skip=True,
#             upsample_feat="near_neigh", upsample_rgb="bilinear", use_norm=False,
#             **kwargs):
#         super().__init__()
#         self.final_actvn = final_actvn
#         self.input_dim = input_dim
#         self.use_rgb_skip = use_rgb_skip
#         self.use_norm = use_norm
#         n_blocks = int(log2(img_size) - 4)
#
#         assert (upsample_feat in ("near_neigh", "bilinear"))
#         if upsample_feat == "near_neigh":
#             self.upsample_2 = nn.Upsample(scale_factor=2.)
#         elif upsample_feat == "bilinear":
#             self.upsample_2 = nn.Sequential(nn.Upsample(
#                 scale_factor=2, mode='bilinear', align_corners=False), Blur())
#
#         assert (upsample_rgb in ("near_neigh", "bilinear"))
#         if upsample_rgb == "near_neigh":
#             self.upsample_rgb = nn.Upsample(scale_factor=2.)
#         elif upsample_rgb == "bilinear":
#             self.upsample_rgb = nn.Sequential(nn.Upsample(
#                 scale_factor=2, mode='bilinear', align_corners=False), Blur())
#
#         if n_feat == input_dim:
#             self.conv_in = lambda x: x
#         else:
#             self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)
#
#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] +
#             [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
#                        max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1)
#              for i in range(0, n_blocks - 1)]
#         )
#         if use_rgb_skip:
#             self.conv_rgb = nn.ModuleList(
#                 [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
#                 [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
#                            out_dim, 3, 1, 1) for i in range(0, n_blocks)]
#             )
#         else:
#             self.conv_rgb = nn.Conv2d(
#                 max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)
#
#         if use_norm:
#             self.norms = nn.ModuleList([
#                 nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
#                 for i in range(n_blocks)
#             ])
#         self.actvn = nn.LeakyReLU(0.2, inplace=True)
#
#     def forward(self, x):
#
#         net = self.conv_in(x)
#
#         if self.use_rgb_skip:
#             rgb = self.upsample_rgb(self.conv_rgb[0](x))
#
#         for idx, layer in enumerate(self.conv_layers):
#             hid = layer(self.upsample_2(net))
#             if self.use_norm:
#                 hid = self.norms[idx](hid)
#             net = self.actvn(hid)
#
#             if self.use_rgb_skip:
#                 rgb = rgb + self.conv_rgb[idx + 1](net)
#                 if idx < len(self.conv_layers) - 1:
#                     rgb = self.upsample_rgb(rgb)
#
#         if not self.use_rgb_skip:
#             rgb = self.conv_rgb(net)
#
#         if self.final_actvn:
#             rgb = torch.sigmoid(rgb)
#         return rgb
#
#
# class Blur(nn.Module):
#     def __init__(self):
#         super().__init__()
#         f = torch.Tensor([1, 2, 1])
#         self.register_buffer('f', f)
#
#     def forward(self, x):
#         f = self.f
#         f = f[None, None, :] * f[None, :, None]
#         return filter2D(x, f, normalized=True)
