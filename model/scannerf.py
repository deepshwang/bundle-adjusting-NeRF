import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import visdom
import matplotlib.pyplot as plt
from math import log2
#from kornia.filters import filter2D
import importlib
import wandb
import ipdb
import util, util_vis
from util import log, debug
from . import nerf
from . import base
import camera
import itertools


# ============================ main engine for training and evaluation ============================

class Model():

    def __init__(self, opt):
        self.ep = None
        self.timer = None
        self.sched_latent = None
        self.optim_latent = None
        self.sched_pose = None
        self.optim_pose = None
        self.sched = None
        self.optim = None
        self.it = None

    def initialize_wandb(self, opt):
        if opt.wandb:
            wandb.login(key="c757da017bc82753ff5e2fd3261549dfd5b3cc8c")
            wandb.init(project=opt.project)
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="train/step")


    def build_networks(self, opt):
        graph = importlib.import_module("model.{}".format(opt.model))
        log.info("building networks...")
        self.graph = graph.Graph(opt).to(opt.device)
        self.graph.se3_refine = torch.nn.Embedding(len(self.train_data), 6).to(opt.device)
        torch.nn.init.zeros_(self.graph.se3_refine.weight)

    def load_dataset(self, opt, eval_split="val"):
        data = importlib.import_module("data.{}".format(opt.data.dataset))
        log.info("loading training data...")
        self.train_data = data.Dataset(opt, split="train")
        self.train_loader = self.train_data.setup_loader(opt, shuffle=True)
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all,opt.device))
        log.info("loading test data...")
        self.test_data = data.Dataset(opt, split="val")
        self.test_loader = self.test_data.setup_loader(opt, shuffle=False)

    def restore_checkpoint(self, opt):
        epoch_start, iter_start = None, None
        if opt.resume:
            log.info("resuming from previous checkpoint...")
            iter_start = util.restore_checkpoint(opt, self, resume=opt.resume)
        elif opt.load is not None:
            log.info("loading weights from checkpoint {}...".format(opt.load))
            iter_start = util.restore_checkpoint(opt, self, load_name=opt.load)
        else:
            log.info("initializing weights from scratch...")
        self.iter_start = iter_start or 0

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
        self.optim = optimizer([dict(params=self.graph.scannerf.parameters(), lr=opt.optim.lr)])
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
            if opt.optim.lr_end:
                assert (opt.optim.sched.type == "ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end / opt.optim.lr) ** (1. / opt.max_iter)
            kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
            self.sched = scheduler(self.optim, **kwargs)

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
        # set up scheduler
        if opt.optim.sched_latent:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched_latent.type)
            if opt.optim.lr_latent_end:
                assert (opt.optim.sched_latent.type == "ExponentialLR")
                opt.optim.sched_latent.gamma = (opt.optim.lr_latent_end / opt.optim.lr_latent) ** (1. / opt.max_iter)
            kwargs = {k: v for k, v in opt.optim.sched_latent.items() if k != "type"}
            self.sched_latent = scheduler(self.optim_latent, **kwargs)

    def train(self, opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.graph.train()
        self.ep = 0  # dummy for timer
        # training
        loader = tqdm.trange(opt.max_iter, desc="training", leave=False)
        var = self.train_data.all
        for b in range(opt.scannerf.N_block):
            in_idx = (b+1) * len(self.train_data) // opt.scannerf.N_block
            var_in = edict()
            for k in var.keys(): var_in[k] = var[k][:in_idx]
            var_in.block = b
            for self.it in loader:
                if self.it < self.iter_start:
                    continue
                # set var to all available images (NOTE: For stricter control of annealing schedule...?)
                self.train_iteration(opt, var_in, loader)
                if self.it % opt.freq.ckpt == 0:
                    self.save_checkpoint(opt, it=self.it)
            log.title("TRAINING DONE for {}th block".format(b))
            if b == 0:
                return

    # WORKING
    def train_iteration(self, opt, var, loader):
        self.graph.train()
        # zero-grad optimizers
        self.optim.zero_grad()
        self.optim_pose.zero_grad()
        self.optim_latent.zero_grad()

        # Warm-ups if necessary
        if opt.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.optim_pose.param_groups[0]["lr_orig"] = self.optim_pose.param_groups[0][
                "lr"]  # cache the original learning rate
            self.optim_pose.param_groups[0]["lr"] *= min(1, self.it / opt.optim.warmup_pose)

        # Forward network & calculate loss g
        # before train iteration
        self.timer.it_start = time.time()
        var = self.graph.forward(opt, var, mode="train")
        loss = self.graph.compute_loss(opt, var, mode="train")
        loss = self.summarize_loss(opt, loss)
        loss.all.backward()

        # Proceed main network optimizer schedules
        self.optim.step()
        if opt.optim.warmup_pose:
            self.optim.param_groups[0]["lr"] = self.optim.param_groups[0]["lr_orig"]  # reset learning rate
        if opt.optim.sched:
            self.sched.step()

        # Proceed pose optimizer schedules
        self.optim_pose.step()
        if opt.optim.sched_pose:
            self.sched_pose.step()

        # Proceed latent optimizer schedules
        self.optim_latent.step()
        if opt.optim.sched_pose:
            self.sched_latent.step()

        # Track positional encoding annealing parameter (BARF)
        for i in range(self.graph.N_obj):
            self.graph.scannerf[2 * (i + 1)].progress.data.fill_(self.it / opt.max_iter)
            if opt.nerf.fine_sampling:
                self.graph.scannerf[2 * (i + 1) + 1].progress.data.fill_(self.it / opt.max_iter)

        # Record Scalars (loss / learning rate)
        if (self.it + 1) % opt.freq.scalar == 0 and opt.wandb:
            self.log_scalars(opt, loss, key_list=["render", "render_fine", "all"], step=self.it + 1, split="train")
            self.log_scalars(opt, var, key_list=["render", "render_fine", "all"], step=self.it + 1, split="train")

        # Visualize current performance
        if (self.it + 1) % opt.freq.vis == 0 and opt.wandb:
            self.visualize(opt, var, sample_image_idx=opt.viz.sample_image_idx, step=self.it + 1)
        loader.set_postfix(it=self.it, loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        util.update_timer(opt, self.timer, self.ep, len(loader))
        self.it += 1
        return loss

    def summarize_loss(self, opt, loss):
        loss_all = 0.
        assert ("all" not in loss)
        # weigh losses
        for key in loss:
            assert (key in opt.loss_weight)
            assert (loss[key].shape == ())
            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]), "loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]), "loss {} is NaN".format(key)
                loss_all += 10 ** float(opt.loss_weight[key]) * loss[key]
        loss.update(all=loss_all)
        return loss

    def save_checkpoint(self, opt, it=0, latest=False):
        util.save_checkpoint(opt, self, it=it, latest=latest)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, (iteration {2})".format(opt.group, opt.name, it))

    @torch.no_grad()
    def log_scalars(self, opt, val, key_list=[], metric=None, step=0, split="train"):
        log_dict={"{}/step".format(split): step}
        for key, value in val.items():
            if key in key_list:
                log_dict["{}/{}".format(split, key)] = value.detach().cpu().numpy()
        if metric is not None:
            for key, value in metric.items():
                log_dict["{}/{}".format(split, key)] = value

        if split == "train":
            # Log network learning rate
            lr = self.optim.param_groups[0]["lr"]
            log_dict["{}/{}".format(split, "lr")] = lr
            # log pose learning rate
            lr_pose = self.optim_pose.param_groups[0]["lr"]
            log_dict["{}/{}".format(split, "lr_pose")] = lr_pose
            # log latent learning rate
            lr_latent = self.optim_latent.param_groups[0]["lr"]
            log_dict["{}/{}".format(split, "lr_latent")] = lr_latent
        wandb.log(log_dict)


    @torch.no_grad()
    def visualize(self, opt, var, sample_image_idx=None, step=0, eps=1e-6):
        self.graph.eval()
        var = self.graph.forward(opt, var, sample_image_idx=sample_image_idx, mode="val") # Forward all images
        invdepth = 1/(var.depth_fine/var.opacity_fine+eps)
        rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
        # invdepth_map = invdepth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]

        ### Visualize rendered Image ###
        wandb.log({"[Step {} / {}] RGB Image".format(step, opt.max_iter): [wandb.Image(np.transpose(t.detach().cpu().numpy(), (1, 2, 0))) for t in rgb_map]})
        # wandb.log({"Depth Image": [t.detach().cpu().numpy()  for t in invdepth_map]})

        ### Visualize learned poses ###
        # Retrieve learned pose from graph
        poses = self.graph.se3_refine.weight.detach().cpu()
        poses = camera.lie.se3_to_SE3(poses)

        # Visualize pose
        fig = plt.figure(figsize=(20, 10))
        util_vis.plot_object_poses_as_coordinates(fig, poses, step)
        wandb.log({"Learned pose": wandb.Image(plt)})
        plt.clf()


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
            loss = self.graph.compute_loss(opt, var)
            loss = self.summarize_loss(opt, loss)
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

    def forward(self, opt, var, sample_image_idx=None, mode=None):
        # sample random rays for optimization
        num_rand_rays = opt.nerf.rand_rays if mode == "train" else opt.nerf.val_rand_rays
        var.ray_idx = torch.randperm(opt.H * opt.W, device=opt.device)[:num_rand_rays]

        pose = self.get_object_pose(opt, var, sample_image_idx=sample_image_idx, mode=mode)
        # render images
        if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
            ret = self.render(opt, pose, self.latent, intr=var.intr, ray_idx=var.ray_idx, mode=mode)  # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices for validations)
            ret = self.render_by_slices(opt, pose, self.latent, sample_image_idx=sample_image_idx, intr=var.intr, mode=mode)
        var.update(ret)
        return var

    def sample_planar_rays(self, opt):
        rand_h = torch.rand(opt.H)


    def render(self, opt, pose, latent, sample_image_idx=None, intr=None, ray_idx=None, mode=None):
        if sample_image_idx is None:
            batch_size = pose.shape[0]
        else:
            batch_size = len(sample_image_idx)
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
            if i == 0:
                rgb_samples, density_samples = self.scannerf[2 * i].forward_samples(opt, center, ray,
                                                                                    depth_samples,
                                                                                    mode=mode)


            # Foreground objects: bundle adjusting pose and conditional nerf
            else:
                latent = self.latent.weight[i - 1]
                rgb_samples, density_samples = self.scannerf[2 * i].forward_samples(opt, center, ray,
                                                                                    depth_samples, latent, pose,
                                                                                    mode=mode)

            composite_rgb_samples += rgb_samples / (self.N_obj + 1)
            composite_density_samples += density_samples / (self.N_obj + 1)
            prob = self.composite(opt, ray, rgb_samples, density_samples, depth_samples, prob_only=True)

            if opt.nerf.fine_sampling:
                with torch.no_grad():
                    # resample depth acoording to coarse empirical distribution
                    depth_samples_fine = self.sample_depth_from_pdf(opt, batch_size, pdf=prob[..., 0])  # [B,HW,Nf,1]
                    depth_samples_fine = torch.cat([depth_samples.expand([depth_samples_fine.shape[0], -1, -1, -1]),
                                                    depth_samples_fine], dim=2)  # [B,HW,N+Nf,1]
                    depth_samples_fine = depth_samples_fine.sort(dim=2).values
                    if i == 0:
                        rgb_samples, density_samples = self.scannerf[2 * i + 1].forward_samples(opt, center, ray,
                                                                                                depth_samples_fine,
                                                                                                mode=mode)
                    # Foreground objects: bundle adjusting pose and conditional nerf
                    else:
                        rgb_samples, density_samples = self.scannerf[2 * i + 1].forward_samples(opt, center, ray,
                                                                                                depth_samples_fine,
                                                                                                latent,
                                                                                                pose,
                                                                                                mode=mode)
                    composite_rgb_samples_fine += rgb_samples / (self.N_obj + 1)
                    composite_density_samples_fine += density_samples / (self.N_obj + 1)

        rgb, depth, opacity, prob = self.composite(opt, ray, composite_rgb_samples, composite_density_samples,
                                                   depth_samples)
        rgb_fine, depth_fine, opacity_fine, prob_fine = self.composite(opt, ray, composite_rgb_samples_fine,
                                                                       composite_density_samples_fine,
                                                                       depth_samples_fine)

        return edict(rgb=rgb, depth=depth, opacity=opacity,
                     rgb_fine=rgb_fine, depth_fine=depth_fine, opacity_fine=opacity_fine)  # [B,HW,K]

    def render_by_slices(self, opt, pose, latent, sample_image_idx=None, intr=None, mode=None):
        if sample_image_idx is not None:
            intr = intr[sample_image_idx]
        ret_all = edict(rgb=[], depth=[], opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[], depth_fine=[], opacity_fine=[])
        # render the image by slices for memory considerations
        for c in tqdm.tqdm(range(0, opt.H * opt.W, opt.nerf.val_rand_rays)):
            ray_idx = torch.arange(c, min(c + opt.nerf.val_rand_rays, opt.H * opt.W), device=opt.device)
            ret = self.render(opt, pose, latent, sample_image_idx=sample_image_idx, intr=intr, ray_idx=ray_idx, mode=mode)  # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k], dim=1)
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

    def sample_depth_from_pdf(self, opt, batch_size, pdf):
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

    def get_object_pose(self, opt, var, sample_image_idx=None, mode=None):
        if mode in ["train", "val"]:
            # add learnable pose correction
            if sample_image_idx is None:
                var.se3_refine = self.se3_refine.weight[var.idx]
            else:
                var.se3_refine = self.se3_refine.weight[sample_image_idx]
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

    def compute_loss(self, opt, var, mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size, 3, opt.H * opt.W).permute(0, 2, 1)
        if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
            image = image[:, var.ray_idx]
        # compute image losses
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb, image)
        if opt.loss_weight.render_fine is not None:
            assert opt.nerf.fine_sampling
            loss.render_fine = self.MSE_loss(var.rgb_fine, image)
        return loss


class CondNeRF(torch.nn.Module):
    """
    Latent code-conditional Neural Radiance Field
    """

    def __init__(self, opt):
        super().__init__()
        self.define_network(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.))  # use Parameter so it could be checkpointed (BARF)

    def define_network(self, opt):
        input_3D_dim = 3 + opt.arch.dim_latent + 6 * opt.arch.posenc.L_3D if opt.arch.posenc \
            else 3 + opt.arch.dim_latent
        input_view_dim = 3 + 6 * opt.arch.posenc.L_view

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

    def forward_samples(self, opt, center, ray, depth_samples, latent, pose ,mode=None):
        """
        center, ray -> [B, n_rays, 3]
        depth_samples -> [B, n_rays, n_samples, 1]
        pose -> [B, 3, 4]
        latent => [dim_latent]
        """
        # 3D points in inhomogeneous coordinate
        points_3D_samples = camera.get_3D_points_from_depth(opt, center, ray, depth_samples,
                                                            multi_samples=True)  # [B, n_rays, n_samples, 3]
        # [B, n_rays, n_samples, 4]
        # Rigid transformation of points with learned pose

        points_3D_samples = camera.world2cam(points_3D_samples, pose[:, None, ...])
        latent = latent[None, None, None, :].expand(points_3D_samples.shape[0], points_3D_samples.shape[1],
                                                    points_3D_samples.shape[2], -1)

        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,HW,3]

            # Rigid so(2)-transformation of rays with learned pose
            ray_unit_samples = camera.world2cam_rays(ray_unit, pose)[..., None, :].expand_as(points_3D_samples)

        else:
            ray_unit_samples = None

        rgb_samples, density_samples = self.forward(opt, points_3D_samples, latent, ray_unit=ray_unit_samples,
                                                    mode=mode)  # [B,HW,N],[B,HW,N,3]
        return rgb_samples, density_samples

    def positional_encoding(self, opt, input, L):  # [B,...,N] (BARF-style)
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=opt.device) * np.pi  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]

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
