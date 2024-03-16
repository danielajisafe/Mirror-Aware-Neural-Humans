import os
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import lpips
from numpy.linalg import norm
from torch import linalg as LA

from .utils.skeleton_utils import axisang_to_rot6d, get_geodesic_dists
from torch.profiler import profile, record_function, ProfilerActivity



# TODO: loss fns, put them in a different file later
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def img2mse(x, y, reduction="mean", to_yuv=False, scale_yuv=[0.1, 1.0, 1.0]):

    if to_yuv:
        x = rgb_to_yuv(x)
        y = rgb_to_yuv(y)
        scale_yuv = torch.tensor(scale_yuv).float().view(1, 3)
        sqr_diff = (x - y)**2 * scale_yuv
    else:
        sqr_diff = (x - y)**2

    if reduction == "mean":
        return torch.mean(sqr_diff)
    elif reduction == "sum":
        return torch.sum(sqr_diff)
    else:
        return sqr_diff

def img2l1(x, y, reduction="mean", to_yuv=False, scale_yuv=[0.1, 1.0, 1.0]):

    if to_yuv:
        x = rgb_to_yuv(x)
        y = rgb_to_yuv(y)
        scale_yuv = torch.tensor(scale_yuv).float().view(1, 3)
        diff = (x - y).abs() * scale_yuv
    else:
        diff = (x - y).abs()

    if reduction == "mean":
        return torch.mean(diff)
    elif reduction == "sum":
        return torch.sum(diff)
    else:
        return diff

def acc2bce(x, y, reduction="mean", eps=1e-8):
    bce_loss = -(y * torch.log(x + eps) + (1. - y) * torch.log(1 - x + eps))
    if reduction == "mean":
        return torch.mean(bce_loss)
    elif reduction == "sum":
        return torch.sum(bce_loss)
    elif reduction == "off":
        bce_loss = bce_loss[y < 1.0]
        return torch.mean(bce_loss)
    else:
        return bce_loss


def img2huber(x, y, reduction="mean", beta=0.1):
    return F.smooth_l1_loss(x, y, reduction=reduction, beta=beta)

def img2psnr(img, target):
    return mse2psnr(img2mse(img, target))

def cosine_dist(a, b, **kwargs):
    a = F.normalize(a, dim=-1, p=2)
    b = F.normalize(b, dim=-1, p=2)

    return 1 - (a * b).sum(-1)

def batchify_rays(rays_flat, rays_flat_v=None, chunk=1024*64, ray_caster=None, use_mirr=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        batch_kwargs = {k: kwargs[k][i:i+chunk] if torch.is_tensor(kwargs[k]) else kwargs[k]
                        for k in kwargs}
        # ret = ray_caster(rays_flat[i:i+chunk], **batch_kwargs)
        if use_mirr:
            #print("batch_kwargs", batch_kwargs.keys())
            ret = ray_caster(rays_flat[i:i+chunk], ray_batch_v=rays_flat_v[i:i+chunk], use_mirr=use_mirr, **batch_kwargs)
        else:
            #print("batch_kwargs", batch_kwargs.keys())
            ret = ray_caster(rays_flat[i:i+chunk], use_mirr=use_mirr, **batch_kwargs)
        
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render_pts(ray_caster, **kwargs):
    # assume no batchify needed
    return ray_caster(fwd_type='density', **kwargs)

def render(H, W, focal, chunk=1024*64, rays=None, c2w=None,
           near=0., far=1., near_v=0., far_v=1., center=None,
           use_viewdirs=False, c2w_staticcam=None, use_mirr=False,
           index=None,
            **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if (c2w is not None) and (rays is None):
        # special case to render full image
        center = center.ravel() if center is not None else None
        # import pdb; pdb.set_trace()

        rays_o, rays_d, _ = get_rays(H, W, focal, c2w, center=center)
    else:
        # use provided ray batch

        # rendering time
        if len(rays) == 2:
            '''whether real and/or virt, rays are indexed to have remain single (rays_o, rays_d) before getting here'''
            rays_o, rays_d  = rays
        
        # train time (then 4)
        elif not use_mirr:
            rays_o, rays_d, _, _ = rays
            #import ipdb; ipdb.set_trace()
        else:   
            rays_o, rays_d, rays_o_v, rays_d_v  = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d

        try:
            if use_mirr: viewdirs_v = rays_d_v
        except:
            import ipdb; ipdb.set_trace()

        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            #rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
            import ipdb; ipdb.set_trace()
            # if not use_mirr:
            #     rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
            # else:
            rays_o, rays_d, rays_o_v, rays_d_v = get_rays(H, W, focal, c2w_staticcam)
        
        # normalize ray length
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        if use_mirr:
            # normalize ray length (virt) to ensure unit norm
            viewdirs_v = viewdirs_v / torch.norm(viewdirs_v, dim=-1, keepdim=True)
            viewdirs_v = torch.reshape(viewdirs_v, [-1,3]).float()
    # Create ray batch
    sh = rays_d.shape # [..., 3]
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    #rays = torch.cat([rays_o, rays_d, near, far], -1)
    rays = torch.cat([rays_o.to(near.device), rays_d.to(near.device), near, far], -1)


    if use_mirr:
        rays_o_v = torch.reshape(rays_o_v, [-1,3]).float()
        rays_d_v = torch.reshape(rays_d_v, [-1,3]).float()

        near_v, far_v = near_v * torch.ones_like(rays_d_v[...,:1]), far_v * torch.ones_like(rays_d_v[...,:1])
        rays_v = torch.cat([rays_o_v.to(near_v.device), rays_d_v.to(near_v.device), near_v, far_v], -1)
    

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
        if use_mirr: rays_v = torch.cat([rays_v, viewdirs_v], -1)


    # Render and reshape
    #all_ret = batchify_rays(rays, chunk, **kwargs)
    if not use_mirr:
        #import ipdb; ipdb.set_trace()
        #use_mirr=False
        all_ret = batchify_rays(rays_flat=rays, chunk=chunk, use_mirr=use_mirr, **kwargs)
    else:
        #import ipdb; ipdb.set_trace()
        #use_mirr=True 
        all_ret = batchify_rays(rays_flat=rays, rays_flat_v=rays_v, chunk=chunk, use_mirr=use_mirr, **kwargs)

    for k in all_ret:
        if all_ret[k].dim() >= 4:
            continue
        if k in ['graph_feat', 'bone_logit']:
            continue
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret

def get_loss_fn(loss_name, beta):
    if loss_name == "MSE":
        loss_fn = img2mse
    elif loss_name == "L1":
        loss_fn = img2l1
    elif loss_name == "Huber":
        loss_fn = lambda x, y, reduction='mean', beta=beta: img2huber(x, y, reduction, beta)
    else:
        raise NotImplementedError

    return loss_fn

def get_reg_fn(reg_name):
    if reg_name is None:
        return None
    if reg_name == "L1":
        reg_fn = img2l1
    elif reg_name == "MSE":
        reg_fn = img2mse
    elif reg_name == "BCE":
        reg_fn = acc2bce
    else:
        raise NotImplementedError
    return reg_fn

# helper funcs
def decay_optimizer_lrate(lrate, lrate_decay, decay_rate, optimizer,
                          global_step=None, decay_unit=1000):

    #decay_steps = lrate_decay * decay_unit
    decay_steps = lrate_decay
    optim_step = optimizer.state[optimizer.param_groups[0]['params'][0]]['step'] // decay_unit

    #new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
    new_lrate = lrate * (decay_rate ** (optim_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    return new_lrate, None

def dict_to_device(d, device):
    return {k: d[k].to(device) if d[k] is not None else None for k in d}

def set_tag_name(name, coarse):
    return name if not coarse else name + '0'

@torch.no_grad()
def get_gradnorm(module):
    total_norm  = 0.0
    cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        cnt += 1
    avg_norm = (total_norm / cnt) ** 0.5
    total_norm = total_norm ** 0.5
    return total_norm, avg_norm

class Trainer:

    def __init__(self, args, data_attrs, optimizer, pose_optimizer,
                 render_kwargs_train, render_kwargs_test, popt_kwargs=None,
                 device=None):
        '''
        args: args for training
        data_attrs: data attributes needed for training
        optimizer: optimizer for NeRF network
        pose_optimizer: optimizer for poses
        render_kwargs_train: kwargs for rendering outputs on training batch
        render_kwargs_test: kwargs for rendering testing data
        popt_kwargs: pose optimization layer and relevant keyword arguments
        '''
        self.args = args
        self.optimizer = optimizer
        self.pose_optimizer = pose_optimizer
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.popt_kwargs = popt_kwargs
        self.device = device

        self.hwf = data_attrs['hwf']
        self.data_attrs = data_attrs

        if self.args.use_lpips_loss:
            assert False, 'Not supported now!'
            lpips_model = lpips.LPIPS(net='vgg', lpips=False)
            print('use perceptual')
            if not args.debug:
                self.lpips = nn.DataParallel(lpips_model)
            else:
                self.lpips = lpips_model


    def train_batch(self, batch, i=0, global_step=0):
        '''
        i: current iteration
        global_step: global step (TODO: does this matter?)
        '''
        args = self.args
        H, W, focal = self.hwf
        batch = dict_to_device(batch, self.device)
        ray_caster = self.render_kwargs_train['ray_caster']

        # step 1a: get keypoint/human poses data
        popt_detach = not (args.opt_pose_stop is None or i < args.opt_pose_stop)
        kp_args, extra_args = self.get_kp_args(batch, detach=popt_detach)

        # step 1b: get other args for forwarding
        fwd_args = self.get_fwd_args(batch)

        # step 2: ray caster forward
        preds = render(H, W, focal, chunk=args.chunk, verbose=i < 10,
                       retraw=False, index=i, N_uniques=args.N_sample_images,
                       **kp_args, **fwd_args, **self.render_kwargs_train,
                       use_mirr=args.use_mirr)

        # step 3: loss computation and updates
        loss_dict, stats = self.compute_loss(batch, preds, kp_opts={**kp_args, **extra_args},
                                             popt_detach=popt_detach or not args.opt_pose,
                                             global_step=global_step, use_mirr=args.use_mirr)

        # import ipdb; ipdb.set_trace()
        optim_stats = self.optimize(loss_dict['total_loss'], i, popt_detach or not args.opt_pose)

        # step 4: rate decay
        new_lrate, _ = decay_optimizer_lrate(args.lrate, args.lrate_decay,
                                             decay_rate=args.lrate_decay_rate,
                                             optimizer=self.optimizer,
                                             global_step=global_step,
                                             decay_unit=args.decay_unit)
        import pdb
        # pdb.set_trace()
        if not args.finetune:
            ray_caster.module.update_embed_fns(global_step, args)

        # step 5: logging
        stats = {'lrate': new_lrate,
                 'alpha': preds['acc_map'].mean().item(),
                 'cutoff': ray_caster.module.network.pe_fn.get_tau(),
                 **stats, **optim_stats}

        return loss_dict, stats

    def get_fwd_args(self, batch):
        '''
        get basic data required for ray casting from batch
        '''
        return {
            'rays': batch['rays'],
            'cams': batch['cam_idxs'] if self.args.opt_framecode else None,
            'subject_idxs': batch.get('subject_idxs', None),
        }

    def get_kp_args(self, batch, detach=False):
        '''
        get human pose data from either batch or popt layer
        '''

        extras = {}
        if self.popt_kwargs is None or self.popt_kwargs['popt_layer'] is None:
            # is this for non-pose_opt model? i think yes, DANBO assumes optimized pose (so no internal pose-opt)
            #import ipdb; ipdb.set_trace()
            return self._create_kp_args_dict(kps=batch['kp3d'], skts=batch['skts'],
                                             bones=batch['bones'], cyls=batch['cyls'],
                                             #use virt pose + real pose
                                            #kps_v=batch['kp3d_v'], cyls_v=batch['cyls_v'],
                                            A_dash=batch['A_dash'], m_normal=batch['m_normal'],
                                            avg_D=batch['avg_D'], idx_repeat=batch['idx_repeat'], 
                                            pixel_loc_repeat=batch['pixel_loc_repeat'], 
                                            pixel_loc_v_repeat=batch['pixel_loc_v_repeat']
                                            ), extras

        kp_idx = batch['kp_idx']
        if torch.is_tensor(kp_idx):
            kp_idx = kp_idx.cpu().numpy()

        # forward to get optimized pose data
        popt_layer = self.popt_kwargs['popt_layer']
        kps, bones, skts, _, rots = popt_layer(kp_idx)
        kp_args = self._create_kp_args_dict(kps=kps, skts=skts,
                                            bones=bones, cyls=batch['cyls'], 
                                            #use virt pose + optimized real pose
                                            #kps_v=batch['kp3d_v'], cyls_v=batch['cyls_v'],
                                            A_dash=batch['A_dash'], m_normal=batch['m_normal'],
                                            avg_D=batch['avg_D'], idx_repeat=batch['idx_repeat'], 
                                            pixel_loc_repeat=batch['pixel_loc_repeat'], 
                                            pixel_loc_v_repeat=batch['pixel_loc_v_repeat']
                                            )

        # not for ray casting, for loss computation only.
        extras['rots'] = rots

        if detach:
            kp_args = {k: kp_args[k].detach() for k in kp_args if kp_args[k] is not None}
            extras = {k: extras[k].detach() for k in extras if extras[k] is not None}

        return kp_args, extras


    def _create_kp_args_dict(self, kps, skts=None, kps_v=None, cyls_v=None, A_dash=None,
                            m_normal=None, avg_D=None, bones=None, cyls=None, rots=None,
                            idx_repeat=None, pixel_loc_repeat=None, pixel_loc_v_repeat=None):
        # TODO: unified the variable names (across all functions)?
        return {'kp_batch': kps, 'skts': skts, 'bones': bones, 'cyls': cyls,
                'kp_batch_v': kps_v, 'cyls_v': cyls_v,
                'A_dash': A_dash, 'm_normal': m_normal, 'avg_D': avg_D, 
                'idx_repeat':idx_repeat, 'pixel_loc_repeat':pixel_loc_repeat,
                'pixel_loc_v_repeat':pixel_loc_v_repeat}

    def brightness(self, img):
        # ref: https://stackoverflow.com/a/62780132/12761745
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return torch.mean(LA.norm(img, dim=2)) / torch.sqrt(torch.tensor(3))
        else:
            # Grayscale
            return torch.mean(img)

    def compute_loss(self, batch, preds,
                     kp_opts=None, popt_detach=False, global_step=0, use_mirr=False):
        '''
        popt_detach: False if we need kp_loss

        Assume returns from _compute_loss functions are tuples of (losses, stats)
        '''
        args = self.args

        total_loss = 0.
        results = []

        #import ipdb; ipdb.set_trace()
        if use_mirr and 'rgb_map_ref' not in preds:
            print("there is an issue. where are mirrored/reflected rays?")
            import ipdb; ipdb.set_trace()

        rgb_pred_ref, acc_pred_ref, v_bright_loss = None, None, None
        if use_mirr:
            rgb_pred_ref=preds['rgb_map_ref']
            acc_pred_ref=preds['acc_map_ref']

        # update with color factor
        eps = 1e-36
        """
        if args.opt_r_color and use_mirr:
            r_factor = self.render_kwargs_train['r_color_factor']
            # r_factor = self.render_kwargs_train['ray_caster'].module.network.r_color_factor
            # positive constraint
            r_factor = torch.sqrt(r_factor**2 + eps)
            preds['rgb_map'] = preds['rgb_map'] * r_factor
            # print(f"r_factor {r_factor}")
        """
        if args.opt_v_color and use_mirr:
            # add contraints for virt brightness to be close to real 
            if args.v_bright_reg:
                # import ipdb; ipdb.set_trace()
                # get total brightness in R and V
                Br = self.brightness(preds['rgb_map']) #/255.
                Bv = self.brightness(rgb_pred_ref)
                # add contraint to become equal
                v_bright_loss = torch.sum((Br - Bv)**2)

            # scale intended "bright" virt to dark, to fit GT dark
            v_factor = self.render_kwargs_train['v_color_factor']
            # v_factor = self.render_kwargs_train['ray_caster'].module.network.v_color_factor

            # make color factor positive
            v_factor = torch.sqrt(v_factor**2 + eps)
            rgb_pred_ref = rgb_pred_ref * v_factor
            # print(f"v_factor {v_factor}")

            

        # rgb loss of nerf
        results.append(self._compute_nerf_loss(batch, rgb_pred=preds['rgb_map'], acc_pred=preds['acc_map'],
                                                rgb_pred_ref=rgb_pred_ref, acc_pred_ref=acc_pred_ref,
                                                use_mirr=use_mirr, v_bright_loss=v_bright_loss))
        if 'rgb0' in preds:

            rgb0_ref, acc0_ref, v_bright_loss0 = None, None, None
            if use_mirr:
                rgb0_ref=preds['rgb0_ref']
                acc0_ref=preds['acc0_ref']

                if args.opt_v_color:
                    # add contraints for virt brightness to be close to real 
                    if args.v_bright_reg:
                        # import ipdb; ipdb.set_trace()
                        # get total brightness in R and V
                        Br = self.brightness(preds['rgb0']) #/255.
                        Bv = self.brightness(rgb0_ref)
                        # add contraint to become equal
                        v_bright_loss0 = torch.sum((Br - Bv)**2)
  

            results.append(self._compute_nerf_loss(batch, rgb_pred=preds['rgb0'], acc_pred=preds['acc0'],
                                                    rgb_pred_ref=rgb0_ref, acc_pred_ref=acc0_ref,
                                                    coarse=True, loss_weight=args.coarse_weight, use_mirr=use_mirr,
                                                    v_bright_loss=v_bright_loss0))
        if args.use_lpips_loss:
            results.append(self._compute_lpips_loss(batch, rgb_pred=preds['rgb_map'], acc_pred=preds['acc_map'],
                                                    rgb_pred_ref=rgb_pred_ref, acc_pred_ref=acc_pred_ref,
                                                    use_mirr=use_mirr))
            #if 'rgb0' in preds:
            #    results.append(self._compute_lpips_loss(batch, preds['rgb0'], preds['acc0'],
            #                                            coarse=True, loss_weight=args.coarse_weight))

        # assignment-related loss
        if 'confd' in preds and args.agg_type == 'sigmoid':
            results.append(self._compute_soft_softmax_loss(preds, global_step, use_mirr=use_mirr))
            
        # volume-related
        if args.opt_vol_scale:
            results.append(self._compute_volume_scale_loss())

        # need to update human poses, computes it.
        if not popt_detach:
            results.append(self._compute_kp_loss(batch, kp_opts))

        # collect losses and stats
        loss_dict, stats = {}, {}
        for i, (loss, stat) in enumerate(results):
            for k in loss:
                if loss[k] is not None:
                    total_loss = total_loss + loss[k]
                # else:
                #     import ipdb; ipdb.set_trace()
                    
                loss_dict[k] = loss[k]
            for k in stat:
                stats[k] = stat[k].item()

        loss_dict['total_loss'] = total_loss

        return loss_dict, stats

    def _compute_nerf_loss(self, batch, rgb_pred, acc_pred, rgb_pred_ref=None, acc_pred_ref=None, base_bg=1.0,
                           coarse=False, loss_weight=1.0, tag_prefix='', use_mirr=False, v_bright_loss=None):
        '''compute loss
        TODO: base_bg from argparse?
        '''
        args = self.args
        loss_fn = get_loss_fn(args.loss_fn, args.loss_beta)
        reg_fn = get_reg_fn(args.reg_fn)

        bgs = batch['bgs'] if 'bgs' in batch else base_bg
        bgs_v = batch['bgs_v'] if 'bgs_v' in batch else base_bg

        if args.use_background:
            
            if use_mirr:
                # the real occludes the mirrored person and its mask is affected/cut-out by the occlusion errors 

                # L^ = L^ + (1-&^) Ib)
                # initial L^ already weighted with &^, see rgb_map in networks/nerf.py
                rgb_pred_ref = rgb_pred_ref + (1. - acc_pred_ref)[..., None] * bgs_v

                n_rays = batch['n_rays_per_img_dups'][0].item()
                n_overlap_pixels = batch['n_overlap_pixels_dups'][0].item()

                # use layered background
                if args.overlap_rays and n_rays != -1 and args.layered_bkgd: 
                    # use mask for differentiability 
                    zero_mask1 = torch.ones_like(bgs.reshape(-1,n_rays, 3))
                    zero_mask2 = torch.ones_like(rgb_pred_ref.reshape(-1,n_rays, 3))
                
                    zero_mask1[:,:-n_overlap_pixels,:] = 0; 
                    zero_mask2[:,-n_overlap_pixels:,:] = 0; 

                    bgs_masked = bgs.reshape(-1,n_rays, 3) * zero_mask2
                    rgb_pred_ref_masked = rgb_pred_ref.reshape(-1,n_rays, 3) * zero_mask1
                    
                    # first n is background, n:end is overlap_rays pixels
                    bgs = bgs_masked + rgb_pred_ref_masked
                    bgs = bgs.reshape(-1, 3)

            # L = L + (1-&)L^
            # initial L already weighted with &, see rgb_map in networks/nerf.py
            rgb_pred = rgb_pred + (1. - acc_pred)[..., None] * bgs

        rgb_loss = loss_fn(rgb_pred, batch['target_s'], reduction='mean')

        if use_mirr:
            # the mirrored person's mask is less affected, and farthest from the camera
            rgb_loss_v = loss_fn(rgb_pred_ref, batch['target_s_v'], reduction='mean')
            rgb_loss = rgb_loss*args.r_weight + rgb_loss_v*args.v_weight

            if v_bright_loss is not None:
                rgb_loss = rgb_loss + v_bright_loss
            
        rgb_loss = rgb_loss * loss_weight * args.rgb_loss_coef
        
        psnr = img2psnr(rgb_pred.detach(), batch['target_s'])
        if use_mirr:
            psnr_v = img2psnr(rgb_pred_ref.detach(), batch['target_s_v'])
        
        loss_dict = {}
        stat_dict = {}

        loss_tag = set_tag_name(f'{tag_prefix}rgb_loss', coarse)
        bright_tag = set_tag_name(f'{tag_prefix}bright_loss', coarse)
        stat_tag = set_tag_name(f'{tag_prefix}psnr', coarse)

        stat_dict[stat_tag] = psnr
        if use_mirr:
            stat_tag_v = set_tag_name(f'{tag_prefix}psnr_v', coarse)
            stat_dict[stat_tag_v] = psnr_v

        loss_dict[loss_tag] = rgb_loss
        loss_dict[bright_tag] = v_bright_loss

        if reg_fn is not None:
            reg_loss = reg_fn(acc_pred, batch['fgs'][..., 0], reduction='off') # * args.reg_coef

            if use_mirr:
                reg_loss_v = reg_fn(acc_pred_ref , batch['fgs_v'][..., 0], reduction='off') # * args.reg_coef
                reg_loss = reg_loss*args.r_weight + reg_loss_v*args.v_weight

            reg_loss = reg_loss * args.reg_coef

            reg_tag = set_tag_name(f'{tag_prefix}reg_loss', coarse)
            loss_dict[reg_tag] = reg_loss

        return loss_dict, stat_dict


    def _compute_lpips_loss(self, batch, rgb_pred, acc_pred, rgb_pred_ref, acc_pred_ref, base_bg=1.0,
                            coarse=False, loss_weight=1.0, use_mirr=False):
        args = self.args # Nice!
        p = args.patch_size

        bgs = batch['bgs'] if 'bgs' in batch else base_bg
        if use_mirr:
            bgs_v = batch['bgs_v'] if 'bgs_v' in batch else bgs

        if args.use_background:
            rgb_pred = rgb_pred + (1. - acc_pred)[..., None] * bgs
            if use_mirr:
                rgb_pred_ref = rgb_pred_ref + (1. - acc_pred_ref)[..., None] * bgs_v

        # normalize to [-1, 1]
        rgb_pred = rgb_pred.reshape(-1, p, p, 3).permute(0, 3, 1, 2) * 2. - 1.
        rgb_gt = batch['target_s'].reshape(-1, p, p, 3).permute(0, 3, 1, 2) * 2. - 1.

        if use_mirr:
            rgb_pred_ref = rgb_pred_ref.reshape(-1, p, p, 3).permute(0, 3, 1, 2) * 2. - 1.
            rgb_gt_v = batch['target_s_v'].reshape(-1, p, p, 3).permute(0, 3, 1, 2) * 2. - 1.
        
        loss_dict = {}

        _, per_layer_loss = self.lpips(rgb_gt, rgb_pred, retPerLayer=True)
        if use_mirr:
            _, per_layer_loss_v = self.lpips(rgb_gt_v, rgb_pred_ref, retPerLayer=True)

        lpips_loss = torch.stack(per_layer_loss)[:3].sum(0).mean() # * args.lpips_coef
        if use_mirr:
            lpips_loss_v = torch.stack(per_layer_loss_v)[:3].sum(0).mean() # * args.lpips_coef

        loss_tag = set_tag_name('lpips_loss', coarse)
        loss_dict[loss_tag] = lpips_loss
        
        if use_mirr:
            loss_tag_v = set_tag_name('lpips_loss_v', coarse)
            loss_dict[loss_tag_v] = lpips_loss_v

        return loss_dict, {}

    def _compute_kp_loss(self, batch, kp_opts):

        args = self.args
        anchors = self.popt_kwargs['popt_anchors']
        kp_idx = batch['kp_idx']

        # shapes (N_rays, N_joints, *)
        if args.opt_rot6d:
            reg_bones = anchors['rots'][kp_idx, ..., :3, :2].flatten(start_dim=-2)
            bones = kp_opts['rots'][..., :3, :2].flatten(start_dim=-2)
        else:
            reg_bones = anchors['bones'][kp_idx]
            bones = kp_opts['bones']
        assert len(reg_bones) == len(bones)

        # loss = 0 if bone_loss < tol
        # loss = bone_loss - tol otherwise
        tol = args.opt_pose_tol
        kp_loss = (reg_bones - bones).pow(2.)[:, 1:] # exclude root (idx = 0)
        loss_mask = (kp_loss > tol).float()
        kp_loss = torch.lerp(torch.zeros_like(kp_loss),kp_loss-tol, loss_mask).sum(-1)
        kp_loss = kp_loss.mean() * args.opt_pose_coef

        loss_dict = {'kp_loss': kp_loss}

        # TODO: add temporal loss
        if args.use_temp_loss:
            popt_layer = self.popt_kwargs['popt_layer']
            # obtain indices of previous/next pose
            kp_idx_prev = batch['kp_idx'] - 1
            kp_idx_next = (batch['kp_idx'] + 1) % len(popt_layer.bones)
            if torch.is_tensor(kp_idx_prev):
                kp_idx_prev = kp_idx_prev.cpu().numpy()
                kp_idx_next = kp_idx_next.cpu().numpy()

            # obtain optimized prev/next poses
            prev_kps, prev_bones, _, _, prev_rots = popt_layer(kp_idx_prev)
            next_kps, next_bones, _, _, next_rots = popt_layer(kp_idx_next)

            if args.opt_rot6d:
                prev_bones = prev_rots[..., :3, :2].flatten(start_dim=-2)
                next_bones = next_rots[..., :3, :2].flatten(start_dim=-2)

            # detach the reguralizer
            prev_kps, prev_bones = prev_kps.detach(), prev_bones.detach()
            next_kps, next_bones = next_kps.detach(), next_bones.detach()
            temp_valid = batch['temp_val']
            kps = kp_opts['kp_batch']

            ang_vel = ((bones - prev_bones) - (next_bones - bones)).pow(2.).sum(-1)
            joint_vel = ((kps - prev_kps) - (next_kps - kps)).pow(2.).sum(-1)
            temp_loss = (ang_vel + joint_vel) * temp_valid[..., None]
            temp_loss = temp_loss.mean() * args.temp_coef

            loss_dict['temp_loss'] = temp_loss

        # mean per-joint change
        pjpc = (anchors['kps'][kp_idx] - kp_opts['kp_batch'].detach()).pow(2.).sum(-1).pow(0.5)
        mpjpc = pjpc.mean() / args.ext_scale

        return loss_dict, {'MPJPC': mpjpc}

    def _compute_soft_softmax_loss(self, preds, global_step, use_mirr=False):
        args = self.args
        ray_caster = self.render_kwargs_train['ray_caster'].module
        network = ray_caster.network
        assert args.attenuate_invalid == False, 'attenuate_invalid needs to be False to use contrast loss'

        # (N_rays, N_samples, N_joints)
        confd = preds['confd']
        if use_mirr:
            confd_v = preds['confd_ref']

        # (N_rays, N_samples)
        labels = ((preds['T_i'] * preds['alpha']) > 0).float()
        if use_mirr:
            labels_v = ((preds['T_i_ref'] * preds['alpha_ref']) > 0).float()

        # (N_rays, N_samples, N_joints)
        part_invalid = preds['part_invalid']
        part_valid = 1 - part_invalid
        
        if use_mirr:
            part_invalid_v = preds['part_invalid_ref']
            part_valid_v = 1 - part_invalid_v

        p = network.sigmoid(confd, part_invalid, mask_invalid=False, clamp=False)
        p_valid = (p * part_valid).sum(-1)
        loss_valid = (labels - p_valid).pow(2.).mean()

        if use_mirr:
            p_v = network.sigmoid(confd_v, part_invalid_v, mask_invalid=False, clamp=False)
            p_valid_v = (p_v * part_valid_v).sum(-1)
            loss_valid_v = (labels_v - p_valid_v).pow(2.).mean()
            # update
            loss_valid = loss_valid*args.r_weight + loss_valid_v*args.v_weight

        soft_softmax_loss = args.soft_softmax_loss_coef * loss_valid

        # TODO: replace this with caster.get_prob
        # only consider place with positive density
        '''stay with the real pose'''
        valid_count = ((part_valid.sum(-1) > 0) * labels).sum()
        sigmoid_act = (p * part_valid).sum(-1) * labels
        act_avg = sigmoid_act.detach().sum() / valid_count
        act_max = sigmoid_act.detach().max()

        loss_dict = {'soft_softmax_loss': soft_softmax_loss}

        return loss_dict, {'sigmoid_avg_act': act_avg, 'sigmoid_max_act': act_max}

    def _compute_volume_scale_loss(self):
        args = self.args
        # assume only use single_net = True
        caster = self.render_kwargs_train['ray_caster'].module
        if not hasattr(caster.network.graph_net, 'axis_scale'):
            return {}, {}
        scale = caster.network.graph_net.axis_scale
        init_scale = caster.network.graph_net.init_scale.to(scale.device)
        # limit the minimum size of the volume for numerical stability
        scale = scale.abs().clamp(min=init_scale * 0.05)
        scale_loss = torch.prod(scale, dim=-1).sum() * args.vol_scale_penalty

        scale_avg = scale.detach().mean(0)
        scale_x, scale_y, scale_z = scale_avg

        return {'vol_scale_loss': scale_loss}, {'opt_scale_x': scale_x, 'opt_scale_y': scale_y, 'opt_scale_z': scale_z}

    def _optim_step(self):
        '''
        step the optimizers for per-iteration parameter updates.
        '''

        self.optimizer.step()
        self.optimizer.zero_grad()

    def optimize(self, loss, i, popt_detach=False):
        '''step the optimizer
        loss: loss tensor to backward
        i: current iteration
        popt_detach: True if no need to update poses
        '''

        args = self.args
        # no pose optimization required, just backward.
        if popt_detach:
            loss.backward()
            total_norm, avg_norm = get_gradnorm(self.render_kwargs_train['ray_caster'])

            # debug necessary values
            if i < 5 and args.opt_v_color:
                print(f"{i} RGB: {self.render_kwargs_train['v_color_factor']} | grad:, {self.render_kwargs_train['v_color_factor'].grad}")

            self._optim_step()
            return {'total_norm': total_norm, 'avg_norm': avg_norm}

        # only need to retain comp. graph if step between pose update > 1
        retain_graph = args.opt_pose_cache and args.opt_pose_step > 1
        loss.backward(retain_graph=retain_graph)

        # reset gradient immediately after parameter update
        self._optim_step()

        total_norm, avg_norm = get_gradnorm(self.render_kwargs_train['ray_caster'])

        # update pose after enough gradient accumulation
        if i % args.opt_pose_step == 0:
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad()

            if args.opt_pose_cache:
                self.popt_kwargs['popt_layer'].update_cache()

        return {'total_norm': total_norm, 'avg_norm': avg_norm}

    def save_nerf(self, path, global_step):
        '''Save all state dict.
        path: path to save data
        global_step: global training step x()
        '''
        args = self.args
        ray_caster = self.render_kwargs_train['ray_caster']
        popt_sd, poptim_sd, popt_anchors = None, None, None
        if self.popt_kwargs is not None:
            popt_sd = self.popt_kwargs['popt_layer'].state_dict()
            poptim_sd = self.pose_optimizer.state_dict()
            popt_anchors = self.popt_kwargs['popt_anchors']

        torch.save({
            'global_step': global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'poseopt_layer_state_dict': popt_sd,
            'pose_optimizer_state_dict': poptim_sd,
            'poseopt_anchors': popt_anchors,
            **ray_caster.module.state_dict(),
        }, path)
        print('Saved checkpoints at', path)

    def save_popt(self, path, global_step):
        '''Save pose optimization outcomes
        '''
        args = self.args
        torch.save({'global_step': global_step,
                    'poseopt_layer_state_dict': self.popt_kwargs['popt_layer'].state_dict(),
                    'poseopt_anchors': self.popt_kwargs['popt_anchors'],
                    }, path)
        print('Saved pose at', path)

