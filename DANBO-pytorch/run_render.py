
import os
import pdb
import cv2
import time
import ipdb
import math
import h5py
import torch
import shutil
import imageio
import datetime
import platform
import numpy as np
import deepdish as dd
import time as timenow
import matplotlib.pyplot as plt


from tqdm import tqdm, trange
import torch.nn.functional as F
from run_nerf import render_path
from render.post_process import cca_image
from run_nerf import config_parser as nerf_config_parser

from core.pose_opt import load_poseopt_from_state_dict, pose_ckpt_to_pose_data
from core.load_data import generate_bullet_time, get_dataset
from core.raycasters import create_raycaster

from core.utils.evaluation_helpers import txt_to_argstring
from core.utils.compute_eval import eval_opt_kp
from core.utils.skeleton_utils import CMUSkeleton, smpl_rest_pose, get_smpl_l2ws, swap_mat, get_per_joint_coords
from core.utils.skeleton_utils import draw_skeletons_3d, rotate_x, rotate_y, \
                                      skeleton3d_to_2d, axisang_to_rot, rot_to_axisang
from pytorch_msssim import SSIM

# import sys
# sys.path.append("core/utils")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_dir = "."


GT_PREFIXES = {
    'h36m': 'data/h36m/',
    'h36m_zju': 'data/h36m_zju/',
    'surreal': None,
    'perfcap': 'data/',
    'mixamo': 'data/mixamo/',
    'mirror':'data/mirror',
}
n_joints = 26
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # nerf config
    parser.add_argument('--nerf_args', type=str, required=True,
                        help='path to nerf configuration (args.txt in log)')
    parser.add_argument('--ckptpath', type=str, required=True,
                        help='path to ckpt')

    # render config
    parser.add_argument('--render_res', nargs='+', type=int, default=[1080, 1920],
                        help='tuple of resolution in (H, W) for rendering')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset to render')
    parser.add_argument("--data_path", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument('--n_bullet', type=int, default=3,
                        help='no of bullet views to render')
    parser.add_argument('--n_bubble', type=int, default=10,
                        help='no of bubble views to render')
    parser.add_argument('--bullet_ang', type=int, default=360,
    help='angle of novel bullet view to render') 
    parser.add_argument('--bubble_ang', type=int, default=30,
    help='angle of novel bullet view to render') 

    parser.add_argument('--switch_cam', action='store_true', default=False,
                        help='replace (have both) real camera and virtual camera')
    parser.add_argument('--evaluate_pose', action='store_true', default=False,
                        help='evaluate the [refined/initial] pose for e.g trainset or images')
    parser.add_argument('--psnr_images', action='store_true', default=False,
                        help='evaluate images for psnr - use same img ids or offsets')
    parser.add_argument("--refined_path", type=str, default='',
                        help='path to refined poses in .tar')
    parser.add_argument("--render_folder", type=str, default='',
                        help='use existing render folder')
    parser.add_argument("--crop_folder", type=str, default='',
                        help='specify crop folder to add cropped images')
    parser.add_argument('--moved_model', action='store_true', default=False,
                        help='doing rendering with refined driving skeletons (moved from cluster A)')
    parser.add_argument('--outliers', action='store_false', default=True,
                         help='remove outliers during rendering? False if called')
    parser.add_argument('--mirror_retarget', action='store_true', default=False,
                        help='retargeting in mirror scene')
    parser.add_argument("--cca_with_mirr_bkgb", action='store_true',
                        help='use mirror background in rendering', default=False)
    
    
    # used to update the same parameters copied/passed over from run_nerf 
    parser.add_argument("--train_size", type=int, default=None,
                        help='no of samples for training')
    parser.add_argument("--data_size", type=int, default=None,
                        help='no of samples for training')
    parser.add_argument("--start_idx", type=int, default=None,
                        help='start id for rendering on sub sequence')
    parser.add_argument("--stop_idx", type=int, default=None,
                        help='stop id for rendering on sub sequence') 
    parser.add_argument("--render_interval", type=int, default=1,
                        help='rendering interval for rendering on sub sequence')
      
    parser.add_argument("--ray_tr_type", type=str, default='local',
                        help='use global view direction or skeleton-relative \
                        (transformed) local view direction')
    parser.add_argument("--idx_map_in_ckpt", action='store_false',
                        help='training idxs map exists in the model checkpoint. \
                        False if called', default=True)
    

                                                
    parser.add_argument("--use_mirr", action='store_true',
                        help='use both masks')
    parser.add_argument('--entry', type=str, required=True,
                        help='entry in the dataset catalog to render')
    parser.add_argument('--white_bkgd', action='store_true',
                        help='render with white background')
    parser.add_argument('--randomize_view_dir', action='store_true',
                        help='randomize the ray/view direction to check the effect of view direction on the rendered body, \
                        for debugging purposes', default=False)
    
    parser.add_argument('--render_type', type=str, default='retarget',
                        help='type of rendering to conduct')
    parser.add_argument("--render_mesh", action='store_true',
                        help='render mesh instead of images')
    parser.add_argument('--save_gt', action='store_true',
                        help='save gt frames')
    parser.add_argument('--save_crops', action='store_true',
                        help='crop existing renderings')
    parser.add_argument('--fps', type=int, default=14,
                        help='fps for video')
    parser.add_argument('--mesh_res', type=int, default=255,
                        help='resolution for marching cubes')
    parser.add_argument("--mesh_radius", type=float, default=1.8,
                        help='size of the volume that encapsulates the mesh for marching cube')
    parser.add_argument('--render_confd', action='store_true',
                        help='render probability map')
    parser.add_argument('--render_entropy', action='store_true',
                        help='render entropy heat map')
    # kp-related
    parser.add_argument('--render_refined', action='store_true',
                        help='render from refined poses')
    parser.add_argument('--subject_idx', type=int, default=0,
                        help='which subject to render (for MINeRF)')
    parser.add_argument('--top_expand_ratio', type=float, default=1.60,
                        help='margin for head joint e.g SMPL has no head joint')
    parser.add_argument('--extend_mm', type=float, default=250,
                        help='margin for extending box if too tight')

    # frame-related
    parser.add_argument('--train_len', type=int, default=0,
                        help='length of trainset')
    parser.add_argument('--selected_idxs', nargs='+', type=int, default=-1,
                        help='hand-picked idxs for rendering')
    parser.add_argument('--global_idxs', nargs='+', type=int, default=None,
                        help='idxs relative to the raw video frames')
    parser.add_argument('--selected_framecode', type=int, default=None,
                        help='hand-picked framecode for rendering')

    # camera-releated
    parser.add_argument("--cam_factor", type=float, default=None,
                        help='to zoom in/out')

    # saving
    parser.add_argument('--outputdir', type=str, default='render_output/',
                        help='output directory')
    parser.add_argument('--runname', type=str, required=True,
                        help='run name as an identifier ')

    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='to do evaluation at the end or not (only in bounding box)')

    parser.add_argument('--no_save', action='store_true',
                        help='no image saving operation')

    return parser

def load_nerf(args, nerf_args, skel_type=CMUSkeleton):

    ckptpath = args.ckptpath
    nerf_args.ft_path = args.ckptpath

    # some info are unknown/not provided in nerf_args
    # dig those out from state_dict directly
    nerf_sdict = torch.load(ckptpath)

    # update refined path to moved models
    if args.moved_model:
        if nerf_args.refined_path != args.refined_path:
            nerf_args.refined_path = args.refined_path
            assert nerf_args.refined_path == args.refined_path, "is the transferred refined path up-to-date?"

    # get data_attrs used for training the models
    data_attrs = get_dataset(nerf_args).get_meta()
    if 'framecodes.codes.weight' in nerf_sdict['network_fn_state_dict']:
        framecodes = nerf_sdict['network_fn_state_dict']['framecodes.codes.weight']
        data_attrs['n_views'] = framecodes.shape[0]
    # load poseopt_layer (if exist)
    popt_layer = None
    if nerf_args.opt_pose:
        popt_layer = load_poseopt_from_state_dict(nerf_sdict)
    nerf_args.finetune = True

    render_kwargs_train, render_kwargs_test, _, grad_vars, _, _ = create_raycaster(nerf_args, data_attrs, device=device)

    # freeze weights
    for grad_var in grad_vars:
        grad_var.requires_grad = False

    render_kwargs_test['ray_caster'] = render_kwargs_train['ray_caster']
    render_kwargs_test['ray_caster'].eval()

    rest_pose = None
    if hasattr(render_kwargs_test['ray_caster'].module, 'rest_poses'):
        rest_pose = render_kwargs_test['ray_caster'].module.rest_poses
        if len(rest_pose.shape) > 2:
            rest_pose = rest_pose[args.subject_idx]

    return render_kwargs_test, popt_layer, rest_pose

def load_render_data(args, nerf_args, poseopt_layer=None, rest_pose=None, opt_framecode=True):
    # TODO: note that for models trained on SPIN data, they may not react well
    catalog = init_catalog(args)[args.dataset][args.entry]
    render_data = catalog.get(args.render_type, {})
    data_h5 = catalog['data_h5']
    
    kps, bones = None, None

    if args.use_mirr or args.switch_cam:
        print("Also loading virtual camera!")
        # catalog = init_catalog(args)[args.dataset][args.entry]
        if 'data_h5_v' in catalog: data_h5_v = catalog['data_h5_v']

    # to real cameras (due to the extreme focal length they were trained on..)
    # TODO: add loading with opt pose option
    if rest_pose is None:
        if poseopt_layer is not None:
            rest_pose = poseopt_layer.get_rest_pose().cpu().numpy()[0]
            print("Load rest pose for poseopt!")
        else:
            try:
                rest_pose = dd.io.load(data_h5, '/rest_pose')
                print(f"Load rest pose from h5: {data_h5}!")
            except:
                rest_pose = smpl_rest_pose * dd.io.load(data_h5, '/ext_scale')
                print("Load smpl rest pose!")
    else:
        print('rest pose is provided.')

    #"""
    if args.render_refined:
        if 'refined' in catalog:
            print(f"loading refined poses from {catalog['refined']}")
            # poseopt_layer = load_poseopt_from_state_dict(torch.load(catalog['refined']))
            kps, bones = pose_ckpt_to_pose_data(catalog['refined'], legacy=False, idx_map_in_ckpt=args.idx_map_in_ckpt)[:2]
        else:
            with torch.no_grad():
                bones = poseopt_layer.get_bones().cpu().numpy()
                kps = poseopt_layer(np.arange(len(bones)))[0].cpu().numpy()

        if render_data is not None:
            render_data['refined'] = [kps, bones]
            render_data['idx_map'] = catalog.get('idx_map', None)
        else:
            render_data = {'refined': [kps, bones],
                           'idx_map': catalog['idx_map']}
    else:
        render_data['idx_map'] = catalog.get('idx_map', None)
    #"""

    render_data['args'] = args
    pose_keys = ['/kp3d', '/bones']
    # Do partial load here!
    # Need:
    # 1. kps: for root location
    # 2. bones: for bones
    # 3. camera stuff: focals, c2ws, ... etc
    refined = None if not args.render_refined else catalog.get('refined', None)
    c2ws, centers, focals, H, W = read_camera_data(data_h5)

    if kps is None and bones is None:
        kps, bones, rest_pose_h5 = read_pose_data(data_h5, refined=refined, poseopt_layer=poseopt_layer) 
    else:
        # get init restpose
        _, _, rest_pose_h5 = read_pose_data(data_h5, refined=None, poseopt_layer=None) 
    
    rest_pose = rest_pose if rest_pose is not None else rest_pose_h5
    # a map for poseopt kp. sometimes the kp idxs will be different from the poseopt ones
    render_data['idx_map'] = catalog.get('idx_map', None)

    if args.switch_cam:
        A_mirr =  dd.io.load(data_h5_v, '/A_dash')
        n_size = A_mirr.shape[0]

        '''flip the camera about the x-axis, to simulate flipped image'''
        v_cam_3_4 = np.concatenate((swap_mat(A_mirr[...,:3,:3]), A_mirr[...,:3,3:4]),2)
        last_row = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), [n_size,1,1])
        v_cam =  np.concatenate([v_cam_3_4, last_row], axis=1)

        print("c2ws", c2ws[0])
        print("v_cam", v_cam[0])
    
    h5file = h5py.File(data_h5, 'r')
    _, H, W, _ = h5file['img_shape']
    h5file.close()
    # _, H, W, _ = dd.io.load(data_h5, ['/img_shape'])[0]

    # handle resolution
    if args.render_res is not None and not args.mirror_retarget:
        assert len(args.render_res) == 2, "Image resolution should be in (H, W)"
        H_r, W_r = args.render_res
        # TODO: only check one side for now ...
        scale = float(H_r) / float(H)
        focals *= scale
        H, W = H_r, W_r
        if centers is not None:
            centers *= scale


    bones_keep = bones.copy()

    # Load data based on type:
    bones = None
    root = None
    bg_imgs, bg_indices = None, None
    if args.render_type in ['retarget', 'mesh']:
        print(f'Load data for retargeting!')
        kps, skts, c2ws, cam_idxs, focals, bones, centers = load_retarget(
                                                                kps, bones_keep, c2ws, focals,
                                                                rest_pose,
                                                                cam_factor=args.cam_factor,
                                                                **render_data)
    elif args.render_type == 'bullet':
        if args.evaluate_pose:
            print(f'Load data for evaluation!')
            kps, skts, c2ws, cam_idxs, focals, bones = eval_bullettime_kps(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    #centers=centers_n,
                                                                    **render_data)
        elif args.psnr_images:
                print(f'Load data to generate images.')
                kps, skts, c2ws, cam_idxs, focals, bones = generate_psnr_imgs(data_h5, c2ws, focals,
                                                                        rest_pose, pose_keys,
                                                                        centers=centers,
                                                                        **render_data)
        else:
            
            if args.switch_cam:
                print(f'Load data for two views.')

                '''only c2ws changes'''
                v_focals = focals.copy()
                v_centers = centers.copy()
                print("working on real...")
                kps, skts, c2ws, cam_idxs, focals, bones, centers, root = load_bullettime(data_h5, c2ws, focals,
                                                                        rest_pose, pose_keys,
                                                                        centers=centers,
                                                                        **render_data)
                print("working on virtual...")
                _, _, c2ws_virt, _, _, _, centers_virt, _ = load_bullettime(data_h5, v_cam, v_focals,
                                                                        rest_pose, pose_keys,
                                                                        centers=v_centers,
                                                                        real = False,
                                                                        **render_data)
                """combined real and virt rendering"""
                c2ws = np.concatenate([c2ws, c2ws_virt])

                # TODO: double check if new virtual center is an issue, or not needed
                centers = np.concatenate([centers, centers_virt]) 

            else:
                kps, skts, c2ws, cam_idxs, focals, bones, centers, root = load_bullettime(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    centers=centers,
                                                                    **render_data)

    elif args.render_type == 'poserot':

        kps, skts, bones, c2ws, cam_idxs, focals = load_pose_rotate(data_h5, c2ws, focals,
                                                                    rest_pose, pose_keys,
                                                                    **render_data)
    elif args.render_type == 'interpolate':
        print(f'Load data for pose interpolation!')
        kps, skts, c2ws, cam_idxs, focals = load_interpolate(data_h5, c2ws, focals,
                                                             rest_pose, pose_keys,
                                                             **render_data)

    elif args.render_type == 'animate':
        print(f'Load data for pose animate!')
        kps, skts, c2ws, cam_idxs, focals, bones = load_animate(data_h5, c2ws, focals,
                                                                rest_pose, pose_keys,
                                                                **render_data)
    elif args.render_type == 'bubble':
        print(f'Load data for bubble camera!')
        kps, skts, c2ws, cam_idxs, focals, bones, centers = load_bubble(data_h5, c2ws, focals,
                                                        rest_pose, pose_keys,
                                                        centers=centers,
                                                        **render_data)
    elif args.render_type == 'selected':
        selected_idxs = np.array(args.selected_idxs)
        kps, skts, c2ws, cam_idxs, focals, bones, centers = load_selected(
                                                                data_h5, c2ws, focals,
                                                                rest_pose, pose_keys,
                                                                selected_idxs, 
                                                                centers=centers, 
                                                                **render_data)
    elif args.render_type.startswith('val'):
        print(f'Load data for valditation!')
        is_surreal = ('surreal_val' in data_h5) or (args.entry == 'hard' and 'surreal' in data_h5)
        is_neuralbody = 'neuralbody' in args.dataset
        is_h36m_zju = 'h36m_zju' in args.dataset
        kps, skts, c2ws, cam_idxs, focals, bones, centers = load_retarget(
                                                                kps, bones, c2ws, focals,
                                                                rest_pose,
                                                                is_surreal=is_surreal,
                                                                centers=centers,
                                                                is_neuralbody=is_neuralbody,
                                                                cam_factor=args.cam_factor,
                                                                is_h36m_zju=is_h36m_zju,
                                                            **render_data)
        if is_h36m_zju:
            bg_imgs = np.zeros((1, H, W, 3), dtype=np.float32)
            bg_indices = np.zeros((len(cam_idxs),), dtype=np.int32)
            cam_idxs = cam_idxs * 0 -1
        elif not is_surreal:
            try:
                bg_imgs = dd.io.load(data_h5, '/bkgds').astype(np.float32).reshape(-1, H, W, 3) / 255.
                bg_indices = dd.io.load(data_h5, '/bkgd_idxs')[cam_idxs].astype(np.int64)
            except:
                bg_imgs = np.zeros((1, H, W, 3), dtype=np.float32)
                bg_indices = np.zeros((len(cam_idxs),), dtype=np.int32)
            if args.dataset == 'perfcap':
                # treat frame properly
                h5_file = h5py.File(data_h5, 'r')
                last_idx = h5_file['img_shape'][0] - catalog['val_start'] - 1
                h5_file.close()
                cam_idxs = cam_idxs * 0 + last_idx 
                # cam_idxs = np.arange(len(cam_idxs))
            else:
                cam_idxs = cam_idxs * 0 -1

    elif args.render_type == 'correction':
        print(f'Load data for visualizing correction!')
        kps, skts, c2ws, cam_idxs, focals = load_correction(data_h5, c2ws, focals,
                                                            rest_pose, pose_keys,
                                                            **render_data)
        bg_imgs = dd.io.load(data_h5, '/bkgds').astype(np.float32).reshape(-1, H, W, 3) / 255.
        bg_indices = dd.io.load(data_h5, '/bkgd_idxs')[cam_idxs].astype(np.int64)
    elif args.render_type == 'inspvol':
        print(f'Load data for retargeting!')
        # TODO: temporary name
        kps, skts, c2ws, cam_idxs, focals, bones = load_inspect_vol(data_h5, c2ws, focals,
                                                                 rest_pose, pose_keys,
                                                                 **render_data)
    else:
        raise NotImplementedError(f'render type {args.render_type} is not implemented!')

    # added for rendering background image
    if not args.white_bkgd:
        _, bg_H, bg_W, bg_C = dd.io.load(data_h5, '/img_shape')
        bg_imgs = dd.io.load(data_h5, '/bkgds').astype(np.float32).reshape(-1, bg_H, bg_W, bg_C) / 255.
        bg_indices = dd.io.load(data_h5, '/bkgd_idxs')[cam_idxs].astype(np.int64)

        # should in case (your retargeting), use image dimension and background from driving pose data.


    gt_paths, gt_mask_paths = None, None
    is_gt_paths = True
    if args.save_gt or args.eval:
        if args.render_type == 'retarget':
            extract_idxs = cam_idxs
        elif 'selected_idxs' in render_data:
            extract_idxs = render_data['selected_idxs']
        else:
            extract_idxs = selected_idxs
        # unique without sort
        unique_idxs = np.unique(extract_idxs, return_index=True)[1]
        extract_idxs = np.array([extract_idxs[idx] for idx in sorted(unique_idxs)])
        try:
            gt_to_mask_map = init_catalog(args)[args.dataset].get('gt_to_mask_map', ('', ''))
            gt_paths = np.array(dd.io.load(data_h5, '/img_path'))[extract_idxs]
            gt_prefix = GT_PREFIXES[args.dataset]
            gt_paths = [os.path.join(gt_prefix, gt_path) for gt_path in gt_paths]
            gt_mask_paths = [p.replace(*gt_to_mask_map) for p in gt_paths]
        except:
            print('gt path does not exist, grab images directly!')
            is_gt_paths = False

            gt_paths = dd.io.load(data_h5, '/imgs', sel=dd.aslice[extract_idxs, ...])
            gt_mask_paths = dd.io.load(data_h5, '/masks', sel=dd.aslice[extract_idxs, ...])

    # handle special stuff
    if args.selected_framecode is not None:
        cam_idxs[:] = args.selected_framecode

    # perfcap have a peculiar camera setting
    if args.dataset == 'perfcap':
        c2ws[..., :3, -1] /= 1.05

    subject_idxs = None
    if nerf_args.nerf_type.startswith('mi'):
        subject_idxs = (np.ones((len(kps),)) * args.subject_idx).astype(np.int64)

    ret_dict = {'kp': kps, 'skts': skts, 'render_poses': c2ws,
                'cams': cam_idxs if opt_framecode else None,
                'hwf': (H, W, focals),
                'centers': centers,
                'bones': bones,
                'bg_imgs': bg_imgs,
                'bg_indices': bg_indices,
                'subject_idxs': subject_idxs,
                'root':root}

    gt_dict = {'gt_paths': gt_paths,
               'gt_mask_paths': gt_mask_paths,
               'is_gt_paths': is_gt_paths,
               'bg_imgs': bg_imgs,
               'bg_indices': bg_indices}

    return ret_dict, gt_dict, render_data['selected_idxs']

def init_catalog(args, n_bullet=10):
    n_bullet = args.n_bullet
    bullet_ang = args.bullet_ang
    bubble_ang = args.bubble_ang

    RenderCatalog = {
        'h36m': None,
        'h36m_zju': None,
        'surreal': None,
        'mirror': None,
        'perfcap': None,
        'mixamo': None,
        '3dhp': None,
    }

    def load_idxs(path):
        if not os.path.exists(path):
            print(f'Index file {path} does not exist.')
            return []
        return np.load(path)
    def set_dict(selected_idxs, **kwargs):
        return {'selected_idxs': np.array(selected_idxs), **kwargs}

    # H36M
    s9_idx = [121, 500, 1000, 1059, 1300, 1600, 1815, 2400, 3014, 3702, 4980]
    h36m_s9 = {
        'data_h5': 'data/h36m/S9_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/h36m/s9_sub64_500k.tar',
        'retarget': set_dict(s9_idx, length=5),
        'bullet': set_dict(s9_idx,
                           n_bullet=n_bullet, undo_rot=False,
                           center_cam=True),
        'interpolate': set_dict(s9_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'correction': set_dict(load_idxs('data/h36m/S9_top50_refined.npy')[:1], n_step=30),
        'animate': set_dict([1000, 1059, 2400], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([17,19,21,23]), from_rest=True, undo_rot=False),
        'bubble': set_dict(s9_idx, n_step=30),
        'poserot': set_dict(np.array([1000])),
        'val': set_dict(load_idxs('data/h36m/S9_val_idxs.npy'), length=1, skip=1),
    }
    s11_idx = [213, 656, 904, 1559, 1815, 2200, 2611, 2700, 3110, 3440, 3605]
    h36m_s11 = {
        'data_h5': 'data/h36m/S11_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/h36m/s11_sub64_500k.tar',
        'retarget': set_dict(s11_idx, length=5),
        'bullet': set_dict(s11_idx, n_bullet=n_bullet, undo_rot=True,
                           center_cam=True),
        'interpolate': set_dict(s11_idx, n_step=10, undo_rot=True,
                                center_cam=True),

        'correction': set_dict(load_idxs('data/h36m/S11_top50_refined.npy')[:1], n_step=30),
        'animate': set_dict([2507, 700, 900], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([3,6,9,12,15,16,18])),
        'bubble': set_dict(s11_idx, n_step=30),
        'val': set_dict(load_idxs('data/h36m/S11_val_idxs.npy'), length=1, skip=1),
    }

    # MIRROR
    view = args.data_path.split("_cam_")[1][0]
    uniq_id = comb = args.data_path.split("_cam_")[0].split("/")[-1]

    if args.evaluate_pose:
        easy_idx = np.arange(0, args.train_len, args.render_interval)

    elif args.psnr_images and args.eval:

        import pickle
        with open(f'{args.data_path}/calib{view}_20000iter_May_11.pickle', 'rb') as handle:
            from_pickle = pickle.load(handle)

        mirrr_raw_idxs = from_pickle['chosen_frames'][:args.train_len]
        
        n = len(mirrr_raw_idxs)
        offset_indices = []

        for i in range(0,n,args.render_interval):
            index = mirrr_raw_idxs.index(i)
            offset_indices.append(index)

        easy_idx = offset_indices #[-1:]
        args.selected_idxs = easy_idx 
        # pick a subset that is common to both
        pdb.set_trace()

    elif args.psnr_images and not args.eval:
        """using common set"""
        import pickle#5 as pickle

        subj3_ID = "e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0"
        uniq_id = comb

        if uniq_id==subj3_ID:
            # pick a subset that is common to both (both data currrently stored in vanilla Tim data path) though we are in mirror now
            s3_pickle_path = f"data/vanilla/0/e1531958-26bb-4f46-b3b4-bad1910798c9_cam_0"
    
            # step 2 (example) reconstruction data
            with open(f'{s3_pickle_path}/calib{view}_20000iter_step2_3d.pickle', 'rb') as handle:
                mirror_pickle = pickle.load(handle)

            # spin estimates pkl file
            with open(f'{s3_pickle_path}/spin_output.pkl', 'rb') as handle:
                spin_pickle = pickle.load(handle)

            mirr_frames = mirror_pickle["chosen_frames"]
            spin_frames = spin_pickle["valid_idxs"]
            if "vanilla" in args.data_path:
                target = spin_frames.tolist()
            elif "mirror" in args.data_path:
                target = mirr_frames
            else:
                print("what is the target method?")

            common_global_idxs = sorted(set(mirr_frames) - set(spin_frames))
            selected_global_idxs = common_global_idxs #[0:1]#[::20] #[1::2]

            n = len(selected_global_idxs)
            offset_local_indices = []

            for idx in selected_global_idxs:
                    # you need specific global idx
                    if int(args.selected_idxs[0]) != -1:
                        specific_global_idx = args.selected_idxs

                        if idx not in specific_global_idx:
                            continue
                    index = target.index(idx)
                    offset_local_indices.append(index)

            # render batch by batch
            if args.start_idx is not None and args.stop_idx is not None:
                range_idx = np.arange(args.start_idx, args.stop_idx, args.render_interval)

                offset_local_indices = np.array(offset_local_indices)[range_idx]
                selected_global_idxs = np.array(selected_global_idxs)[range_idx]

            else:
                # assume same no of frames for mirror vs vanilla
                offset_local_indices = selected_global_idxs = np.arange(0, args.train_len, args.render_interval)#[0:1]

            # easy_idx 
            args.selected_idxs = easy_idx = offset_local_indices
            args.global_idxs = selected_global_idxs 

        elif args.selected_idxs != [-1]: 
            # specific idx
            easy_idx = args.selected_idxs 
            args.global_idxs = args.selected_idxs

        else: 
            # render part or all of train set
            if args.start_idx is None and args.stop_idx is None:
                easy_idx = np.arange(0, args.train_len)
            else:
                easy_idx = np.arange(args.start_idx, args.stop_idx, args.render_interval)
            
            if args.render_type=="mesh":
                print("please (add code for subject 3) and use right idxs")
   
            args.selected_idxs = easy_idx
            args.global_idxs = easy_idx


    mirror_easy = {
        'data_h5': args.data_path + '/mirror_train_h5py.h5',,
        'data_h5_v': args.data_path + '/v_mirror_train_h5py.h5',
        'retarget': set_dict(easy_idx, length=25, skip=2, center_kps=True),
        'bullet': set_dict(easy_idx, n_bullet=args.n_bullet, bullet_ang=args.bullet_ang),
        'bubble': set_dict(easy_idx, n_step=args.n_bubble, bubble_ang=args.bubble_ang),
        'poserot': set_dict(easy_idx, n_bullet=args.n_bullet),
        'mesh': set_dict(easy_idx, length=1, skip=1),
        'refined': args.refined_path
    }

    # TODO: create validation indices
    test_val_idx = np.arange(args.train_size)[-10:] # use last 10 images on train size 
    mirror_val = {
        'data_h5': args.data_path + '/mirror_val_h5py.h5',
        'data_h5_v': args.data_path + '/v_mirror_train_h5py.h5',
        'val': set_dict(test_val_idx, length=1, skip=1),
        'val2': set_dict(test_val_idx, length=1, skip=1),
    }
    
    hard_idx = [140, 210, 280, 490, 560, 630, 700, 770, 840, 910]
    mirror_hard = {
        'data_h5': args.data_path + '/mirror_train_h5py.h5', #'/tim_train_h5py.h5', #1620
        'data_h5_v': args.data_path + '/v_mirror_train_h5py.h5', #1620
        'retarget': set_dict(hard_idx, length=60, skip=5, center_kps=True),
        #'bullet': set_dict([190,  210,  230,  490,  510,  530,  790,  810,  830,  910,  930, 950, 1090, 1110, 1130],
        #                   n_bullet=n_bullet, center_kps=True, center_cam=False),
        'bubble': set_dict(hard_idx, n_step=30),
        #'val': set_dict(np.array([1200 * i + np.arange(420, 700)[::5] for i in range(0, 9, 2)]).reshape(-1), length=1, skip=1),
        'mesh': set_dict([0], length=1, skip=1),
    }


    # H36M_ZJU
    s9_idx = [0]
    # this is novel_view + novel_pose, matching their publicly released data
    h36m_zju_num_eval = {'S1': 34, 'S5': 64, 'S6': 39, 'S7': 84,
                         'S8': 57, 'S9': 67, 'S11': 48}

    # for full h36m unseen poses
    #h36m_zju_num_eval = {'S1': 49, 'S5': 127, 'S6': 83, 'S7': 200,
    #                      'S8': 87, 'S9': 133, 'S11': 82}
    meshviz = {'S1': 33, 'S5': 57, 'S6': 34, 'S7': 78,
              'S8': 53, 'S9': 62, 'S11': 45}
    #meshviz = {'S1': 33, 'S5': 57, 'S6': 34, 'S7': 78,
    #           'S8': 53, 'S9': 20, 'S11': 45}
    # Note: they subsample every 6 frame
    h36m_zju_s = {
        f'{k}': {
            #'data_h5': f'data/h36m_zju/{k}_train.h5' ,
            #'data_h5': f'data/h36m_zju/{k}_anim.h5' ,
            'data_h5': f'data/h36m_zju/{k}_test.h5' ,
            #'val': set_dict(np.arange(h36m_zju_num_eval[k])[meshviz[k]:meshviz[k]+1], skip=1, length=1), 
            'val': set_dict(np.arange(h36m_zju_num_eval[k]), skip=1, length=1), 
            'bullet': set_dict(np.arange(h36m_zju_num_eval[k])[meshviz[k]:meshviz[k]+1], n_bullet=3, center_cam=False, center_kps=True), 
            'animate': set_dict([0, 20, 40], n_step=10, center_cam=False, center_kps=True,
                            joints=np.arange(24), from_rest=True, undo_rot=False),
        } for k in ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    }
    h36m_zju_s9 = {
        'data_h5': 'data/h36m_zju/S9_train.h5',
        'bullet': set_dict(np.array([560]), n_bullet=6, center_cam=False, center_kps=True),
        'retarget': set_dict(np.arange(780), length=1, skip=1),
    }

    h36m_zju_s8 = {
        'data_h5': 'data/h36m_zju/S8_train.h5',
        'bullet': set_dict(np.array([70]), n_bullet=90, center_cam=False, center_kps=True),
        'retarget': set_dict(np.arange(780), length=1, skip=1),
        'bubble': set_dict(np.array([633]), n_step=30),
    }
    # 33, 115, 225
    h36m_zju_s7 = {
        'data_h5': 'data/h36m_zju/S7_train.h5',
        'bullet': set_dict(np.array([22]), n_bullet=90, center_cam=False, center_kps=True),
        'retarget': set_dict(np.arange(780), length=1, skip=1),
    }
    """
    h36m_zju_s1 = {
        'data_h5': 'data/h36m_zju/S1_test.h5',
        'val': set_dict(np.arange(49)[:1], skip=1, length=1),
    }
    """
 
    # SURREAL
    #easy_idx = [10, 70, 350, 420, 490, 910, 980, 1050]
    easy_idx = [10]
    surreal_val = {
        'data_h5': 'data/surreal/surreal_val_h5py.h5',
        'retarget': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy')[:300][::3], length=1, skip=1),
        'val': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy')[::3], length=1, skip=1),
        #'val': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy')[428:600][:50], length=1, skip=1),
        #'val2': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy')[:300], length=1, skip=1),
    }
    surreal_easy = {
        'data_h5': 'data/surreal/surreal_train_h5py.h5',
        'retarget': set_dict(easy_idx, length=25, skip=2, center_kps=True),
        'bullet': set_dict(easy_idx, n_bullet=n_bullet),
        'bubble': set_dict(easy_idx, n_step=30),
    }
    #hard_idx = [490, 560, 630, 700, 770, 840, 910]
    hard_idx = [490, 560, 770, 840, 910]#, 560, 630, 700, 770, 840, 910]
    #hard_idx = [0, 70, 140, 210, 280, 490, 560, 630, 700, 770, 840, 910]
    #hard_idx = [0, 70, 140, 210, 280, 350, 490, 560, 630, 700, 770, 840, 910, 980, 1050, 1120]
    #joints=np.array([16,17,18,19,20,21,22,23])
    #joints = np.array([1,2,4,5,7,8])
    #joints = np.array([3,6,9,12,15,17,19,21,23]) # right arms
    #hard_idx = [870] # kick
    #hard_idx = [350]
    #hard_idx = [480]  # cartwheel
    joints = np.array([12,15,17,19,21,23]) # right arms
    surreal_hard = {
        'data_h5': 'data/surreal/surreal_train_h5py.h5',
        #'retarget': set_dict(hard_idx, length=60, skip=4, center_kps=True),
        'retarget': set_dict(hard_idx, length=30, skip=2, center_kps=True),
        #'retarget': set_dict(hard_idx, length=30, skip=1, center_kps=True),
        'bullet': set_dict([490,510],#[190,  210,  230,  490,  510,  530,  790,  810,  830,  910,  930, 950, 1090, 1110, 1130][:5],
                           n_bullet=n_bullet, center_kps=True, center_cam=False),
        'bubble': set_dict(hard_idx, n_step=30),
        'val': set_dict(np.array(1200 + np.array(hard_idx)).reshape(-1), length=60, skip=1),
        'mesh': set_dict([930], length=1, skip=1),
        'inspvol': set_dict([0, 490], n_bullet=n_bullet, center_kps=True, center_cam=False),
        'animate': set_dict([0, 510], n_step=1, center_cam=False, center_kps=True,
                            joints=joints, from_rest=True, undo_rot=False, rotate_y_ang=0.),
    }

    # PerfCap
    perfcap_skip = 5
    # weipeng_idx = [0, 50, 100, 150, 200, 250, 300, 350, 430, 480, 560,
    #               600, 630, 660, 690, 720, 760, 810, 850, 900, 950, 1030,
    #               1080, 1120]
    weipeng_idx = [640]
    perfcap_weipeng = {
        'data_h5': 'data/MonoPerfCap/Weipeng_outdoor/Weipeng_outdoor_processed_h5py.h5',
        'refined': 'data/MonoPerfCap/Weipeng_outdoor/weipeng_refined_new.tar',
        #'refined': 'data/MonoPerfCap/Weipeng_outdoor/danbo_refined_wp.tar',
        'retarget': set_dict(weipeng_idx, length=30, skip=2),
        'bullet': set_dict(weipeng_idx, n_bullet=n_bullet),
        'interpolate': set_dict(weipeng_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(weipeng_idx, n_step=30),
        'val_start': 230,
        'val': set_dict(np.arange(1151)[-230:][::perfcap_skip], length=1, skip=1),
        #'val': set_dict(np.arange(1151)[-230+2:][::perfcap_skip], length=1, skip=1),
        'animate': set_dict([300, 480, 700], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([1,4,7,10,17,19,21,23])),
    }

    nadia_idx = [0, 65, 100, 125, 230, 280, 410, 560, 600, 630, 730, 770,
                 830, 910, 1010, 1040, 1070, 1100, 1285, 1370, 1450, 1495,
                 1560, 1595]
    perfcap_nadia = {
        'data_h5': 'data/MonoPerfCap/Nadia_outdoor/Nadia_outdoor_processed_h5py.h5',
        'refined': 'data/MonoPerfCap/Nadia_outdoor/nadia_refined_new.tar',
        #'refined': 'data/MonoPerfCap/Nadia_outdoor/danbo_refined_nd.tar',
        'retarget': set_dict(nadia_idx, length=30, skip=2),
        'bullet': set_dict(nadia_idx, n_bullet=n_bullet),
        'interpolate': set_dict(nadia_idx, n_step=10, undo_rot=True,
                                center_cam=True, center_kps=True),
        'bubble': set_dict(nadia_idx, n_step=30),
        'animate': set_dict([280, 410, 1040], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([1,2,4,5,7,8,10,11])),
        #'val': set_dict(np.arange(1635)[-327:], length=1, skip=1),
        'val_start': 327,
        #'val': set_dict(np.arange(1635)[-327+2:][::perfcap_skip], length=1, skip=1),
        'val': set_dict(np.arange(1635)[-327:][::perfcap_skip], length=1, skip=1),
    }

    # Mixamo
    james_idx = [20, 78, 118, 333, 1149, 3401, 2221, 4544]
    mixamo_james = {
        'data_h5': 'data/mixamo/James_processed_h5py.h5',
        'idx_map': load_idxs('data/mixamo/James_selected.npy'),
        'refined': 'data/mixamo/james_refined_500ktv.tar',
        'retarget': set_dict(james_idx, length=30, skip=2),
        'bullet': set_dict(james_idx, n_bullet=n_bullet, center_cam=True, center_kps=True),
        'interpolate': set_dict(james_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(james_idx, n_step=30),
        'animate': set_dict([3401, 1149, 4544], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([18,19,20,21,22,23])),
        'mesh': set_dict([20, 78], length=1, undo_rot=False),
    }

    archer_idx = [158, 374, 672, 1886, 2586, 2797, 4147, 4465]
    mixamo_archer = {
        'data_h5': 'data/mixamo/Archer_processed_h5py.h5',
        'idx_map': load_idxs('data/mixamo/Archer_selected.npy'),
        'refined': 'data/mixamo/archer_refined_500ktv.tar',
        'retarget': set_dict(archer_idx, length=60, skip=2),
        'bullet': set_dict(archer_idx, n_bullet=n_bullet, center_cam=True, center_kps=True),
        'interpolate': set_dict(archer_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(archer_idx, n_step=30),
        'animate': set_dict([1886, 2586, 4465], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([18,19,20,21,22,23])),
        'mesh': set_dict([4147], length=1, undo_rot=False),
    }

    # NeuralBody
    nb_subjects = ['315', '377', '386', '387', '390', '392', '393', '394']
    # TODO: hard-coded: 6 views
    nb_idxs = np.arange(len(np.concatenate([np.arange(1, 31), np.arange(400, 601)])) * 6)
    nb_dict = lambda subject: {'data_h5': f'data/zju_mocap/{subject}_test_h5py.h5',
                               'val': set_dict(nb_idxs, length=1, skip=1)}

    RenderCatalog['h36m'] = {
        'S9': h36m_s9,
        'S11': h36m_s11,
        'gt_to_mask_map': ('imageSequence', 'Mask'),
    }
    # only for evaluation
    RenderCatalog['h36m_zju'] = {
        **h36m_zju_s,
    }
    # for rendering
    RenderCatalog['h36m_zju_render'] = {
        'S9': h36m_zju_s9,
        'S8': h36m_zju_s8,
        'S7': h36m_zju_s7,
    }
    RenderCatalog['surreal'] = {
        'val': surreal_val,
        'easy': surreal_easy,
        'hard': surreal_hard,
    }
    RenderCatalog['mirror'] = {
        'val': mirror_val,
        'easy': mirror_easy,
        'hard': mirror_hard,
    }
    RenderCatalog['perfcap'] = {
        'weipeng': perfcap_weipeng,
        'nadia': perfcap_nadia,
        'gt_to_mask_map': ('images', 'masks'),
    }
    RenderCatalog['mixamo'] = {
        'james': mixamo_james,
        'archer': mixamo_archer,
    }
    RenderCatalog['neuralbody'] = {
        f'{subject}': nb_dict(subject) for subject in nb_subjects
    }

    return RenderCatalog

def find_idxs_with_map(selected_idxs, idx_map):
    if idx_map is None:
        return selected_idxs
    match_idxs = []
    for sel in selected_idxs:
        for i, m in enumerate(idx_map):
            if m == sel:
                match_idxs.append(i)
                break
    return np.array(match_idxs)

def load_correction(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, n_step=8, refined=None, center_kps=False,
                    idx_map=None):
    assert refined is not None
    c2ws = c2ws[selected_idxs]
    focals = focals[selected_idxs]

    init_kps, init_bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
    refined_kps, refined_bones = refined
    refined_kps = refined_kps[selected_idxs]
    refined_bones = refined_bones[selected_idxs]

    w = np.linspace(0, 1.0, n_step, endpoint=False).reshape(-1, 1, 1)
    interp_bones = []
    for i, (init_bone, refine_bone) in enumerate(zip(init_bones, refined_bones)):
        # interpolate from initial bone to refined bone
        interp_bone = init_bone[None] * (1-w) + refine_bone[None] * w
        interp_bones.append(interp_bone)
    interp_bones = np.concatenate(interp_bones, axis=0)
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in interp_bones])
    l2ws = l2ws.reshape(len(selected_idxs), n_step, 26, 4, 4)
    l2ws[..., :3, -1] += refined_kps[:, None, :1, :]
    l2ws = l2ws.reshape(-1, 26, 4, 4)
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    c2ws = c2ws[:, None].repeat(n_step, 1).reshape(-1, 4, 4)
    focals = focals[:, None].repeat(n_step, 1).reshape(-1) # .reshape(-1,2)
    cam_idxs = selected_idxs[:, None].repeat(n_step, 1).reshape(-1)

    return kps, skts, c2ws, cam_idxs, focals

def load_retarget(kps, bones, c2ws, focals, rest_pose,
                  selected_idxs, length, skip=1, centers=None, pose_h5=None, pose_keys=None,
                  center_kps=False, idx_map=None, is_surreal=False, refined=None,
                  cam_factor=None, undo_rot=False, is_neuralbody=False, 
                  is_h36m_zju=False, args=None):


    l = length
    # TODO: old code
    if skip > 1 or l > 1:
        selected_idxs = np.concatenate([np.arange(s, min(s+l, len(c2ws)))[::skip] for s in selected_idxs])

    # uncomment these two lines for mixamo rendering
    #if refined is not None:
    #    selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    c2ws = c2ws[selected_idxs]

    if cam_factor is not None:
        c2ws[..., :3, -1] *= cam_factor

    # TODO: old code
    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]
    if centers is not None:
        centers = centers[selected_idxs]
    cam_idxs = selected_idxs

    """
    if refined is None:
        if (not is_surreal and not is_neuralbody) or is_h36m_zju:
            kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        elif is_neuralbody:
            kps, bones = dd.io.load(pose_h5, pose_keys)
            # TODO: hard-coded, could be problematic
            kps = kps.reshape(-1, 1, 26, 3).repeat(6, 1).reshape(-1, 26, 3)
            bones = bones.reshape(-1, 1, 26, 3).repeat(6, 1).reshape(-1, 26, 3)
        else:
            kps, bones = dd.io.load(pose_h5, pose_keys)
            kps = kps[None].repeat(9, 0).reshape(-1, 26, 3)[selected_idxs]
            bones = bones[None].repeat(9, 0).reshape(-1, 26, 3)[selected_idxs]
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        #selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    """
    kps = kps[selected_idxs].copy()
    bones = bones[selected_idxs].copy()

    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    return kps, skts, c2ws, cam_idxs, focals, bones, centers

def load_animate(pose_h5, c2ws, focals, rest_pose, pose_keys,
                 selected_idxs, joints, n_step=10, refined=None,
                 undo_rot=False, center_cam=False,
                 from_rest=False, center_kps=False, idx_map=None,
                 rotate_y_ang=0.0):

    # TODO: figure out how to pick c2ws more elegantly?
    c2ws = c2ws[selected_idxs]

    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.
    if rotate_y_ang > 0:
        import math
        rot_mat = rotate_y(math.radians(rotate_y_ang))
        c2ws = rot_mat[None] @ c2ws 
    c2ws[..., 0, -1] *= 0.5
    c2ws = generate_bullet_time(c2ws, 10).transpose(1, 0, 2, 3).reshape(-1, 4, 4)

    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]
    
    # focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
    # cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]

    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root
    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]
    
    # forcefully rewrite the first pose to zero pose
    if from_rest:
        bones[:1, 1:] = 0

    if undo_rot:
        bones[..., 0, :] = np.array([0., 0., 1.5708*2], dtype=np.float32).reshape(1, 1, 3)

    interp_bones= []
    w = np.linspace(0, 1.0, n_step, endpoint=False).reshape(-1, 1, 1)
    for i in range(len(bones)-1):
        bone = bones[i:i+1, joints]
        next_bone = bones[i+1:i+2, joints]
        interp_bone = bone * (1 - w) + next_bone * w
        interp_bones.append(interp_bone)

    interp_bones.append(bones[-1:, joints])
    interp_bones = np.concatenate(interp_bones, axis=0)
    base_bones = bones[:1].repeat(len(interp_bones), 0).copy()
    base_bones[:, joints] = interp_bones
    base_bones[:, [12, 15]] = bones[-1, [12, 15], :]
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in base_bones])
    l2ws[..., :3, -1] += kps[:1, :1, :].copy() # only use the first pose as reference
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    #c2ws = c2ws[:1].repeat(len(kps), 0)
    c2ws = c2ws[:1].repeat(len(kps), 0)
    focals = focals[:1].repeat(len(kps), 0)
    cam_idxs = selected_idxs[:1].repeat(len(kps), 0)
    return kps, skts, c2ws, cam_idxs, focals, base_bones

def load_pose_rotate(pose_h5, c2ws, focals, rest_pose, pose_keys,
                     selected_idxs, refined=None, n_bullet=30,
                     undo_rot=False, center_cam=True, center_kps=False,
                     idx_map=None, args=None):

                    #  (pose_h5, c2ws, focals, rest_pose, pose_keys,
                    # selected_idxs, refined=None, n_bullet=30, bullet_ang=360,
                    # cam_factor=1.0,
                    # centers=None,
                    # undo_rot=False, center_cam=True, center_kps=True,
                    # idx_map=None, args=None, real=True):

    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]

    rots = torch.zeros(len(bones), 4, 4)
    rots_ = axisang_to_rot(torch.tensor(bones[..., :1, :]).reshape(-1, 3))
    rots[..., :3, :3] = rots_
    rots[..., -1, -1] = 1.
    rots = rots.cpu().numpy()

    # import ipdb; ipdb.set_trace()
    # rots_y = torch.tensor(generate_bullet_time(rots[0], n_views=n_bullet//3, axis='y'))
    # rots_x = torch.tensor(generate_bullet_time(rots[0], n_views=n_bullet//3, axis='x'))
    # rots_z = torch.tensor(generate_bullet_time(rots[0], n_views=n_bullet//3, axis='z'))
    # rots = torch.cat([rots_y, rots_x, rots_z], dim=0)
    
    rots_bullets = torch.tensor(generate_bullet_time(rots, n_views=n_bullet, axis='y')).permute(1, 0, 2, 3).reshape(-1, 4, 4)
    

    root_rotated = rot_to_axisang(rots_bullets[..., :3, :3]).cpu().numpy()
    bones = bones.repeat(len(root_rotated), 0)
    bones[..., 0, :] = root_rotated
    c2ws = c2ws[selected_idxs].repeat(len(root_rotated), 0)
    focals = focals[selected_idxs].repeat(len(root_rotated), 0)

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    cam_idxs = selected_idxs.repeat(len(root_rotated), 0)

    return kps, skts, bones, c2ws, cam_idxs, focals

def load_interpolate(pose_h5, c2ws, focals, rest_pose, pose_keys,
                     selected_idxs, n_step=10, refined=None,
                     undo_rot=False, center_cam=False,
                     center_kps=False, idx_map=None):

    # TODO: figure out how to pick c2ws more elegantly?
    c2ws = c2ws[selected_idxs]
    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]
    #focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
    #cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]

    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root
    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    interp_bones= []
    w = np.linspace(0, 1.0, n_step, endpoint=False).reshape(-1, 1, 1)
    for i in range(len(bones)-1):
        bone = bones[i:i+1]
        next_bone = bones[i+1:i+2]
        interp_bone = bone * (1 - w) + next_bone * w
        interp_bones.append(interp_bone)
    interp_bones.append(bones[-1:])
    interp_bones = np.concatenate(interp_bones, axis=0)
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in interp_bones])
    l2ws[..., :3, -1] += kps[:1, :1, :].copy() # only use the first pose as reference
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    c2ws = c2ws[:1].repeat(len(kps), 0)
    focals = focals[:1].repeat(len(kps), 0)
    cam_idxs = selected_idxs[:1].repeat(len(kps), 0)
    return kps, skts, c2ws, cam_idxs, focals



def eval_bullettime_kps(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30, 
                    #centers=None,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None):

    undo_rot=False; center_cam=True; center_kps=True

    # prepare camera
    c2ws = c2ws[selected_idxs]
    # centers = centers[selected_idxs]

    '''Note: DANBO does not refine pose'''


def generate_psnr_imgs(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30, bullet_ang=360,
                    cam_factor=1.0,
                    centers=None,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None, args=None):

    """we need the global root position for PSNR images, so no root-centering
    except you center the camera using the root location as well"""

    undo_rot=False; center_cam=False; center_kps=True
    # center_kps = True; center_cam= False

    # prepare camera
    # TODO: temporary hack
    try:
        c2ws = c2ws[selected_idxs]

        if isinstance(focals, float):
            focals = np.array([focals] * len(selected_idxs))
        else:
            focals = focals[selected_idxs]

        if isinstance(centers, float):
            centers = np.array([centers] * len(selected_idxs))
        else:
            if len(centers.shape) > 2:
                centers = centers[0][selected_idxs]
            else:
                centers = centers[selected_idxs]

            # centers = centers[selected_idxs]

    except:
        ipdb.set_trace()

        import h5py
        h5_file = h5py.File(pose_h5, 'r')
        cam_idxs = h5_file['img_pose_indices'][selected_idxs]
        c2ws = c2ws[cam_idxs]
        focals = focals[cam_idxs]
        h5_file.close()

        
    if center_cam:
        # get and update
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.
     #c2ws[..., 0, -1] *= cam_factor


    # prepare pose
    # TODO: hard-coded for now so we can quickly view the outcomes!
    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        
        print("are you using the right selected_idxs?")
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        if args.switch_cam:
            print("are you using the right data path for pose_h5?")
            time.sleep(3)
            
    else:
        print("is there refined pose?")
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    if center_kps:
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root
        c2ws[..., :3, -1] -= root.reshape(-1,3)

    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    # c2ws[..., 0, -1] *= cam_factor
    c2ws = generate_bullet_time(c2ws, n_views=n_bullet, bullet_ang=bullet_ang).transpose(1, 0, 2, 3).reshape(-1, 4, 4)
    
    if len(focals.shape) == 2 and focals.shape[1] == 2:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)

    # if centers is not None:
    #     #centers = centers[selected_idxs] if len(centers) > 1 else centers
        
    #     centers = centers[selected_idxs] if len(centers) > 1 else centers
    #     centers = centers[:, None, :].repeat(n_bullet, 1).reshape(-1, 2)
    
    if len(centers.shape) == 2 and centers.shape[1] == 2:
        centers = centers[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        centers = centers[:, None].repeat(n_bullet, 1).reshape(-1)


    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    # expand shape for repeat
    kps = kps[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 3)
    skts = skts[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 4, 4)
    bones = bones[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 3)
    
    return kps, skts, c2ws, cam_idxs, focals, bones




def load_bullettime(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30, bullet_ang=360,
                    cam_factor=1.0,
                    centers=None,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None, args=None, real=True):

    '''center kps or center cam'''
    undo_rot=False; center_cam=False; center_kps=True
    # prepare camera
    # TODO: temporary hack

    try:
        # if real:
        c2ws = c2ws[selected_idxs]

        if isinstance(focals, float):
            focals = np.array([focals] * len(selected_idxs))
        else:
            focals = focals[selected_idxs]

        if isinstance(centers, float):
            centers = np.array([centers] * len(selected_idxs))
        else:
            centers = centers[selected_idxs]

    except:
        ipdb.set_trace()

        import h5py
        h5_file = h5py.File(pose_h5, 'r')
        cam_idxs = h5_file['img_pose_indices'][selected_idxs]
        c2ws = c2ws[cam_idxs]
        focals = focals[cam_idxs]
        h5_file.close()

        
    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.
    
    # prepare pose
    # TODO: hard-coded for now so we can quickly view the outcomes!
    # if refined is None:
    if not args.render_refined:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        
        print("are you using the right selected_idxs?")
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        if args.switch_cam:
            print("are you using the right data path for pose_h5?")
            time.sleep(3)
            
    else:
        
        print("is there refined pose?")
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_bullet, 1).reshape(-1)

    if center_kps:
        
        root = kps[..., :1, :].copy() # assume to be CMUSkeleton
        kps[..., :, :] -= root
        try:
            c2ws[..., :3, -1] -= root.reshape(-1,3)
        except:
            ipdb.set_trace()

    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    #c2ws[..., 0, -1] *= cam_factor
    c2ws = generate_bullet_time(c2ws, n_views=n_bullet, bullet_ang=bullet_ang).transpose(1, 0, 2, 3).reshape(-1, 4, 4)

    if len(focals.shape) == 2 and focals.shape[1] == 2:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        focals = focals[:, None].repeat(n_bullet, 1).reshape(-1)
        

    # if centers is not None:
    #     #centers = centers[selected_idxs] if len(centers) > 1 else centers
        
    #     centers = centers[selected_idxs] if len(centers) > 1 else centers
    #     centers = centers[:, None, :].repeat(n_bullet, 1).reshape(-1, 2)

    if len(centers.shape) == 2 and centers.shape[1] == 2:
        centers = centers[:, None].repeat(n_bullet, 1).reshape(-1, 2)
    else:
        centers = centers[:, None].repeat(n_bullet, 1).reshape(-1)


    if undo_rot:
        bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    # expand shape for repeat
    kps = kps[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 3)
    skts = skts[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 4, 4)
    bones = bones[:, None].repeat(n_bullet, 1).reshape(len(selected_idxs) * n_bullet, -1, 3)
    
    return kps, skts, c2ws, cam_idxs, focals, bones, centers, root


def load_selected(pose_h5, c2ws, focals, rest_pose, pose_keys,
                  selected_idxs, centers=None, refined=None, idx_map=None):

    # get cameras
    c2ws = c2ws[selected_idxs]
    if isinstance(focals, float):
        focals = np.array([focals] * len(selected_idxs))
    else:
        focals = focals[selected_idxs]
    
    if centers is not None:
        centers = centers[selected_idxs]

    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs//3, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs

    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    return kps, skts, c2ws, cam_idxs, focals, bones, centers

def load_bubble(pose_h5, c2ws, focals, rest_pose, pose_keys,
                selected_idxs, x_deg=15., y_deg=25., z_t=0.1,
                bubble_ang=30,
                refined=None, centers=None, n_step=5, center_kps=True, idx_map=None, args=None):
   
    center_cam = False; center_kps=True

    x_rad = x_deg * np.pi / 180.
    y_rad = y_deg * np.pi / 180.

    # center camera
    # c2ws = c2ws[selected_idxs]
   
    try:
        c2ws = c2ws[selected_idxs]

        if isinstance(focals, float):
            focals = np.array([focals] * len(selected_idxs))
        else:
            focals = focals[selected_idxs]

        if isinstance(centers, float):
            centers = np.array([centers] * len(selected_idxs))
        else:
            if len(centers.shape) > 2:
                centers = centers[0][selected_idxs]
            else:
                centers = centers[selected_idxs]

    except:
        # import h5py
        ipdb.set_trace()

    if center_cam:
        shift_x = c2ws[..., 0, -1].copy()
        shift_y = c2ws[..., 1, -1].copy()
        c2ws[..., :2, -1] = 0.

    # load poses
    if refined is None:
        kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    else:
        selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
        kps, bones = refined
        kps = kps[selected_idxs]
        bones = bones[selected_idxs]
    cam_idxs = selected_idxs[:, None].repeat(n_step, 1).reshape(-1)

    # center kps
    if center_kps:
        root = kps[..., :1, :].copy()
        kps[..., :, :] -= root
        # added
        c2ws[..., :3, -1] -= root.reshape(-1,3)

    elif center_cam:
        kps[..., :, 0] -= shift_x[:, None]
        kps[..., :, 1] -= shift_y[:, None]

    z_t = z_t * c2ws[0, 2, -1]
    

    # begin bubble generation
    
    motions = np.linspace(0., math.radians(bubble_ang), n_step, endpoint=True)
    x_motions = (np.cos(motions) - 1.) * x_rad
    y_motions = np.sin(motions) * y_rad
    z_trans = (np.sin(motions) + 1.) * z_t
    cam_motions = []
    for x_motion, y_motion in zip(x_motions, y_motions):
        cam_motion = rotate_x(x_motion) @ rotate_y(y_motion)
        cam_motions.append(cam_motion)

    bubble_c2ws = []
    for c2w in c2ws:
        bubbles = []
        for cam_motion, z_tran in zip(cam_motions, z_trans):
            c = c2w.copy()
            c[2, -1] += z_tran
            bubbles.append(cam_motion @ c)
        bubble_c2ws.append(bubbles)

    
    # if isinstance(focals, float):
    #     focals = np.array([focals] * len(selected_idxs))
    # else:
    #     focals = focals[selected_idxs]

    if len(focals.shape) == 2 and focals.shape[-1] == 2:
        focals = focals[:, None, :].repeat(n_step, 1).reshape(-1, 2)
    else:
        focals = focals[:, None].repeat(n_step, 1).reshape(-1)

    if centers is not None:
        # centers = centers[selected_idxs]
        centers = centers[:, None, :].repeat(n_step, 1).reshape(-1, 2)


    # # load poses
    # if refined is None:
    #     #TODO: rewrite the whole renderer
    #     kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs//3, ...])
    #     selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    # else:
    #     selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    #     kps, bones = refined
    #     kps = kps[selected_idxs]
    #     bones = bones[selected_idxs]
    # cam_idxs = selected_idxs[:, None].repeat(n_step, 1).reshape(-1)

    # undo rot
    #bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)


    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    # expand shape for repeat
    kps = kps[:, None].repeat(n_step, 1).reshape(len(selected_idxs) * n_step, -1, 3)
    skts = skts[:, None].repeat(n_step, 1).reshape(len(selected_idxs) * n_step, -1, 4, 4)

    c2ws = np.array(bubble_c2ws).reshape(-1, 4, 4)
    return kps, skts, c2ws, cam_idxs, focals, bones, centers


# TODO: temporary
def load_inspect_vol(pose_h5, c2ws, focals, rest_pose, pose_keys,
                    selected_idxs, refined=None, n_bullet=30,
                    undo_rot=False, center_cam=True, center_kps=True,
                    idx_map=None):

    kps, skts, c2ws, cam_idxs, focals, bones = load_bullettime(
        pose_h5, c2ws, focals, rest_pose, pose_keys,
        selected_idxs, refined=refined, n_bullet=n_bullet,
        undo_rot=undo_rot, center_cam=center_cam, center_kps=center_kps,
        idx_map=idx_map)

    return kps, skts, c2ws, cam_idxs, focals, bones

def read_camera_data(data_h5):
    h5_file = h5py.File(data_h5, 'r')
    c2ws, focals, centers = None, None, None

    c2ws = h5_file['c2ws'][:]
    focals = h5_file['focals'][:]
    if 'centers' in h5_file:
        centers = h5_file['centers'][:]
    _, H, W, _ = h5_file['img_shape'][:]

    if 'img_pose_indices' in h5_file:
        img_pose_indices = h5_file['img_pose_indices'][:]
        c2ws = c2ws[img_pose_indices]
        focals = focals[img_pose_indices]
        if centers is not None:
            centers = centers[img_pose_indices]

    h5_file.close()
    return  c2ws, centers, focals, H, W

def read_pose_data(
        data_h5, 
        poseopt_layer=None, 
        refined=None):
    '''
    Read poses from either
    (1) data_h5
    (2) poseopt_layer: optimized poses from ckpt
    (3) refined: a specific ckpt
    '''

    h5_file = h5py.File(data_h5, 'r')

    if refined is not None:
        # TODO: fill this in
        raise NotImplementedError
    elif poseopt_layer is not None:
        raise NotImplementedError
    else:
        kp3d = h5_file['kp3d'][:]
        bones = h5_file['bones'][:]
        
        if 'kp_idxs' in h5_file.keys():
            kp_idxs = h5_file['kp_idxs'][:]
        else:
            kp_idxs = np.arange(len(kp3d))
        rest_pose = h5_file['rest_pose'][:]

    kp3d = kp3d[kp_idxs]
    bones = bones[kp_idxs]

    return kp3d, bones, rest_pose

def find_cam_idxs_via_kp_project(h5_path, kps, c2ws, H, W, 
                                 focals, centers, val_start=327,
                                 ):

    # TODO: currently only works for PerfCap
    print(f'val_start from that last {val_start}')
    kp2d = skeleton3d_to_2d(kps, c2ws, H, W, focals, centers)

    h5_file = h5py.File(h5_path, 'r')
    training_kps = h5_file['kp3d'][-val_start:]
    training_c2ws = h5_file['c2ws'][-val_start:]
    training_focals = h5_file['focals'][-val_start:]
    training_centers = None
    if 'centers' in h5_file:
        training_centers = h5_file['centers'][-val_start:]
    training_kp2d = skeleton3d_to_2d(training_kps, training_c2ws, H, W,
                                      training_focals, training_centers)
    h5_file.close()

    # output shape = (n_eval, n_train, n_joints)
    diff = ((kp2d[:, None]/1000. - training_kp2d[None, :]/1000.)**2).mean(-1)
    cam_idxs = diff.sum(-1).argmin(-1)
    return cam_idxs

def to_tensors(data_dict):
    tensor_dict = {}
    for k in data_dict:
        if isinstance(data_dict[k], np.ndarray):
            if k == 'bg_indices' or k == 'subject_idxs':
                tensor_dict[k] = torch.tensor(data_dict[k]).long()
            else:
                tensor_dict[k] = torch.tensor(data_dict[k]).float()
        elif k == 'hwf' or k == 'cams' or k == 'bones' or k =='centers':
            tensor_dict[k] = data_dict[k]
        elif data_dict[k] is None:
            tensor_dict[k] = None
        else:
            raise NotImplementedError(f"{k}: only nparray and hwf are handled now!")
    return tensor_dict

def evaluate_metric(rgbs, accs, bboxes, gt_dict, basedir, time=""):
    '''
    Always evaluate in the box
    '''
    gt_paths = gt_dict['gt_paths']
    mask_paths = gt_dict['gt_mask_paths']
    is_gt_paths = gt_dict['is_gt_paths']

    bg_imgs = gt_dict['bg_imgs']
    bg_indices = gt_dict['bg_indices']

    psnrs, ssims = [], []
    fg_psnrs, fg_ssims = [], []
    ssim_eval = SSIM(size_average=False)

    for i, (rgb, acc, bbox, gt_path) in enumerate(zip(rgbs, accs, bboxes, gt_paths)):

        if (i + 1) % 150 == 0:
            np.save(os.path.join(basedir, time, 'scores.npy'),
                    {'psnr': psnrs, 'ssim': ssims, 'fg_psnr': fg_psnrs, 'fg_ssim': fg_ssims},
                     allow_pickle=True)

        tl, br = bbox
        if is_gt_paths:
            gt_img = imageio.imread(gt_path) / 255.
            gt_mask = None
            if mask_paths is not None:
                gt_mask = imageio.imread(mask_paths[i])
                gt_mask[gt_mask < 128] = 0
                gt_mask[gt_mask >= 128] = 1
                mask_cropped = gt_mask[tl[1]:br[1], tl[0]:br[0]].astype(np.float32)
        else:
            # TODO: hard-coded, fix this.
            gt_img = gt_path.reshape(*rgb.shape) / 255.
            gt_mask = mask_paths[i].reshape(*rgb.shape[:2], 1) if mask_paths is not None else None

        # h36m special case
        if gt_img.shape[0] == 1002:
            gt_img = gt_img[1:-1]

        if gt_mask is not None:
            if gt_mask.shape[0] == 1002:
                gt_mask = gt_mask[1:-1]
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask[..., :1]
            elif len(gt_mask.shape) == 2:
                gt_mask = gt_mask[:, :, None]
            if bg_imgs is not None:
                bg_img = bg_imgs[bg_indices[i]]
                gt_img = gt_img * gt_mask + (1.-gt_mask) * bg_img

            mask_cropped = gt_mask[tl[1]:br[1], tl[0]:br[0]].astype(np.float32)
            if mask_cropped.sum() < 1:
                print(f"frame {i} is empty, skip")
                continue

        gt_cropped = gt_img[tl[1]:br[1], tl[0]:br[0]].astype(np.float32)
        rgb_cropped = rgb[tl[1]:br[1], tl[0]:br[0]]

        se = np.square(gt_cropped - rgb_cropped)
        box_psnr = -10. * np.log10(se.mean())

        rgb_tensor = torch.tensor(rgb_cropped[None]).permute(0, 3, 1, 2)
        gt_tensor = torch.tensor(gt_cropped[None]).permute(0, 3, 1, 2)
        box_ssim = ssim_eval(rgb_tensor, gt_tensor).permute(0, 2, 3, 1).cpu().numpy()

        ssims.append(box_ssim.mean())
        psnrs.append(box_psnr)

        if gt_mask is not None:
            denom = (mask_cropped.sum() * 3.)
            masked_mse = (se * mask_cropped).sum() / denom
            fg_psnr = -10. * np.log10(masked_mse)
            fg_ssim = (box_ssim[0] * mask_cropped).sum() / denom

            fg_ssims.append(fg_ssim.mean())
            fg_psnrs.append(fg_psnr)

    score_dict = {'psnr': psnrs, 'ssim': ssims, 'fg_psnr': fg_psnrs, 'fg_ssim': fg_ssims}

    np.save(os.path.join(basedir, time, 'scores.npy'), score_dict, allow_pickle=True)

    with open(os.path.join(basedir, time, 'score_final.txt'), 'w') as f:
        for k in score_dict:
            avg = np.mean(score_dict[k])
            f.write(f'{k}: {avg}\n')

@torch.no_grad()
def render_mesh(basedir, render_kwargs, tensor_data, chunk=4096, radius=1.80,
                res=255, threshold=10., time="", args=None):
    import mcubes, trimesh
    ray_caster = render_kwargs['ray_caster']

    rel_indices = args.global_idxs
    # rel_indices = args.selected_idxs    

    os.makedirs(os.path.join(basedir, time, 'meshes'), exist_ok=True)
    kps, skts, bones = tensor_data['kp'], tensor_data['skts'], tensor_data['bones']
    v_t_tuples = []
    for i in trange(len(kps)):
        raw_d = ray_caster(kps=kps[i:i+1], skts=skts[i:i+1], bones=bones[i:i+1],
                           radius=radius, render_kwargs=render_kwargs['preproc_kwargs'],
                           res=res, netchunk=chunk, fwd_type='mesh')
        sigma = np.maximum(raw_d.cpu().numpy(), 0)
        vertices, triangles = mcubes.marching_cubes(sigma, threshold)
        mesh = trimesh.Trimesh(vertices / res - .5, triangles)

        i = rel_indices[i]
        mesh.export(os.path.join(basedir, time, 'meshes', f'{i:03d}.ply'))

def run_render():
    args = config_parser().parse_args()

    if args.render_folder == '':
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # ("%Y-%m-%d-%H-%M-%S")
        print(f"new render folder: {time}")
    else:
        time = args.render_folder
        print(f"existing render folder: {time}")
    
    view = args.data_path.split("_cam_")[1][0]
    comb = args.data_path.split("_cam_")[0].split("/")[-1]
    print(f"camera: {view} comb: {comb}")
    
    # update data_path
    if args.dataset== 'mirror':
        GT_PREFIXES[args.dataset] = args.data_path
    # parse nerf model args
    nerf_args = txt_to_argstring(args.nerf_args)

    # calls the configurations parameters in run_nerf and re-uses them here
    nerf_args, unknown_args = nerf_config_parser().parse_known_args(nerf_args)
    print(f'UNKNOWN ARGS: {unknown_args}')
    
    # update with parameters from run_render directly
    nerf_args.train_size =  args.train_size
    nerf_args.data_size = args.data_size
    nerf_args.randomize_view_dir = args.randomize_view_dir
    # nerf_args.ray_tr_type = args.ray_tr_type
    nerf_args.idx_map_in_ckpt = args.idx_map_in_ckpt
    nerf_args.data_path =  args.data_path

    # load nerf model
    render_kwargs, poseopt_layer, rest_pose = load_nerf(args, nerf_args)

    # prepare the required data
    render_data, gt_dict, selected_idxs = load_render_data(args, nerf_args, poseopt_layer, rest_pose, nerf_args.opt_framecode)
    tensor_data = to_tensors(render_data)

    basedir = os.path.join(args.outputdir, args.runname)
    os.makedirs(os.path.join(basedir, time), exist_ok=True)

    if args.render_mesh:
        render_mesh(basedir, render_kwargs, tensor_data, res=args.mesh_res, radius=args.mesh_radius, time=time, args=args)
        return

    render_kwargs['render_confd'] = args.render_confd
    render_kwargs['render_entropy'] = args.render_entropy

    print(f"render_path called in run_render.py")
    
    def get_crop_HW(tensor_data):
        kps = tensor_data['kp'].cpu().numpy()
        hwf = tensor_data['hwf']
        center = tensor_data['centers'].cpu().numpy()
        c2ws_expanded = tensor_data['render_poses'].cpu().numpy() #[None, ...]
        
        # create crop directories
        skel_dir = os.path.join(basedir, args.crop_folder, f'skel_{view}')
        gt_dir = os.path.join(basedir, args.crop_folder, f'gt')
        os.makedirs(skel_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        crop_HWs = []
        cropped_imgs = []

        for i, rel_idx in enumerate(args.global_idxs):
            print(f"rel_idx {rel_idx}")
            
            subj_H, subj_W = 700, 500
            # Fr_H, Fr_W = xx, xx
            # Ti_H, Ti_W = xx, xx

            # exist_renders
            img_url  = os.path.join(basedir, args.render_folder, f'skel_{view}', f'{rel_idx:05d}_glob.png')
            img = imageio.imread(img_url)

            kp2d = skeleton3d_to_2d(kps[i:i+1], c2ws_expanded[i:i+1], int(hwf[0]), int(hwf[1]), hwf[2][i:i+1], center[i:i+1])

            # save cropped image
            # mid_x = kp2d[0,:,0].min() + ((kp2d[0,:,0].max() - kp2d[0,:,0].min())/2)
            # mid_y = kp2d[0,:,1].min() + ((kp2d[0,:,1].max() - kp2d[0,:,1].min())/2)

            # use the hip as tracking point for cropping
            mid_x = kp2d[0,0,0]
            mid_y = kp2d[0,0,1]

            crop_minH = int(mid_y - subj_H//2)  
            crop_minW = int(mid_x - subj_W//2)

            crop_maxH = int(mid_y + subj_H//2)
            crop_maxW = int(mid_x + subj_W//2)

            # handle boundary cases
            if crop_minH < 0:
                crop_minH = 0
                crop_maxH = subj_H

            if crop_maxH > int(hwf[0]):
                crop_maxH = int(hwf[0])
                crop_minH = int(hwf[0]) - subj_H

            if crop_minW < 0:
                crop_minW = 0
                crop_maxW = subj_W
            
            if crop_maxW > int(hwf[1]):
                crop_maxW = int(hwf[1])
                crop_minW = int(hwf[1]) - subj_W

            assert (crop_maxH-crop_minH) == subj_H, "crop height needs to be consistent"
            assert (crop_maxW-crop_minW) == subj_W, "crop width needs to be consistent"

            cropped_img = img[crop_minH:crop_maxH, crop_minW:crop_maxW]

            crop_HWs.append((crop_minH, crop_maxH, crop_minW, crop_maxW))
            cropped_imgs.append(cropped_img)

        return crop_HWs, cropped_imgs
    

    if args.save_crops:
        # crop existing renderings

            crop_HWs, cropped_imgs = get_crop_HW(tensor_data)

            for i, rel_idx in enumerate(args.global_idxs):
                # print(f"rel_idx {rel_idx}")
                cropped_img = cropped_imgs[i]
            
                cropped_img_url  = os.path.join(basedir, args.crop_folder, f'skel_{view}', f'{rel_idx:05d}_cropped.png')
                imageio.imwrite(cropped_img_url, cropped_img)

    if not args.save_crops:
        # do forward pass only in normal-full renderings
        rgbs, _, accs, _, bboxes = render_path(render_kwargs=render_kwargs,
                                        chunk=nerf_args.chunk,
                                        ext_scale=nerf_args.ext_scale,
                                        ret_acc=True,
                                        white_bkgd=args.white_bkgd,
                                        top_expand_ratio=args.top_expand_ratio,
                                        extend_mm=args.extend_mm,
                                        randomize_view_dir=args.randomize_view_dir,
                                        **tensor_data,
                                        args=args)

    if gt_dict['gt_paths'] is not None:
        H, W, focal = render_data['hwf']

        if args.eval:
            # print("are you using full or raw rgb?")
            evaluate_metric(rgbs, accs, bboxes, gt_dict, basedir, time)
            pass
        
        elif args.save_gt:
            
            gt_paths = gt_dict['gt_paths']
            os.makedirs(os.path.join(basedir, time, 'gt'), exist_ok=True)
            ipdb.set_trace()

            for i in trange(len(gt_paths)):
                rel_idx = args.global_idxs[i]

                if gt_dict['is_gt_paths']:
                    shutil.copyfile(gt_paths[i], os.path.join(basedir, time, 'gt',  f'{i:05d}.png'))
                else:
                    # TODO: temporary fix
                    if args.save_crops:
                        gt_img = gt_paths[i].reshape(H, W, 3)

                        (crop_minH, crop_maxH, crop_minW, crop_maxW) = crop_HWs[i]
                        crop_gt_img = gt_img[crop_minH:crop_maxH, crop_minW:crop_maxW]
                        imageio.imwrite(os.path.join(basedir, args.crop_folder, 'gt',  f'{rel_idx:05d}.png'), crop_gt_img)
                        

                    else:   
                        imageio.imwrite(os.path.join(basedir, time, 'gt',  f'{i:05d}.png'), gt_paths[rel_idx].reshape(H, W, 3))


    if args.no_save:
        return

    # take to [0, 255] for skeleton plot
    rgbs = (rgbs * 255).astype(np.uint8)
    accs = (accs * 255).astype(np.uint8)

    """
    if args.switch_cam:
        # duplicate kp and focal twice
        render_data['kp'] = np.tile(render_data['kp'],[2,1,1])
        render_data['hwf'] = (render_data['hwf'][0], 
                              render_data['hwf'][1],
                              np.tile(render_data['hwf'][2],[2,1]))

    # overlay on body reconstruction
    skeletons = draw_skeletons_3d(rgbs, render_data['kp'],
                                  render_data['render_poses'],
                                  *render_data['hwf'])

    plt.savefig(f"checkers/imgs/rgb_c.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)
    """

    os.makedirs(os.path.join(basedir, time, f'image_{view}'), exist_ok=True)
    os.makedirs(os.path.join(basedir, time, f'skel_{view}'), exist_ok=True)
    os.makedirs(os.path.join(basedir, time, f'acc_{view}'), exist_ok=True)

    chk_img_url = f"{project_dir}/checkers/imgs"
    '''post-rendering real-virt blending'''
    
    def blend(real_imgs, acc_map_real, virt_imgs, acc_map_virt):
        # ref https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
        # h, w = render_data['hwf'][0], render_data['hwf'][1]
        n_size, h, w, _ = real_imgs.shape

        try:
            bkgd_img = render_data['bg_imgs'].reshape(1, h, w, 3)
        except:

            # possible retargeting on diff background
            if render_data['bg_imgs'] != None:
                _, h,w,_  = render_data['bg_imgs'].shape
                bkgd_img = render_data['bg_imgs'].reshape(1, h, w, 3)

            else:
                bkgd_img = np.ones_like(real_imgs).reshape(1, h, w, 3).astype(np.float32)

        bkgd_img = np.tile(bkgd_img, [n_size,1,1,1])

        '''overlay rendering: order - virtual first, then real'''
        # acc_map_virt allows virt_img where virt fog exists
        I_image =  virt_imgs * acc_map_virt 
        # (1-acc_map_real) wont allow I_image (bkgd or virt fog) where real fog exists
        I_image =  I_image * (1-acc_map_real) + real_imgs * acc_map_real 

        # order - virtual first, then real
        I_bgkd = bkgd_img * (1-acc_map_virt) 
        I_bgkd = I_bgkd * (1-acc_map_real) 
        I_render = I_bgkd + I_image

        return I_render


    half_size = rgbs.shape[0]
    if args.switch_cam:
        half_size = rgbs.shape[0]//2

    if args.switch_cam:
        if args.outliers:
            r_skels_2d = skeleton3d_to_2d(render_data['kp'], render_data['render_poses'][:half_size], *render_data['hwf'])
            v_skels_2d = skeleton3d_to_2d(render_data['kp'], render_data['render_poses'][half_size:], *render_data['hwf'])
            
            temp_rgbs, temp_accs = np.ones_like(rgbs), np.ones_like(accs), 
            start = timenow.time()
            for h_i in range(half_size):
                
                r_rgb, r_acc, _, r_out_stats = cca_image(r_skels_2d[h_i], img=rgbs[:half_size][h_i], acc_img=accs[:half_size][h_i], plot=False, chk_folder=chk_img_url, white_bkgd=False)
                v_rgb, v_acc, _, v_out_stats = cca_image(v_skels_2d[h_i], img=rgbs[half_size:][h_i], acc_img=accs[half_size:][h_i], plot=False, chk_folder=chk_img_url, white_bkgd=False)
                if h_i==0: print("cca took: %.2f seconds" %(timenow.time() - start))

                # store items
                temp_rgbs[h_i] = r_rgb
                temp_rgbs[h_i + half_size] = v_rgb

                temp_accs[h_i] = r_acc
                temp_accs[h_i + half_size] = v_acc
 
            rgbs = temp_rgbs.copy()
            accs = temp_accs.copy()

        # perform blending (operation) in 0,1 space
        rgbs = rgbs/255.0
        accs = accs/255.0

        rgbs = blend(rgbs[:half_size], accs[:half_size], rgbs[half_size:], accs[half_size:])   
        accs = np.clip(accs[:half_size] +  accs[half_size:], a_min=0, a_max=1)
        
    if args.switch_cam:
        # overlay on body reconstruction
        skel_stage1,_ = draw_skeletons_3d(rgbs, render_data['kp'],
                                    render_data['render_poses'][:half_size],
                                    *render_data['hwf'])

        skeletons,_ = draw_skeletons_3d(skel_stage1, render_data['kp'],
                                    render_data['render_poses'][half_size:],
                                    *render_data['hwf'])
    
    else:
        skeletons, skels_2d = draw_skeletons_3d(rgbs, render_data['kp'],
                                    render_data['render_poses'],
                                    *render_data['hwf'])

    selected_idxs = args.global_idxs
    rel_idx = 0
    s_idx = 0
    b_idx = f"_{s_idx:04d}"

    for i, (rgb, acc, skel) in tqdm(enumerate(zip(rgbs, accs, skeletons)), total=len(skeletons)):
        if args.n_bullet == 1 or args.n_bubble==1:
            rel_idx = selected_idxs[i]
            b_idx = f"_{s_idx:04d}"
        
        else:
            if args.render_type=="bullet":
                rel_idx=selected_idxs[i//args.n_bullet]
            if args.render_type=="bubble":
                rel_idx=selected_idxs[i//args.n_bubble]

            if args.render_type=="bullet":
                if i % args.n_bullet==0:
                    s_idx = 0 # reset 
                    
            elif args.render_type=="bubble":
                if i % args.n_bubble==0:
                    s_idx = 0 # reset 
            
            b_idx = f"_{s_idx:04d}"
            print(f"i {i} s_idx {s_idx} rel_idx {rel_idx}")     
            s_idx+=1 


        start = timenow.time()
        if args.outliers and not args.switch_cam:
            rgb, acc, skel, out_stats = cca_image(skels_2d[i], img=rgb, acc_img=acc, plot=False, chk_folder=chk_img_url)
            out_whole_kps_bool, out_count_per_whole_kps, out_count_H, out_count_W = out_stats
            if out_whole_kps_bool:
                print(f"i: {i} rel_idx: {rel_idx} out_count_per_whole_kps {out_count_per_whole_kps}, out_count_H {out_count_H}, out_count_W {out_count_W}")

            if i==0: print("cca took: %.2f seconds" %(timenow.time() - start))

        imageio.imwrite(os.path.join(basedir, time, f'image_{view}', f'{rel_idx:05d}{b_idx}.png'), rgb)
        imageio.imwrite(os.path.join(basedir, time, f'acc_{view}', f'{rel_idx:05d}{b_idx}.png'), acc)
        imageio.imwrite(os.path.join(basedir, time, f'skel_{view}', f'{rel_idx:05d}{b_idx}.png'), skel)
    np.save(os.path.join(basedir, time, 'bboxes.npy'), bboxes, allow_pickle=True)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run_render()
