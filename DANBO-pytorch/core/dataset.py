import pdb
import bisect
import h5py, math
import torch
import imageio
import random
import platform
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate

from .pose_opt import pose_ckpt_to_pose_data
from .utils.skeleton_utils import SMPLSkeleton, get_per_joint_coords, cylinder_to_box_2d, \
                                  nerf_c2w_to_extrinsic, calculate_bone_length
from .utils.ray_utils import get_rays_np, kp_to_valid_rays

from .utils.skeleton_utils import SMPLSkeleton, get_per_joint_coords, cylinder_to_box_2d, nerf_c2w_to_extrinsic,\
    skeleton3d_to_2d, plot_skeleton2d, draw_skeleton2d, plot_bbox2D, get_iou

dataset_catalog = {
    'h36m': {},
    'perfcap': {},
    'mixamo': {},
    'surreal': {},
}

checker_folder = "checkers"

class BaseH5Dataset(Dataset):
    # TODO: poor naming
    def __init__(self, h5_path, h5_path_v, N_samples=96, patch_size=1, split='full',
                 N_nms=0, N_nms_v=0, subject=None, mask_img=False, multiview=False, overlap_rays=False,
                 train_size=None, data_size=None, idx_map = None, refined_paths=None, idx_map_in_ckpt=None,
                 args=None,
                 perturb_bg=False):
        '''
        Base class for multi-proc h5 dataset

        args
        ----
        h5_path (str): path to .h5 file
        N_samples (int): number of pixels to sample from each image
        patch_size (int): sample patches of rays of this size.
        split (str): split to use. splits are defined in a dataset-specific manner
        N_nms (float): number of pixel samples to sample from out-of-mask regions (in a bounding box).
        subject (str): name of the dataset subject
        mask_img (bool): replace background parts with estimated background pixels
        multiview (bool): to enable multiview optimization
        perturb_bg (bool): perturb background color during training
        '''
        self.h5_path = h5_path
        self.h5_path_v = h5_path_v
        self.split = split
        self.dataset = None
        self.dataset_v = None
        self.subject = subject
        self.mask_img = mask_img
        self.multiview = multiview
        self.perturb_bg = perturb_bg

        self.N_samples = N_samples
        self.patch_size = patch_size
        self.N_nms = int(math.floor(N_nms)) if N_nms >= 1.0 else float(N_nms)
        self.N_nms_v  = int(math.floor(N_nms_v )) if N_nms_v  >= 1.0 else float(N_nms_v)
        self._idx_map = idx_map 
        self._render_idx_map = None
        self.refined_paths = refined_paths 
        self.idx_map_in_ckpt = idx_map_in_ckpt
        self.args = args

        # self.import_idx_map = None
        self.init_meta()

        if self.import_idx_map is not None:
            self._idx_map = self.import_idx_map.copy()

        self.init_len()
        self.box2d = None
        self.box2d_v = None

        self.box2d_overlap = None
        self.box2d_v_overlap = None
        self.overlap_rays = overlap_rays

        if self.N_nms > 0.0:
            self.init_box2d(scale=1.0)
        if self.N_nms_v > 0.0:
            self.init_box2d_v(scale=1.0)

        if self.overlap_rays:
            self.init_box2d(scale=1.0, box2d_overlap=True)
            self.init_box2d_v(scale=1.0, box2d_overlap=True)

    def __getitem__(self, q_idx):
        '''
        q_idx: index queried by sampler, should be in range [0, len(dataset)].
        Note - self._idx_map maps q_idx to indices of the sub-dataset that we want to use.
               therefore, self._idx_map[q_idx] may only lie within [0, len(subdataset)]
        '''
        import pdb
        if self._idx_map is not None:
            idx = self._idx_map[q_idx]
        else:
            idx = q_idx

        if self.dataset is None and self.dataset_v is None:
            self.init_dataset()

        
        # get camera information
        c2w, focal, center, cam_idxs = self.get_camera_data(idx, q_idx, self.N_samples)

        # get kp index and kp, skt, bone, cyl
        kp_idxs, kps, bones, skts, cyls = self.get_pose_data(idx, q_idx, self.N_samples)
        kp_idxs_v, kps_v, cyls_v, A_dash, m_normal, avg_D = self.get_pose_data_v(idx, q_idx, self.N_samples)
        
        '''Find overlap_rays area'''
        def simple_container(box2d_overlap, box2d_v_overlap):
            tl, br = box2D_real = box2d_overlap
            tl_v, br_v  = box2D_virt = box2d_v_overlap

            bb1 = {'x1':tl[0], 'x2':br[0], 'y1':tl[1], 'y2':br[1]}
            bb2 = {'x1':tl_v[0], 'x2':br_v[0], 'y1':tl_v[1], 'y2':br_v[1]}
            # calculate IOU
            iou_val = get_iou(bb1, bb2)
            return iou_val

        N_rand_ratio = 1.0
        self.overlap_found = False
        inter_box = None

        if self.overlap_rays:
            overlap_thshd = 0.2 # 20%
            
            if self.args.debug:
                pass
            '''single-frame overlap_rays assessment'''
            iou_vals = np.array(list(map(lambda x,y:simple_container(x,y), [self.box2d_overlap[idx]], [self.box2d_v_overlap[idx]])))
            
            if iou_vals[0]>=overlap_thshd:
                box2D_real = self.box2d_overlap[idx]
                box2D_virt = self.box2d_v_overlap[idx]
                self.overlap_found = True
                N_rand_ratio = 0.7 # 70% for foreground masks, 30% for overlap areas
           

        '''
        # all frames debug overlap_rays assessment 
        iou_vals = np.array(list(map(lambda x,y:simple_container(x,y), self.box2d_overlap, self.box2d_v_overlap)))
        bools = iou_vals>overlap_thshd
        n_overlaps = np.sum(bools)
        ov_ratio = n_overlaps/len(iou_vals)
        print(f"n_overlaps:{n_overlaps} ov_ratio:{ov_ratio}")
        ov_idxs = np.where(bools)[0]; 
        ov_idxs, = np.where(bools); 
        
        # check the boxes
        box2D_real = self.box2d_overlap[idx]
        box2D_virt = self.box2d_v_overlap[idx]
        chk_img = self.dataset['imgs'][idx].reshape(1080,1920,3)
        plt.imshow(chk_img); plt.axis("off")

        plot_bbox2D(box2D_real, plt=plt, color="green")
        plot_bbox2D(box2D_virt, plt=plt, color="red")
        plt.savefig(f"checkers/imgs/bbox2D.jpg", dpi=150, bbox_inches='tight', pad_inches = 0)
        pdb.set_trace()
        ''' 

        # sample pixels
        pixel_idxs, r_full_idxs, r_samp_mask = self.sample_pixels(idx, q_idx, N_rand_ratio)

        v_empty=False
        # in case mask is empty
        pixel_idxs_v, v_empty, v_full_idxs, v_samp_mask = self.sample_pixels_v(idx, q_idx, N_rand_ratio)
        
        n_overlap_pixels_dups = np.array([-1])
        n_rays_per_img_dups = np.array([-1])

        if self.overlap_rays and self.overlap_found:

            pixel_idxs_overlap, inter_box, union_idxs = self.sample_overlap_pixels(idx, q_idx, pixel_idxs, pixel_idxs_v, box2D_real,
                                                                        box2D_virt, N_rand_ratio,
                                                                        r_full_pixel_idxs=r_full_idxs, 
                                                                        v_full_pixel_idxs=v_full_idxs, 
                                                                        r_smask = r_samp_mask, 
                                                                        v_smask = v_samp_mask)
            


            pixel_idxs = np.concatenate([pixel_idxs, pixel_idxs_overlap])
            pixel_idxs_v = np.concatenate([pixel_idxs_v, pixel_idxs_overlap])

            n_overlap_pixels_dups = np.array([len(pixel_idxs_overlap)])
            n_rays_per_img_dups = np.array([len(pixel_idxs)])

        n_overlap_pixels_dups = n_overlap_pixels_dups.repeat(self.N_samples, 0)
        n_rays_per_img_dups = n_rays_per_img_dups.repeat(self.N_samples, 0)

        rays_o, rays_d, _ = self.get_rays(c2w, focal, pixel_idxs, center)
        if not v_empty:
            rays_o_v, rays_d_v, _ = self.get_rays_v(c2w, focal, pixel_idxs_v, center)

        # load the image, foreground and background, and get values from sampled pixels
        rays_rgb, fg, bg = self.get_img_data(idx, pixel_idxs, inter_box=inter_box,
                                    n_overlap_pixel_ids=n_overlap_pixels_dups)
        if not v_empty:
            rays_rgb_v, fg_v, bg_v = self.get_img_data_v(idx, pixel_idxs_v,
                                                n_overlap_pixel_ids=n_overlap_pixels_dups)

        # #-------------------------------------------
        """
        # debug pose
        first =  0 
        # H, W = 1080, 1920
        N,H,W,C = self.dataset['img_shape']
        chk_img = self.dataset['imgs'][idx].reshape(H, W, 3) 
        msk = self.dataset['sampling_masks'][idx].reshape(H, W, 1)
        chk_img = chk_img.copy() * msk.copy()
            
        white_bg = np.ones((H,W,3)) * 255
        white_bg = white_bg * (1-msk.copy())
        # blend 
        chk_img = (chk_img + white_bg).astype(np.uint8)
        c2ws_expanded = c2w[None, ...]
        kp2d = skeleton3d_to_2d(kps[first:first+1], c2ws_expanded, int(self.HW[0]), int(self.HW[1]), focal, center[None, ...])
        plot_skeleton2d(kp2d[first], img=chk_img)
        plt.axis("off")
        plt.savefig(f"{self.check_folder}/danbo_pk/kp_3d_to_2d_{idx:04d}.png", dpi=300, bbox_inches='tight', pad_inches = 0)
        pdb.set_trace()
        """
        """
        # debug mask and pose
        H, W = 1080, 1920 #940, 1285
        from core.utils.skeleton_utils import draw_skeletons_3d
        import imageio
        img = self.dataset['imgs'][idx].reshape(H, W, 3)
        msk = self.dataset['sampling_masks'][idx].reshape(H, W, 1)
        img = img.copy() * msk.copy()
        skel_img = draw_skeletons_3d(img[None], kps[:1], c2w[None], H, W, focal[None], center[None])
        imageio.imwrite(f'checkers/danbo_pk/{idx}.png', skel_img.reshape(H, W, 3))
        print(cam_idxs)
        pdb.set_trace()
        """
        """
        # debug patch
        import imageio
        target_s = (rays_rgb.reshape(self.patch_size, self.patch_size, 3) * 255).astype(np.uint8)
        imageio.imwrite(f'h36m_debug/{idx}.png', target_s)
        """
        
        # -----------------------------
        pre_select = 0
        idx_repeat = np.array([idx]).repeat(self.N_samples, 0)
        pixel_loc_repeat = self._2d_pixel_loc[pixel_idxs[pre_select:pre_select+1]].repeat(self.N_samples, 0)
        pixel_loc_v_repeat = self._2d_pixel_loc[pixel_idxs_v[pre_select:pre_select+1]].repeat(self.N_samples, 0)
        #------------------------------

        return_dict = {'rays_o': rays_o,
                       'rays_d': rays_d,

                       'target_s': rays_rgb,
                       'kp_idx': kp_idxs,
                       'kp3d': kps,

                       'bones': bones,
                       'skts': skts,
                       'cyls': cyls,

                       'cam_idxs': cam_idxs,
                       'fgs': fg,
                       'bgs': bg,

                       'A_dash':A_dash,
                       "m_normal": m_normal,
                        "avg_D": avg_D,
                        "idx_repeat": idx_repeat,
                        "pixel_loc_repeat": pixel_loc_repeat,
                        "pixel_loc_v_repeat": pixel_loc_v_repeat,
                       }

        if not v_empty:
            return_dict['rays_o_v'] = rays_o_v
            return_dict['rays_d_v'] = rays_d_v
            return_dict['target_s_v'] = rays_rgb_v
            return_dict['kp_idx_v'] = kp_idxs_v
            return_dict['kp3d_v'] = kps_v
            return_dict['cyls_v'] = cyls_v
            return_dict['bgs_v'] = bg_v
            return_dict['fgs_v'] = fg_v

        # if overlap_found:
            return_dict["n_overlap_pixels_dups"] = n_overlap_pixels_dups
            return_dict["n_rays_per_img_dups"] = n_rays_per_img_dups
            
        return return_dict

    def __len__(self):
        return self.data_len

    def init_len(self):
        if self._idx_map is not None:
            self.data_len = len(self._idx_map)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                self.data_len = len(f['imgs'])

    def init_dataset(self):

        if self.dataset is not None and self.dataset_v is not None:
            return
        print('init dataset')

        self.dataset = h5py.File(self.h5_path, 'r')
        self.dataset_v = h5py.File(self.h5_path_v, 'r')

    def init_meta(self):
        '''
        Init properties that can be read directly into memory (as they are small) - houses all method calls
        '''
        print('init meta')

        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        self.dataset_keys = [k for k in dataset.keys()]

        # initialize some attributes
        self.has_bg = 'bkgds' in self.dataset_keys
        self.centers = None
        self.A_dash = None
        self.m_normal = None
        self.avg_D = None

        if 'centers' in dataset:
            self.centers = dataset['centers'][:]

        # precompute mesh (for ray generation) to reduce computational cost
        img_shape = dataset['img_shape'][:]
        self._N_total_img = img_shape[0]
        self.HW = img_shape[1:3]
        mesh = np.meshgrid(np.arange(self.HW[1], dtype=np.float32),
                           np.arange(self.HW[0], dtype=np.float32),
                           indexing='xy')
                           
        self.mesh = mesh

        i, j = mesh[0].reshape(-1), mesh[1].reshape(-1)
        if self.centers is None:
            offset_y, offset_x = self.HW[0] * 0.5, self.HW[1] * 0.5
        else:
            # have per-image center. apply those during runtime
            offset_y = offset_x = 0.

        self._2d_pixel_loc = np.stack([i, j], axis=-1)

        # pre-computed direction, the first two cols
        # need to be divided by focal
        self._dirs = np.stack([ (i-offset_x),
                              -(j-offset_y),
                              -np.ones_like(i)], axis=-1)

        # pre-computed pixel indices: from image resolution (0 to n, from first cell to last cell)
        self._pixel_idxs = np.arange(np.prod(self.HW)).reshape(*self.HW)

        # store pose and camera data directly in memory (they are small)
        self.gt_kp3d = dataset['gt_kp3d'][:] if 'gt_kp3d' in self.dataset_keys else None
        self.kp_map, self.kp_uidxs = None, None # only not None when self.multiview = True

        # can be updated with refined poses, as _load_pose_data is called
        self.kp3d, self.bones, self.skts, self.cyls, self.import_idx_map = self._load_pose_data(dataset)
        self.focals, self.c2ws = self._load_camera_data(dataset)

        self.temp_validity = self.init_temporal_validity()

        if self.has_bg:
            self.bgs = dataset['bkgds'][:].reshape(-1, np.prod(self.HW), 3)
            self.bg_idxs = dataset['bkgd_idxs'][:].astype(np.int64)

        # TODO: maybe automatically figure this out
        self.skel_type = SMPLSkeleton

        dataset.close()


        '''virtual data starts here'''

        dataset_v = h5py.File(self.h5_path_v, 'r', swmr=True)
        self.dataset_keys_v = [k for k in dataset_v.keys()]

        # pre-computed direction, the first two cols
        # need to be divided by focal
        self._dirs_v = np.stack([ (i-offset_x),
                              -(j-offset_y),
                              -np.ones_like(i)], axis=-1)

        # pre-computed pixel indices
        self._pixel_idxs_v = np.arange(np.prod(self.HW)).reshape(*self.HW)

        # store pose and camera data directly in memory (they are small)
        self.gt_kp3d_v = dataset_v['gt_kp3d'][:] if 'gt_kp3d' in self.dataset_keys_v else None
        self.kp_map_v, self.kp_uidxs_v = None, None # only not None when self.multiview = True
    
        self.kp3d_v, self.cyls_v = self._load_pose_data_v(dataset_v)
        self.A_dash = dataset_v['A_dash'][:]
        self.m_normal = dataset_v['m_normal'][:]
        self.avg_D = dataset_v['avg_D'][:]

        self.temp_validity_v = self.init_temporal_validity()

        dataset_v.close()
        self.check_folder = "checkers" 

    def _load_pose_data(self, dataset):
        '''
        read pose data from .h5 file
        '''
        kp3d, bones, skts, cyls = dataset['kp3d'][:], dataset['bones'][:], \
                                    dataset['skts'][:], dataset['cyls'][:]
        if not self.load_refined:
            return kp3d, bones, skts, cyls, None
        return kp3d, bones, skts, cyls, None

    def _load_pose_data_v(self, dataset):
        '''
        read pose data from .h5 file
        '''
        kp3d, cyls = dataset['kp3d'][:], dataset['cyls'][:]
        if not self.load_refined:
            return kp3d, cyls
        return kp3d, cyls

    def _load_multiview_pose(self, dataset, kp3d, bones, skts, cyls):
        '''
        Multiview data for pose optimization, depends on dataset
        '''
        assert self._idx_map is None, 'Subset is not supported for multiview optimization'
        raise NotImplementedError

    def _load_camera_data(self, dataset):
        '''
        read camera data from .h5 file
        '''
        return dataset['focals'][:], dataset['c2ws'][:]

    def init_box2d(self, scale=1.3, box2d_overlap=False):
        '''
        pre-compute box2d
        '''
        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        l = len(self)
        H, W = self.HW

        temp = []
        for i in range(len(dataset['imgs'])):
            q_idx = i
            if self._idx_map is not None:
               idx = self._idx_map[q_idx]
            else:
                idx = q_idx

            # get camera information
            c2w, focal, center, cam_idxs = self.get_camera_data(idx, q_idx, 1)

            # get kp index and kp, skt, bone, cyl
            _, _, _, _, cyls = self.get_pose_data(idx, q_idx, 1)
            tl, br, _ = cylinder_to_box_2d(cyls[0], [H, W, focal], nerf_c2w_to_extrinsic(c2w),
                                            center=center, scale=scale)
            temp.append((tl, br))

        if box2d_overlap:
            self.box2d_overlap = np.array(temp.copy())
        else:
            self.box2d = np.array(temp.copy())


        dataset.close()

    def init_box2d_v(self, scale=1.3, box2d_overlap=False):
        '''
        pre-compute box2d
        '''
        # use images from real data
        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        l = len(self)
        H, W = self.HW
        
        temp_v = []
        for i in range(len(dataset['imgs'])):
            q_idx = i
            if self._idx_map is not None:
               idx = self._idx_map[q_idx]
            else:
                idx = q_idx

            # get camera information
            c2w, focal, center, cam_idxs = self.get_camera_data(idx, q_idx, 1)

            # get kp index and kp, cyl
            _, _, cyls_v, _, _, _ = self.get_pose_data_v(idx, q_idx, 1)
            tl, br, _ = cylinder_to_box_2d(cyls_v[0], [H, W, focal], nerf_c2w_to_extrinsic(c2w),
                                           center=center, scale=scale)
            temp_v.append((tl, br))

        if box2d_overlap:
            self.box2d_v_overlap = np.array(temp_v.copy())
        else:
            self.box2d_v = np.array(temp_v.copy())


        dataset.close()

    def init_temporal_validity(self):
        return None

    def get_camera_data(self, idx, q_idx, N_samples):
        ''' get camera data'''

        # real_idx: the real data we want to sample from
        # cam_idx: for looking up camera code
        real_idx, cam_idx = self.get_cam_idx(idx, q_idx)
        focal = self.focals[real_idx]
        c2w = self.c2ws[real_idx].astype(np.float32)

        center = None
        if self.centers is not None:
            center = self.centers[real_idx]

        cam_idx = np.array(cam_idx).reshape(-1, 1).repeat(N_samples, 1).reshape(-1)


        return c2w, focal, center, cam_idx

    def plot_current_pixel(self, img, pixel_idxs, pre_select, size):

        rand_idxs = np.random.choice(pixel_idxs, size, replace=False)
        plt.imshow(img); plt.axis("off")
        pts = self._2d_pixel_loc[rand_idxs]
        plt.scatter(pts[:,0], pts[:,1], c="g")
        plt.savefig(f"{checker_folder}/imgs/pixel_loc.jpg", dpi=300, bbox_inches='tight', pad_inches = 0)
        
        return rand_idxs

    def get_img_data(self, idx, pixel_idxs, inter_box=None, n_overlap_pixel_ids=None):
        '''
        get image data (in np.uint8), convert to float
        '''

        if n_overlap_pixel_ids[0] == -1:
            n_overlap_p_ids = len(pixel_idxs)
        else:
            n_overlap_p_ids = n_overlap_pixel_ids[0]
        
        # process real and overlap rays
        a = self.dataset['masks'][idx, pixel_idxs[:-n_overlap_p_ids]]
        b = self.dataset['masks'][idx, pixel_idxs[-n_overlap_p_ids:]]
        fg = np.concatenate([a, b]).astype(np.float32)

        '''Binarize the mask'''
        fg = (fg > 0.5).astype(np.int_)
        a = self.dataset['imgs'][idx, pixel_idxs[:-n_overlap_p_ids]]
        b = self.dataset['imgs'][idx, pixel_idxs[-n_overlap_p_ids:]]
        img = np.concatenate([a, b]).astype(np.float32) / 255.

        bg = None
        if self.has_bg:
            bg_idx = self.bg_idxs[idx]
            a = self.bgs[bg_idx, pixel_idxs[:-n_overlap_p_ids]]
            b = self.bgs[bg_idx, pixel_idxs[-n_overlap_p_ids:]]
            bg = np.concatenate([a, b]).astype(np.float32) / 255.

            if self.perturb_bg:
                noise = np.random.random(bg.shape).astype(np.float32)
                bg = (1 - fg) * noise + fg * bg # do not perturb foreground area!

            if self.mask_img:
                img = img * fg + (1. - fg) * bg

        return img, fg, bg

    def get_img_data_v(self, idx, pixel_idxs, n_overlap_pixel_ids=None):
        '''
        get image data (in np.uint8)
        '''

        if n_overlap_pixel_ids[0] == -1:
            n_overlap_p_ids = len(pixel_idxs)
        else:
            n_overlap_p_ids = n_overlap_pixel_ids[0]

        # process mirror and overlap rays
        a = self.dataset_v['masks'][idx, pixel_idxs[:-n_overlap_p_ids]]
        b = self.dataset_v['masks'][idx, pixel_idxs[-n_overlap_p_ids:]]
        fg = np.concatenate([a, b]).astype(np.float32)

        '''Binarize the mask'''
        fg = (fg > 0.5).astype(np.int_)

        '''reuse image from real data'''
        a = self.dataset['imgs'][idx, pixel_idxs[:-n_overlap_p_ids]]
        b = self.dataset['imgs'][idx, pixel_idxs[-n_overlap_p_ids:]]
        img = np.concatenate([a, b]).astype(np.float32) / 255.


        bg = None
        if self.has_bg:
            bg_idx = self.bg_idxs[idx]
            a = self.bgs[bg_idx, pixel_idxs[:-n_overlap_p_ids]]
            b = self.bgs[bg_idx, pixel_idxs[-n_overlap_p_ids:]]
            bg = np.concatenate([a, b]).astype(np.float32) / 255.

            if self.perturb_bg:
                noise = np.random.random(bg.shape).astype(np.float32)
                bg = (1 - fg) * noise + fg * bg # do not perturb foreground area!

            if self.mask_img:
                ''' set the fg in the left to 1, everything else to 0
                    set the fg in the right to 0, everything else to 1
                hence, get the foreground from the left and place it in the right background'''
                img = img * fg + (1. - fg) * bg
        return img, fg, bg


    def sample_overlap_pixels(self, idx, q_idx, pixel_idxs, pixel_idxs_v, box2D_real, box2D_virt, N_rand_ratio,
                                r_full_pixel_idxs=None, v_full_pixel_idxs=None, r_smask = None, v_smask = None):
        p = self.patch_size
        N_rand = self.N_samples // int(p**2)
        N_rand_overlap = np.ceil(N_rand * (1-N_rand_ratio)).astype(np.int_) # 70% fog, 30% overlap_rays

        tl, br = box2D_real
        tl_v, br_v = box2D_virt

        # extract intersection area
        max_x1 = max(tl[0], tl_v[0])
        min_x2 = min(br[0], br_v[0])
        max_y1 = max(tl[1], tl_v[1])
        min_y2 = min(br[1], br_v[1])
        inter_box = [[max_x1, max_y1], [min_x2, min_y2]]
        # meshgrid pre-computes the raw pixel's 2d locations in parallel
        valid_h, valid_w = np.meshgrid(np.arange(max_y1, min_y2, dtype=np.float32),
                           np.arange(max_x1, min_x2, dtype=np.float32),
                           indexing='xy')
        
        h, w = self.HW # relative to original image
        # convert 2D locations to indices 
        valid_idxs =  (valid_h * w + valid_w).reshape(-1)

        '''Union of both real and virt masks'''
        union_mask = r_smask + v_smask
        union_mask = (union_mask > 0.5).astype(np.int_)

        """the comma unrolls the single-element tuple"""
        union_idxs, = np.where(union_mask>0)

        # remove others from the list
        complement_idxs = set(valid_idxs) - set(union_idxs)
        occluded_mask_idxs = list(set(valid_idxs) - set(complement_idxs))

        sampled_idxs = np.random.choice(occluded_mask_idxs,
                                        N_rand_overlap,
                                        replace=False)

        sampled_idxs = np.sort(sampled_idxs).astype(np.int_)
        return sampled_idxs, inter_box, union_idxs

    def sample_pixels(self, idx, q_idx, N_rand_ratio=1.0):
        '''
        return sampled pixels (in (H*W,) indexing, not (H, W))
        '''
        p = self.patch_size
        N_rand = self.N_samples // int(p**2)

        if self.overlap_rays:
            N_rand = np.floor(N_rand*N_rand_ratio).astype(np.int_)

        # assume sampling masks are of shape (N, H, W, 1)
        sampling_mask = self.dataset['sampling_masks'][idx].reshape(-1)
        '''Binarize mask'''
        sampling_mask = (sampling_mask > 0.5).astype(np.int_)
        
        """the comma unrolls the single-element tuple"""
        valid_idxs, = np.where(sampling_mask>0)
        
        # when there is no segmentation detected in real mask
        if len(valid_idxs) == 0 or len(valid_idxs) < N_rand:
            valid_idxs = np.arange(len(sampling_mask))

        sampled_idxs = np.random.choice(valid_idxs,
                                    N_rand,
                                    replace=False)
        if self.patch_size > 1:

            H, W = self.HW
            hs, ws = sampled_idxs // W, sampled_idxs % W
            # clip to valid range
            hs = np.clip(hs, a_min=0, a_max=H-p)
            ws = np.clip(ws, a_min=0, a_max=W-p)
            _s = []
            for h, w in zip(hs, ws):
                patch = self._pixel_idxs[h:h+p, w:w+p].reshape(-1)
                _s.append(patch)

            sampled_idxs = np.array(_s).reshape(-1)

        if isinstance(self.N_nms, int):
            N_nms = self.N_nms
        else:
            # roll a dice
            N_nms = int(self.N_nms > np.random.random())

        if N_nms > 0:
            # replace some empty-space samples of out-of-mask samples
            nms_idxs = self._sample_in_box2d(idx, q_idx, sampling_mask, N_nms)

            sampled_idxs = np.sort(sampled_idxs)
            sampled_idxs[np.random.choice(len(sampled_idxs), size=(N_nms,), replace=False)] = nms_idxs

        sampled_idxs = np.sort(sampled_idxs)
        return sampled_idxs, valid_idxs, sampling_mask

    
    def sample_pixels_v(self, idx, q_idx, N_rand_ratio=1.0):
        '''
        return sampled pixels_v (in (H*W,) indexing, not (H, W))
        '''
        p = self.patch_size
        N_rand = self.N_samples // int(p**2)

        if self.overlap_rays:
            N_rand = np.floor(N_rand*N_rand_ratio).astype(np.int_) 

        # assume sampling masks are of shape (N, H, W, 1)
        sampling_mask_v = self.dataset_v['sampling_masks'][idx].reshape(-1)
        sampling_mask_v = (sampling_mask_v > 0.5).astype(np.int_)
        """the comma unrolls the single-element tuple"""
        valid_idxs_v, = np.where(sampling_mask_v>0)

        if len(valid_idxs_v)==0 or len(valid_idxs_v) < N_rand:
            '''blank or no segmentation for virtual?, sample real mask randomly'''
            sampling_mask = self.dataset['sampling_masks'][idx].reshape(-1)
            sampling_mask = (sampling_mask > 0.5).astype(np.int_)
            valid_idxs_v, = np.where(sampling_mask>0)

        sampled_idxs_v = np.random.choice(valid_idxs_v,
                                        N_rand,
                                        replace=False)

        if self.patch_size > 1:
            H, W = self.HW
            hs, ws = sampled_idxs_v // W, sampled_idxs_v % W
            # clip to valid range
            hs = np.clip(hs, a_min=0, a_max=H-p)
            ws = np.clip(ws, a_min=0, a_max=W-p)
            _s = []
            for h, w in zip(hs, ws):
                patch = self._pixel_idxs_v[h:h+p, w:w+p].reshape(-1)
                _s.append(patch)

            # update sampled idxs to fall within valid range
            sampled_idxs_v = np.array(_s).reshape(-1)

        if isinstance(self.N_nms_v, int):
            N_nms = self.N_nms_v
        else:
            # roll a dice
            N_nms = int(self.N_nms_v > np.random.random())

        if N_nms > 0:
            # replace some empty-space samples of out-of-mask samples
            nms_idxs = self._sample_in_box2d_v(idx, q_idx, sampling_mask_v, N_nms)
            sampled_idxs_v = np.sort(sampled_idxs_v)
            sampled_idxs_v[np.random.choice(len(sampled_idxs_v), size=(N_nms,), replace=False)] = nms_idxs

        sampled_idxs_v = np.sort(sampled_idxs_v)
        return sampled_idxs_v, False, valid_idxs_v, sampling_mask_v

    def _sample_in_box2d(self, idx, q_idx, fg, N_samples):

        H, W = self.HW
        # get bounding box
        real_idx, _ = self.get_cam_idx(idx, q_idx)
        tl, br = self.box2d[real_idx].copy()

        fg = fg.reshape(H, W)
        cropped = fg[tl[1]:br[1], tl[0]:br[0]]
        vy, vx = np.where(cropped < 1)

        # put idxs from cropped ones back to the non-cropped ones
        vy = vy + tl[1]
        vx = vx + tl[0]
        idxs = vy * W + vx
        # This is faster for small N_samples
        selected_idxs = np.random.default_rng().choice(idxs, size=(N_samples,), replace=False)

        return selected_idxs

    def _sample_in_box2d_v(self, idx, q_idx, fg, N_samples):

        H, W = self.HW
        # get bounding box
        real_idx, _ = self.get_cam_idx(idx, q_idx)
        tl, br = self.box2d_v[real_idx].copy()

        fg = fg.reshape(H, W)
        cropped = fg[tl[1]:br[1], tl[0]:br[0]]
        vy, vx = np.where(cropped < 1)

        # put idxs from cropped ones back to the non-cropped ones
        vy = vy + tl[1]
        vx = vx + tl[0]
        idxs = vy * W + vx
        # This is faster for small N_samples
        selected_idxs_v = np.random.default_rng().choice(idxs, size=(N_samples,), replace=False)

        return selected_idxs_v

    def get_rays(self, c2w, focal, pixel_idxs, center=None):

        dirs = self._dirs[pixel_idxs].copy()
        pic_loc2d = self._2d_pixel_loc[pixel_idxs].copy()

        if center is not None:
            center = center.copy()
            center[1] *= -1
            dirs[..., :2] -= center

        dirs[:, :2] /= focal

        I = np.eye(3)

        if np.isclose(I, c2w[:3, :3]).all():
            rays_d = dirs # no rotation required if rotation is identity
        else:
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
        return rays_o.copy(), rays_d.copy(), pic_loc2d.copy()

    def get_rays_v(self, c2w, focal, pixel_idxs, center=None):

        dirs = self._dirs_v[pixel_idxs].copy()
        pic_loc2d = self._2d_pixel_loc[pixel_idxs].copy()

        if center is not None:
            center = center.copy()
            center[1] *= -1
            dirs[..., :2] -= center

        # goes from 2d pixel location to 3d point with initial direction (with origin at the cam position).
        dirs[:, :2] /= focal

        I = np.eye(3)

        if np.isclose(I, c2w[:3, :3]).all():
            rays_d = dirs # no rotation required if rotation is identity
        else:
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
        return rays_o.copy(), rays_d.copy(), pic_loc2d.copy()

    def get_pose_data(self, idx, q_idx, N_samples):

        # real_idx: the real data we want to sample from
        # kp_idx: for looking up the optimized kp in poseopt layer (or other purpose)
        real_idx, kp_idx = self.get_kp_idx(idx, q_idx)

        kp = self.kp3d[real_idx:real_idx+1].astype(np.float32)
        bone = self.bones[real_idx:real_idx+1].astype(np.float32)
        cyl = self.cyls[real_idx:real_idx+1].astype(np.float32)
        skt = self.skts[real_idx:real_idx+1].astype(np.float32)

        # TODO: think this part through
        temp_val = None
        if self.temp_validity is not None:
            temp_val = self.temp_validity[real_idx:real_idx+1]

        kp_idx = np.array([kp_idx]).repeat(N_samples, 0)
        kp = kp.repeat(N_samples, 0)
        bone = bone.repeat(N_samples, 0)
        cyl = cyl.repeat(N_samples, 0)
        skt = skt.repeat(N_samples, 0)

        return kp_idx, kp, bone, skt, cyl

    def get_pose_data_v(self, idx, q_idx, N_samples):

        # real_idx: the real data we want to sample from
        # kp_idx: for looking up the optimized kp in poseopt layer (or other purpose)
        real_idx, kp_idx = self.get_kp_idx(idx, q_idx)

        kp = self.kp3d_v[real_idx:real_idx+1].astype(np.float32)
        cyl = self.cyls_v[real_idx:real_idx+1].astype(np.float32)
        A_dash = self.A_dash[real_idx:real_idx+1].astype(np.float32)
        m_normal = self.m_normal[real_idx:real_idx+1].astype(np.float32)
        avg_D = self.avg_D 

        # TODO: think this part through
        temp_val = None
        if self.temp_validity_v is not None:
            temp_val = self.temp_validity_v[real_idx:real_idx+1]

        kp_idx = np.array([kp_idx]).repeat(N_samples, 0)
        kp = kp.repeat(N_samples, 0)
        cyl = cyl.repeat(N_samples, 0)
        A_dash = A_dash.repeat(N_samples, 0)
        m_normal = m_normal.repeat(N_samples, 0)
        avg_D = avg_D.repeat(N_samples, 0)

        return kp_idx, kp, cyl, A_dash, m_normal, avg_D


    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx, q_idx

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx, q_idx

    def _get_subset_idxs(self, render=False):
        '''return idxs for the subset data that you want to train on.
        Returns:
        k_idxs: idxs for retrieving pose data from .h5
        c_idxs: idxs for retrieving camera data from .h5
        i_idxs: idxs for retrieving image data from .h5
        kq_idxs: idx map to map k_idxs to consecutive idxs for rendering
        cq_idxs: idx map to map c_idxs to consecutive idxs for rendering
        '''
        if self._idx_map is not None:
            # queried_idxs
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))

        else:
            # queried == actual index
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(len(self.kp3d))
            _c_idxs = _cq_idxs = np.arange(len(self.c2ws))

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

    def get_meta(self):
        '''
        return metadata needed for other parts of the code.
        '''

        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        rest_pose = dataset['rest_pose'][:]
        
        # get idxs to retrieve the correct subset of meta-data
        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs()

        # prepare HWF
        H, W = self.HW
        if not np.isscalar(self.focals):
            H = np.repeat([H], len(c_idxs), 0)
            W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])

        # prepare center if there's one
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()

        # average beta
        betas = dataset['betas'][:]
        try:
            if len(betas) > 1:
                betas = betas[k_idxs]
        except:
            import pdb; pdb.set_trace()
            print
        betas = betas.mean(0, keepdims=True).repeat(len(betas), 0)

        data_attrs = {
            'hwf': hwf,
            'center': center,
            'c2ws': self.c2ws[c_idxs],
            'near': 60., 'far': 100., # don't really need this
            'n_views': self.data_len,
            # skeleton-related info
            'skel_type': self.skel_type,
            'joint_coords': get_per_joint_coords(rest_pose, self.skel_type),
            'rest_pose': rest_pose,
            'gt_kp3d': self.gt_kp3d[k_idxs] if self.gt_kp3d is not None else None,
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
            'bones': self.bones[k_idxs],
            'betas': betas,
            'kp_map': self.kp_map, # important for multiview setting
            'kp_uidxs': self.kp_uidxs, # important for multiview setting
        }


        '''virtual data starts here'''

        data_attrs['gt_kp3d_v'] = self.gt_kp3d_v[k_idxs] if self.gt_kp3d_v is not None else None
        data_attrs['kp3d_v'] = self.kp3d_v[k_idxs]
        data_attrs['A_dash'] = self.A_dash[k_idxs]
        data_attrs['m_normal'] = self.m_normal[k_idxs]
        data_attrs['avg_D'] = self.avg_D[:]
        #data_attrs['kp_map'] = self.kp_map, # important for multiview setting
        #data_attrs['kp_uidxs'] = self.kp_uidxs, # important for multiview setting

        dataset.close()
        return data_attrs

    def get_render_data(self):

        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs(render=True)

        # grab only a subset (15 images) for rendering
        kq_idxs = kq_idxs[::self.render_skip][:self.N_render]
        cq_idxs = cq_idxs[::self.render_skip][:self.N_render]
        i_idxs = i_idxs[::self.render_skip][:self.N_render]
        k_idxs = k_idxs[::self.render_skip][:self.N_render]
        c_idxs = c_idxs[::self.render_skip][:self.N_render]

        
        # get images if split == 'render'
        # note: needs to have self._idx_map
        H, W = self.HW
        render_imgs = dataset['imgs'][i_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][i_idxs].reshape(-1, H, W, 1)
        '''Binarize mask'''
        render_fgs = (render_fgs > 0.5).astype(np.int_)
        render_bgs = self.bgs.reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bg_idxs = self.bg_idxs[i_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])

        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        
        # TODO: c_idxs, k_idxs ... confusion

        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            'idx_map': self._idx_map,
            
            # camera data
            'cam_idxs': c_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[c_idxs],
            'A_dash': self.A_dash[c_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': k_idxs,
            'kp_idxs_len': len(self.kp3d),
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
            'bones':self.bones[k_idxs],
        }

        '''
        from core.utils.skeleton_utils import draw_skeletons_3d
        import imageio
        skel_imgs = draw_skeletons_3d((render_data['imgs']*255).astype(np.uint8), render_data['kp3d'], render_data['c2ws'], *hwf, center)
        import pdb; pdb.set_trace()
        print
        '''
        '''
        '''

        dataset.close()

        return render_data

class PoseRefinedDataset(BaseH5Dataset):

    def __init__(self, *args, load_refined=False, **kwargs):
        self.load_refined = load_refined
        super(PoseRefinedDataset, self).__init__(*args, **kwargs)

    def _load_pose_data(self, dataset):
        '''
        read pose data from .h5 or refined poses
        NOTE: refined poses are defined in a per-dataset basis.
        '''
        
        if not self.load_refined:
            return super(PoseRefinedDataset, self)._load_pose_data(dataset)

        assert hasattr(self, 'refined_paths'), \
            f'Paths to refined poses are not defined for {self.__class__}.'

        refined_path, legacy = self.refined_paths 
        print(f'Read refined poses from {refined_path}')
        idx_map_in_ckpt = self.idx_map_in_ckpt

        po_data = pose_ckpt_to_pose_data(refined_path, ext_scale=0.001, legacy=legacy, idx_map_in_ckpt=idx_map_in_ckpt)
        
        import_idx_map = None
        # the first 4 is kp3d, bones, skts, cyls
        kp3d, bones, skts, cyls = po_data[:4]
        if idx_map_in_ckpt:
            import_idx_map = po_data[6]
        
        if self.multiview:
            return self._load_multiview_pose(dataset, kp3d, bones, skts, cyls)
        
        
        return kp3d, bones, skts, cyls, import_idx_map

class ConcatH5Dataset(ConcatDataset):
    # TODO: poor naming
    # TODO: also allows it to call get_pose?
    def __init__(self, datasets):
        super().__init__(datasets)

        metas = [d.get_meta() for d in self.datasets]
        self.cumulative_views = np.cumsum([m['n_views'] for m in metas])
        self.cumulative_kps = np.cumsum([len(m['kp3d']) for m in metas])

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        ret = self.datasets[dataset_idx][sample_idx]

        if dataset_idx != 0:
            # properly offset the index
            ret['cam_idxs'] = ret['cam_idxs'] + self.cumulative_views[dataset_idx - 1]
            ret['kp_idx'] = ret['kp_idx'] + self.cumulative_kps[dataset_idx - 1]
        # assume 1 dataset == 1 subject
        ret['subject_idxs'] = np.array([dataset_idx]).repeat(len(ret['cam_idxs']), 0)

        return ret

    def get_meta(self):
        # get meta from all dataset
        metas = [d.get_meta() for d in self.datasets]

        merged_meta = {'n_subjects': len(self.datasets)}
        to_stack = ['joint_coords', 'rest_pose']
        to_cat = ['gt_kp3d', 'kp3d', 'bones', 'betas']

        # TODO: currently assume all scalars are the same
        H = np.concatenate([m['hwf'][0] for m in metas])
        W = np.concatenate([m['hwf'][1] for m in metas])

        # focal lengths need special treatment
        max_dim = 1
        focal_lists = []
        for m in metas:
            hwf = m['hwf']
            if len(hwf[2].shape) == 2:
                max_dim = 2
            focal_lists.append(hwf[2].copy())
        
        if max_dim != 1:
            for i, focals in enumerate(focal_lists):
                if len(focals.shape) < 2 or focals.shape[1] < 2:
                    focals = focals.reshape(-1, 1).repeat(2, 1).copy()
                    focal_lists[0] = focals

        focals = np.concatenate(focal_lists)
        merged_meta['hwf'] = (H, W, focals)
        merged_meta['near'] = metas[0]['near']
        merged_meta['far'] = metas[0]['far']
        merged_meta['n_views'] = np.sum([m['n_views'] for m in metas])
        merged_meta['skel_type'] = metas[0]['skel_type']

        # stack arrays
        for k in to_stack:
            merged_meta[k] = np.stack([m[k] for m in metas], axis=0)

        # check if gt exists for all data. If not, ignore them
        has_gt = all(['gt_kp3d' in m for m in metas])
        for k in to_cat:
            if k == 'gt_kp3d' and (not has_gt):
                continue # no GT available
            try:
                merged_meta[k] = np.concatenate([m[k] for m in metas])
            except:
                print(f'Key {k} is skipped due to inconsistent shape. Stop if not expected')

        # handle rest pose
        kp_lens = np.cumsum([len(m['kp3d']) for m in metas])
        rest_pose_idxs = np.searchsorted(kp_lens, np.arange(len(merged_meta['kp3d'])),
                                         side='right')

        merged_meta['rest_pose_idxs'] = rest_pose_idxs
        merged_meta['n_subjects'] = len(self.datasets)
        return merged_meta

    def get_render_data(self):

        render_data = [d.get_render_data() for d in self.datasets]
        merged_data = {}

        # TODO: TEMPORARY HACK, TO ONLY RENDER ONE DATASET IF IMAGE SHAPES
        #       DOES NOT MATCH
        not_match = False
        for i in range(1, len(render_data)):
            H_prev, W_prev, _ = render_data[i-1]['hwf']
            H_cur, W_cur, _ = render_data[i]['hwf']
            if (H_prev != H_cur).any() or (W_prev != W_cur).any():
                not_match = True
                break
        if not_match:
            render_data = render_data[:1]


        # concate hwf
        H = np.concatenate([r['hwf'][0] for r in render_data])
        W = np.concatenate([r['hwf'][1] for r in render_data])
        focals = np.concatenate([r['hwf'][2] for r in render_data])
        merged_data['hwf'] = (H, W, focals)

        centers, has_center = [], []
        for r in render_data:
            center = r['center']
            has_center.append(center is not None)
            centers.append(center)
        has_center = any(has_center)

        if has_center:
            for i, center in enumerate(centers):
                # TODO: verify
                # fill in empty center
                if center is None:
                    H, W, _ = render_data[i]['hwf']
                    centers[i] = np.stack([H, W], axis=-1) // 2
                    import pdb; pdb.set_trace()
                    print
            merged_data['center'] = np.concatenate(centers)
        else:
            merged_data['center'] = None

        # data that can be concatenate directly
        to_cat = ['imgs', 'fgs', 'bgs', 'c2ws',
                  'kp3d', 'skts', 'bones']
        to_cat_offset = ['cam_idxs', 'kp_idxs', 'bg_idxs']
        for k in to_cat:
            merged_data[k] = np.concatenate([r[k] for r in render_data])

        # offset indexs in multi-dataset setting
        for k in to_cat_offset:
            # indices are expected to be stored in (idxs, max_idxs) tuples.
            cumulated_lens = np.cumsum([r[k+'_len'] for r in render_data])
            array = [render_data[0][k]] # first data does not need offset
            for i, r in enumerate(render_data[1:]):
                array.append(r[k] + cumulated_lens[i])
            merged_data[k] = np.concatenate(array)

        # create subject idxs.
        subject_idxs = []
        for i, r in enumerate(render_data):
            n_imgs = len(r['imgs'])
            subject_idxs.extend([i] * n_imgs)
        merged_data['subject_idxs'] = np.array(subject_idxs)

        return merged_data

class DatasetWrapper:

    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_meta(self):
        return self._dataset.get_meta()

    def get_render_data(self):
        return self._dataset.get_render_data()

    def __len__(self):
        return len(self._dataset)

class TemporalDatasetWrapper(DatasetWrapper):
    '''
    A wrapper for dataset with temporal info (e.g., temporal validity for loss)
    '''
    def __init__(self, dataset):
        super(TemporalDatasetWrapper, self).__init__(dataset)
        assert hasattr(self._dataset, 'temp_validity'), f'Dataset type {type(self._dataset)} does not support temporal loss!'

    def __getitem__(self, idx):
        ret_dict = self._dataset.__getitem__(idx)
        next_idx = (idx + 1) % len(self._dataset.temp_validity)

        # only valid when both previous and next poses are correlated
        temp_val = (self._dataset.temp_validity[idx] + self._dataset.temp_validity[next_idx]) // 2
        ret_dict['temp_val'] = temp_val.repeat(ret_dict['kp_idx'].shape[0], 0).astype(np.float32)
        return ret_dict

class CorrSampleDatasetWrapper(DatasetWrapper):
    '''
    A wrapper for sampling extra points for computing correspondence
    '''
    def __init__(self, dataset, skel_type=SMPLSkeleton, N_corr=16, width=0.05,
                 normalize=True):
        '''
        width: farther distance from the point to the bones
        '''
        super(CorrSampleDatasetWrapper, self).__init__(dataset)
        self.N_corr = N_corr
        self.width = width
        self.skel_type = skel_type
        self.joint_trees = self.skel_type.joint_trees
        self.normalize = normalize
        self.geo_dists = None
        self.max_dist = None

    def __getitem__(self, idx):
        ret_dict = self._dataset.__getitem__(idx)

        # the whole batch will have only a single pose
        kp3d = ret_dict['kp3d'][0]
        if self.geo_dists is None:
            self._init_geo_dists(kp3d)
            self.max_dist = np.abs(self.geo_dists).max()

        N_J = kp3d.shape[-2]
        # TODO: only work for SMPLSkeleton now

        # sample joint (excluding root)
        rand_joint = np.random.randint(1, high=N_J, size=(self.N_corr,))
        parent = self.joint_trees[rand_joint]
        kp_parent = kp3d[parent, :]
        kp_child = kp3d[rand_joint, :]

        # select a point between child and parent
        # bound the distance ratio to become [self.width, 1-self.width]
        rand_dist = (np.random.random(size=(self.N_corr, 1)) / (1 - self.width * 2)) + self.width
        dist_from_parent = (kp_child - kp_parent) * rand_dist
        samples = dist_from_parent + kp_parent
        dist_from_parent = (dist_from_parent**2).sum(-1)**0.5

        # add perturbs so the samples are away from the bone
        perturb = (np.random.random(size=(self.N_corr, 3)) * 2 - 1) * self.width
        perturb_samples = perturb + samples

        # from reference's parent to target points' parent
        geo_dists = self.geo_dists[parent[0], parent[:]].copy()

        # distance = distance bewteen parent + target distance from its parent
        #            + ref distance from parent
        # note that, if tar is a offsping of ref, then ref's distance will shorten their diff

        # ref_to_tar = np.abs(geo_dists) + dist_from_parent[1:] + \
        #              np.sign(geo_dists) * dist_from_parent[0]
        ref_to_tar = np.abs(geo_dists) + dist_from_parent
        ref_to_tar[1:] += np.sign(geo_dists)[1:] * dist_from_parent[0]

        if self.normalize:
            ref_to_tar = ref_to_tar / self.max_dist

        is_ref = np.zeros((self.N_corr,), dtype=np.float32)
        is_ref[0] = 1. # the first one is always reference points

        ret_dict['corr_is_ref'] = is_ref
        ret_dict['corr_samples'] = perturb_samples.astype(np.float32)
        ret_dict['corr_ref_to_tar'] = ref_to_tar.astype(np.float32)
        ret_dict['corr_kps'] = ret_dict['kp3d'][:1].repeat(self.N_corr, 0).astype(np.float32)
        ret_dict['corr_skts'] = ret_dict['skts'][:1].repeat(self.N_corr, 0).astype(np.float32)
        ret_dict['corr_bones'] = ret_dict['bones'][:1].repeat(self.N_corr, 0).astype(np.float32)

        return ret_dict

    def _init_geo_dists(self, kp):
        N_J = kp.shape[-2]
        dists = np.ones((N_J, N_J)) * 100000.
        joint_trees = self.joint_trees

        for i, parent in enumerate(joint_trees):
            dist = ((kp[i] - kp[parent])**2.0).sum()**0.5
            dists[i, parent] = dist
            dists[parent, i] = dist
            dists[i, i] = 0.

        for k in range(N_J):
            for i in range(N_J):
                for j in range(N_J):
                    new_dist = dists[i][k] + dists[k][j]
                    if dists[i][j] > new_dist:
                        dists[i][j] = new_dist
        # set distance to offspings as to negative.
        for i, e in enumerate(joint_trees):
            cur = i
            # backtrack
            while cur != 0:
                parent = joint_trees[cur]
                dists[parent][i] *= -1
                cur = parent

        self.geo_dists = dists

class RandIntGenerator:
    '''
    RandomInt generator that ensures all n data will be
    sampled at least one in every n iteration.
    '''

    def __init__(self, n, generator=None):
        self._n = n
        self.generator = generator

    def __iter__(self):

        if self.generator is None:
            # TODO: this line is buggy for 1.7.0 ... but has to use this for 1.9?
            #       it induces large memory consumptions somehow
            generator = torch.Generator(device=torch.tensor(0.).device)
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        yield from torch.randperm(self._n, generator=generator)

    def __len__(self):
        return self._n

class RayImageSampler(Sampler):
    '''
    TODO: does this work with ConcatDataset?
    TODO: handle train/val
    '''

    def __init__(self, data_source, N_images=1024,
                 N_iter=None, generator=None):
        self.data_source = data_source
        self.N_images = N_images
        self._N_iter = N_iter
        self.generator = generator

        if self._N_iter is None:
            self._N_iter = len(self.data_source)

        self.sampler = RandIntGenerator(n=len(self.data_source))

    def __iter__(self):

        sampler_iter = iter(self.sampler)
        batch = []
        for i in range(self._N_iter):
            # get idx until we have N_images in batch
            while len(batch) < self.N_images:
                try:
                    idx = next(sampler_iter)
                except StopIteration:
                    sampler_iter = iter(self.sampler)
                    idx = next(sampler_iter)
                batch.append(idx.item())

            # return and clear batch cache
            yield np.sort(batch)
            batch = []

    def __len__(self):
        return self._N_iter

def ray_collate_fn(batch):

    batch = default_collate(batch)
    # default collate results in shape (N_images, N_rays_per_images, ...)
    # flatten the first two dimensions.
    batch = {k: batch[k].flatten(end_dim=1) for k in batch}
    batch['rays'] = torch.stack([batch['rays_o'], batch['rays_d'], 
                                batch['rays_o_v'], batch['rays_d_v']], dim=0)
    return batch

