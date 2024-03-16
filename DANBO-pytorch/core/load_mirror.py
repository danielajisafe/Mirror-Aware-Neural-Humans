import os
import cv2
import glob
import math
import torch
import pickle
import imageio
import numpy as np
from scipy.io import loadmat

from .process_spin import write_to_h5py
from .dataset import BaseH5Dataset, PoseRefinedDataset
from .utils.skeleton_utils import *
from .utils.ray_utils import get_rays_np

# a fixed scale that roughly make skeleton of all datasets
# to be of similar range
dataset_ext_scale = 0.25 / 0.00035

class MirrorDataset(PoseRefinedDataset):
    '''
    Images/cameras are organized as the following:
    imgs: shape (N_cams, N_kps, ...)
    c2ws: shape (N_cams, N_kps, ...)

    To get the camera/view id, simply divide the index by N_kps.
    And to get the kp id, take modulo of the index
    '''
    #render_skip = 1
    #N_render = 9
    render_skip = 1
    N_render =  1

    rand_kps = {'230': 'data/mirror/mirror_rand_230.npy',
                '400': 'data/mirror/mirror_rand_400.npy',
                }

    def __init__(self, *args, N_rand_kps=None, N_cams=None, **kwargs):
        # TODO: handle rand kps cases
        self._N_rand_kps = N_rand_kps

        self._N_kps = None
        if N_rand_kps is not None:
            self._N_kps = int(N_rand_kps.split('_')[-1])
        self._N_cams = N_cams

        super(MirrorDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        if self.split == 'val':
            self.h5_path = self.h5_path.replace('train_h5py', 'val_h5py')

        super(MirrorDataset, self).init_meta()
        N_total_cams = len(self.c2ws) // len(self.kp3d)
        N_total_kps = len(self.kp3d)


        # get the right numbers of kps/cameras
        # NOTE: assume that we use the same set of kps for all views,
        #       and the cameras data are arranged as (N_cams, N_kps)
        if self._N_kps is None:
            self._N_kps = N_total_kps
        if self._N_cams is None:
            self._N_cams = N_total_cams

        #TODO: temporarily this
        if self.split == 'val':
            self._idx_map = np.load('data/mirror/mirror_val_idxs.npy')[0::2]
            return

        if self._N_kps == N_total_kps and self._N_cams == N_total_cams:
            return

        # create _idx_map if otherwise
        # TODO: hard-coded for now
        if self._N_rand_kps is None:
            selected_kps = np.arange(N_total_kps)
        else:
            selected_kps = np.unique(np.load(self.rand_kps[self._N_rand_kps]))
        
        selected_cams = np.array([0, 3, 6])
        self._idx_map = np.concatenate([selected_kps + N_total_kps * c for c in selected_cams])

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx % len(self.kp3d), q_idx % self._N_kps

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        # NOTE: we already expand cameras data to the same shape of images,
        #       so no special treament needed on idx
        return idx, q_idx // self._N_kps

    def get_meta(self):
        data_attrs = super(MirrorDataset, self).get_meta()
        data_attrs['n_views'] = self._N_cams
        return data_attrs
