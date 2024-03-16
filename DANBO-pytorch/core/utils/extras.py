

import json
import torch
import os,sys
import pickle

from re import L
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.utils.skeletons import CMUSkeleton
from core.utils.skeleton_utils import get_parent_idx, verify_get_parent_idx


def calc_bone_length(poses, root_id, joint_trees):
    '''calculate 
    - bone length, mean bone length, and std deviation across a sequence of poses
    - assumes hip-first ordering format
    If a person is 1.5m tall, arm bone length should be roughly around 0.3m
    '''

    children_trans = torch.cat([poses[:, :root_id], poses[:, root_id+1:]], dim=1)[..., None]
    parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
    parent_trans = poses[:, parent_ids, :, None]
    bv = children_trans - parent_trans 

    bl = torch.norm(bv, dim=2).squeeze(2)
    mean_bl = bl.mean(0)
    std_bl = bl.std(0)
    return bl, mean_bl, std_bl, bv

def alpha_to_hip_1st(h_pose):
    '''Re-order alphapose so "hip starts first" (similar to h36m structure) based on kinematic chain '''
    ordering = torch.tensor([19,12,14,16,25,21,23,11,13,15,24,20,22,
                                18,0,2,4,1,3,17,5,7,9,6,8,10])  
    N_J = len(ordering)
    if h_pose.shape[0]==N_J: 
        new_pose = torch.index_select(h_pose, 0, ordering)
    elif h_pose.shape[1]== N_J:
        new_pose = torch.index_select(h_pose, 1, ordering)
    return new_pose

def hip_1st_to_alpha(h_pose, **k_pipe_kwargs):
    '''Re-order "hip-first 26 skeleton" to standard alphapose ordering'''
    ordering = torch.tensor([14,17,15,18,16,20,23,21,24,22,25,
                            7,1,8,2,9,3,19,13,0,11,5,12,6,10,4])
    N_J = len(ordering)
    if h_pose.shape[0]==N_J: 
        new_pose = torch.index_select(h_pose, 0, ordering)
    elif h_pose.shape[1]== N_J:
        new_pose = torch.index_select(h_pose, 1, ordering)
    return new_pose


def load_pickle(filename):
    ''' args: filename: saved filename to load pickle from'''
    with open(f'{filename}', 'rb') as handle:
        from_pickle = pickle.load(handle)
    return from_pickle

def save2pickle(filename, tuples):
    '''args:
        filename: name to save the resulting pickle to
        tuples: list of tuples to save to pickle file
    '''
    to_pickle = dict(tuples)
    with open(f'{filename}', 'wb') as handle:
        pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)

    w = norm_quat[:, 0]
    x = norm_quat[:, 1]
    y = norm_quat[:, 2]
    z = norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # NB: similar to Rodrigues' Rotation Formula and Zhou et al. CVPR 2019 Paper
    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 
        2 * xy - 2 * wz, 
        2 * wy + 2 * xz, 
        2 * wz + 2 * xy, 
        w2 - x2 + y2 - z2, 
        2 * yz - 2 * wx, 
        2 * xz - 2 * wy, 
        2 * wx + 2 * yz, 
        w2 - x2 - y2 + z2 
    ], dim=1).view(batch_size, 3, 3)

    return rotMat

def axisang_to_rot(axisang):
    """
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py  
    https://github.com/nkolot/SPIN/blob/5c796852ca7ca7373e104e8489aa5864323fbf84/utils/geometry.py#L9
    Args:
        The axis/rotation angle same as theta: size = [B, 3] in degree
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]    
    """

    # converts angle from degree to radian first
    angle = torch.norm(axisang + 1e-8, p=2, dim=-1)[..., None]
 
    axisang_norm = axisang / angle
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)

    quat = torch.cat([v_cos, v_sin * axisang_norm], dim=-1)
    rot = quat2mat(quat)
    return rot


def pad_mat_to_homogeneous(mat):
    """expects (3,4)"""
    last_row = torch.tensor([[0., 0., 0., 1.]]).to(mat.device)
    if mat.dim() == 3:
        last_row = last_row.expand(mat.size(0), 1, 4)
    return torch.cat([mat, last_row], dim=-2)

def mat_to_homo(mat):
    """expects (3,4)"""
    last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
    return np.concatenate([mat, last_row], axis=0)

class KinematicChain_Numpy(nn.Module):

    def __init__(self, rest_pose, skeleton_type=CMUSkeleton, use_rot6d=False, theta_shape=None, where=None):
        """
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        use_rot6d: bool, to use 6d rotation instead of 3d axis-angle representation.
                   (see: https://arxiv.org/abs/1812.07035) - continous 6D rotation
                   
        Paper notes: We use Adam optimization with batch size 64 and learning rate 10e−5 for 
        the first 10^4 iterations and 10e−6 for the remaining iterations.
        """
        super().__init__()

        assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"
        self.rest_pose = rest_pose
        self.skeleton_type = skeleton_type
        self.use_rot6d = use_rot6d
        
    def forward(self, theta, bone_factor=None, rest_pose=None, skeleton_type=None, **k_pipe_kwargs):
        """
        theta: float32, (B, N_joints, 3) or (B, N_joints, 6). SMPL pose parameters in axis-angle
               or 6d representation.
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        """

        if rest_pose is None:
            rest_pose = self.rest_pose
            B, N_J, _ = theta.shape # (B, N_J, _) 
            
            # match theta's dimention
            rest_pose = rest_pose[None].repeat(B, 0)

        if skeleton_type is None:
            skeleton_type = self.skeleton_type
        else:
            assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"

        idx = get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        verify_get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        joint_trees = np.array(idx)
        
        root_id = skeleton_type.root_id
        B, N_J, _ = theta.shape

        # turn rotation parameters (joint angles) into the proper 3x3 rotation matrices
        if self.use_rot6d:
            rots = rot6d_to_rotmat(theta.reshape(-1, 6)).reshape(B, N_J, 3, 3) # converts from 6D rep to proper 3x3 again
        else:
            '''I assume theta is by defaut in axis-angle representation'''
            rots = axisang_to_rot(theta.reshape(-1, 3)).reshape(B, N_J, 3, 3) # coverts axis-angle to proper 3x3 

        # l2w: local-to-world transformation.
        # concatenate the rotation and translation of the root joint
        # to get a 3x4 matrix |R T|
        #                     |0 1|

        # root_l2w = torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1)
        root_l2w = np.concatenate([rots[:, root_id], rest_pose[:, root_id, :, None]], axis=2)
        # pad it to 4x4
        root_l2w = mat_to_homo(root_l2w)

        # assume root_id == 0,
        # this is the per-joint-local to-world matrices
        l2ws = [root_l2w]

        # collect all rotations/translation except for the root one
        # B x (N_J - 1) x 3 x 3
        children_rots = torch.cat([rots[:, :root_id], rots[:, root_id+1:]], dim=1)
        # B x (N_J - 1) x 3 x 1
        children_trans = torch.cat([rest_pose[:, :root_id], rest_pose[:, root_id+1:]], dim=1)[..., None]
        parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
        # B x (N_J - 1) x 3 x 1
        parent_trans = rest_pose[:, parent_ids, :, None]
        # B x (N_J - 1) x 3 x 4, matrix |R T|
        bv = children_trans - parent_trans # (N, 15, 3, 1)

        if bone_factor is not None:
            eps = 1e-36
            bone_factor = torch.sqrt(bone_factor**2 + eps)
            if torch.all(bone_factor>0).item() != True:
                import ipdb; ipdb.set_trace()
            assert torch.all(bone_factor>0).item() == True, "bone factor should be positive"
            bv = bv * bone_factor 
            
        # concatenate the rotation and translation of other joints
        joint_rel_transforms = torch.cat([children_rots, bv], dim=-1) # --> [Rotation | BONE vectors]

        '''optimized rotation + bone vectors'''
        # pad to 4 x 4: |R T|
        #               |0 1|
        joint_rel_transforms = pad_mat_to_homogeneous(joint_rel_transforms.reshape(-1, 3, 4))
        joint_rel_transforms = joint_rel_transforms.reshape(B, N_J-1, 4, 4) # (N, 15, 4, 4)
        
        '''run kinematic chain here successively, starting with the root rotation 
        at [zero pose|identity matrix| x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) + 3d translation| point]
        against other joints at [zero pose + bone vector]
        '''
        for i, parent in enumerate(parent_ids): # (15)
            l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
        l2ws = torch.stack(l2ws, dim=-3) # (N, 16, 4, 4)
  
        # the 3d keypoints are the translation part of the final
        # per-joint local-to-world matrices
        kp3d = l2ws[..., :3, -1] # (N, 16, 3)

        # rotational part of the per-joint local-to-world matrices
        orient = l2ws[..., :3, :3]

        return kp3d, orient, l2ws, bone_factor


class KinematicChain(nn.Module):

    def __init__(self, rest_pose, skeleton_type=CMUSkeleton, use_rot6d=False, theta_shape=None, where=None):
        """
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        use_rot6d: bool, to use 6d rotation instead of 3d axis-angle representation.
                   (see: https://arxiv.org/abs/1812.07035) - continous 6D rotation
                   
        Paper notes: We use Adam optimization with batch size 64 and learning rate 10e−5 for 
        the first 10^4 iterations and 10e−6 for the remaining iterations.
        """
        super().__init__()

        assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"
        self.rest_pose = rest_pose
        self.skeleton_type = skeleton_type
        self.use_rot6d = use_rot6d
        
    def forward(self, theta, bone_factor=None, rest_pose=None, skeleton_type=None, **k_pipe_kwargs):
        """
        theta: float32, (B, N_joints, 3) or (B, N_joints, 6). SMPL pose parameters in axis-angle
               or 6d representation.
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        """

        if rest_pose is None:
            rest_pose = self.rest_pose
            B, N_J, _ = theta.shape # (B, N_J, _) 
            # match theta's dimention
            rest_pose = rest_pose[None].expand(B, N_J, 3)
            
        if skeleton_type is None:
            skeleton_type = self.skeleton_type
        else:
            assert skeleton_type.root_id == 0, "Only support skeletons with root at 0!"

        idx = get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        verify_get_parent_idx(k_pipe_kwargs["joint_names"], k_pipe_kwargs["joint_parents"])
        joint_trees = np.array(idx)
        
        root_id = skeleton_type.root_id
        B, N_J, _ = theta.shape

        # turn rotation parameters (joint angles) into the proper 3x3 rotation matrices
        if self.use_rot6d:
            rots = rot6d_to_rotmat(theta.view(-1, 6)).view(B, N_J, 3, 3) # converts from 6D rep to proper 3x3 again
        else:
            '''I assume theta is by defaut in axis-angle representation'''
            rots = axisang_to_rot(theta.view(-1, 3)).view(B, N_J, 3, 3) # coverts axis-angle to proper 3x3 

        # l2w: local-to-world transformation.
        # concatenate the rotation and translation of the root joint
        # to get a 3x4 matrix |R T|
        #                     |0 1|
        root_l2w = torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1)
        # pad it to 4x4
        root_l2w = pad_mat_to_homogeneous(root_l2w)

        # assume root_id == 0,
        # this is the per-joint-local to-world matrices
        l2ws = [root_l2w]

        # collect all rotations/translation except for the root one
        # B x (N_J - 1) x 3 x 3
        children_rots = torch.cat([rots[:, :root_id], rots[:, root_id+1:]], dim=1)
        # B x (N_J - 1) x 3 x 1
        children_trans = torch.cat([rest_pose[:, :root_id], rest_pose[:, root_id+1:]], dim=1)[..., None]
        parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
        # B x (N_J - 1) x 3 x 1
        parent_trans = rest_pose[:, parent_ids, :, None]
        # B x (N_J - 1) x 3 x 4, matrix |R T|
        bv = children_trans - parent_trans # (N, 15, 3, 1)

        if bone_factor is not None:
            eps = 1e-36
            bone_factor = torch.sqrt(bone_factor**2 + eps)
            if torch.all(bone_factor>0).item() != True:
                import ipdb; ipdb.set_trace()
            assert torch.all(bone_factor>0).item() == True, "bone factor should be positive"
            bv = bv * bone_factor  
            
        # concatenate the rotation and translation of other joints
        joint_rel_transforms = torch.cat([children_rots, bv], dim=-1) # --> [Rotation | BONE vectors]

        '''optimized rotation + bone vectors'''
        # pad to 4 x 4: |R T|
        #               |0 1|
        joint_rel_transforms = pad_mat_to_homogeneous(joint_rel_transforms.view(-1, 3, 4))
        joint_rel_transforms = joint_rel_transforms.view(B, N_J-1, 4, 4) # (N, 15, 4, 4)

        
        '''run kinematic chain here successively, starting with the root rotation 
        at [zero pose|identity matrix| x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) + 3d translation| point]
        against other joints at [zero pose + bone vector]
        '''
        for i, parent in enumerate(parent_ids): # (15)
            l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
        l2ws = torch.stack(l2ws, dim=-3) # (N, 16, 4, 4)
  
        # the 3d keypoints are the translation part of the final
        # per-joint local-to-world matrices
        kp3d = l2ws[..., :3, -1] # (N, 16, 3)

        # rotational part of the per-joint local-to-world matrices
        orient = l2ws[..., :3, :3]

        return kp3d, orient, l2ws, bone_factor
