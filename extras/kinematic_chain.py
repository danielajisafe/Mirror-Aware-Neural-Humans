import os,sys

import torch
import torch.nn as nn
import numpy as np
from core_mirror.transforms import flip_h36m
from extras.utils import rot6d_to_rotmat, axisang_to_rot
from extras.skeletons import CMUSkeleton, get_parent_idx, verify_get_parent_idx
import torch.nn.functional as f 
from extras.utils import axisang_to_rot6d


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
            B, N_J, _ = theta.shape 
            
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
            rots = rot6d_to_rotmat(theta.reshape(-1, 6)).reshape(B, N_J, 3, 3) 
        else:
            '''I assume theta is by defaut in axis-angle representation'''
            rots = axisang_to_rot(theta.reshape(-1, 3)).reshape(B, N_J, 3, 3) 

        # l2w: local-to-world transformation.
        # concatenate the rotation and translation of the root joint
        # to get a 3x4 matrix |R T|
        #                     |0 1|

        root_l2w = torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1)
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
            # use optimized bonelength to scale bone vectors
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

        
        '''Run kinematic chain here successively, starting with the root rotation 
        at [zero pose|identity matrix| x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) + 3d translation| point]
        against other joints at [zero pose + bone vector]
        '''
        for i, parent in enumerate(parent_ids): #(15)
            l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
        l2ws = torch.stack(l2ws, dim=-3) # (N, 16, 4, 4)
  
        # the 3d keypoints are the translation part of the final per-joint local-to-world matrices
        kp3d = l2ws[..., :3, -1] # (N, 16, 3)

        # the orientation of the joints are the final rotational part of the per-joint local-to-world matrices
        orient = l2ws[..., :3, :3]

        return kp3d, orient, l2ws, bone_factor


class KinematicChain(nn.Module):

    def __init__(self, rest_pose, skeleton_type=CMUSkeleton, use_rot6d=False, theta_shape=None, where=None):
        """
        rest_pose: float32, (N_joints, 3), rest pose of a skeleton.
        skeleton_type: named tuples, definition of the skeleton.
        use_rot6d: bool, to use 6d rotation instead of 3d axis-angle representation.
                   (see: https://arxiv.org/abs/1812.07035) - continous 6D rotation
                   
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
            B, N_J, _ = theta.shape
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
            rots = rot6d_to_rotmat(theta.view(-1, 6)).view(B, N_J, 3, 3) 
        else:
            '''I assume theta is by defaut in axis-angle representation'''
            rots = axisang_to_rot(theta.view(-1, 3)).view(B, N_J, 3, 3) 

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
            # use optimized bonelength to scale bone vectors
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

        '''Run kinematic chain here successively, starting with the root rotation 
        at [zero pose|identity matrix| x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) + 3d translation| point]
        against other joints at [zero pose + bone vector]
        '''
        for i, parent in enumerate(parent_ids): # (15)
            l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
        l2ws = torch.stack(l2ws, dim=-3) # (N, 16, 4, 4)
  
        kp3d = l2ws[..., :3, -1] # (N, 16, 3)
        orient = l2ws[..., :3, :3]

        return kp3d, orient, l2ws, bone_factor

