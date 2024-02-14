#custom import 
import io
from turtle import ycor
import cv2
import math
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from collections import deque
from copy import copy, deepcopy
from collections import namedtuple
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
import matplotlib.pyplot as plt

from numpy import pi, cos, sin, sqrt

def get_bone_names(joint_names, joint_parents):
    bone_names = []
    for frm,to in zip(joint_parents, joint_names):
        if frm != to: #skip root-root
            bone_names.append((frm,to))
    return bone_names

def get_parent_idx(joint_names, joint_parents):
    return np.array([joint_names.index(i) for i in joint_parents])

def verify_get_parent_idx(joint_names, joint_parents):
    """ Unit tests to check parent joint consistency
    args:
        skel_type: skeleton class"""

    out = all([joint_names[p_id] == joint_parents[i] for i, p_id in enumerate(get_parent_idx(joint_names, joint_parents))])
    assert out == True, "get_parent_idx values are incorrect"

# ref: https://github.com/LemonATsu/A-NeRF/blob/eb16fe7e38a9f1688a7e785a286ad70c5f31d66f/core/utils/skeleton_utils.py

def rotate_x(phi):
    cos = np.cos(phi)
    sin = np.sin(phi)
    return np.array([[1,   0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0,   0,    0, 1]], dtype=np.float32)

def rotate_z(psi):
    cos = np.cos(psi)
    sin = np.sin(psi)
    return np.array([[cos, -sin, 0, 0],
                     [sin,  cos, 0, 0],
                     [0,      0, 1, 0],
                     [0,      0, 0, 1]], dtype=np.float32)
def rotate_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,   0, -sin, 0],
                     [0,     1,    0, 0],
                     [sin,   0,  cos, 0],
                     [0,   0,      0, 1]], dtype=np.float32)

def generate_bullet_time_kp(kp, rot_axis="y", n_views=20, max_deg=360):
    kp = kp - kp[:1, :]
    y_angles = np.linspace(0, math.radians(max_deg), n_views+1)[:-1]
    kps = []
    for a in y_angles:
        k = kp @ rotate_y(a)[:3, :3]
        kps.append(k)
    return np.array(kps)

def generate_bullet_time_bone_axes(l2ws, rot_axis="y", n_views=20, max_deg=360):
    y_angles = np.linspace(0, math.radians(max_deg), n_views+1)[:-1]
    l2ws_rotate = []
    for a in y_angles:
        # composite transformation
        l = l2ws @ rotate_y(a)
        l2ws_rotate.append(l)
    return np.array(l2ws_rotate)

def generate_bullet_time(c2w, n_views=20, axis='y', max_deg=360):
    if axis == 'y':
        rotate_fn = rotate_y
    elif axis == 'x':
        rotate_fn = rotate_x
    elif axis == 'z':
        rotate_fn = rotate_z
    else:
        raise NotImplementedError(f'rotate axis {axis} is not defined')

    y_angles = np.linspace(0, math.radians(max_deg), n_views+1)[:-1]
    c2ws = []
    for a in y_angles:
        c = rotate_fn(a) @ c2w
        c2ws.append(c)
    return np.array(c2ws)


# ref: https://chart-studio.plotly.com/~empet/15666/roll-pitch-and-yaw-motion-of-an-airplan/#/
def rotx(t):
    return np.array([[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]])

def roty(t):
    return np.array([[cos(t), 0, sin(t)],  [0, 1, 0],[-sin(t), 0, cos(t)]])   
    
def rotz(t):
    return np.array([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])