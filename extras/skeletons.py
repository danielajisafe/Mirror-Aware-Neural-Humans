import torch
import numpy as np

import plotly.offline as py
import plotly.graph_objs as go
from collections import namedtuple
from  extras.skeleton_utils import get_parent_idx, verify_get_parent_idx

# class definition, and attributes
Skeleton = namedtuple("Skeleton", ["joint_names", "joint_parents", "root_id", "nonroot_id"])


CMUSkeleton = Skeleton(
    joint_names= ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 
        # 4-7
        'right_heel', 'right_big_toe', 'right_small_toe', 'left_hip',
        # 8-11
        'left_knee', 'left_ankle',  'left_heel', 'left_big_toe', 
        # 12-15
        'left_small_toe', 'neck', 'nose', 'right_eye', 
        # 16-19
        'right_ear', 'left_eye', 'left_ear', 'head', 
        # 20-23 
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 
        #24-25
        'right_elbow', 'right_wrist'],

    joint_parents = ['pelvis', 'pelvis', 'right_hip', 'right_knee', 
                    'right_ankle', 'right_heel', 'right_heel', 'pelvis', 
                    'left_hip', 'left_knee', 'left_ankle', 'left_heel', 
                    'left_heel', 'pelvis', 'neck', 'nose', 
                    'right_eye', 'nose', 'left_eye', 'nose', 
                    'neck','left_shoulder', 'left_elbow', 'neck',
                    'right_shoulder', 'right_elbow'],

    root_id=0, # root joint is at index 0, which makes it easier to implement
    nonroot_id=[i for i in range(26) if i != 0]
    )


def plot_skeleton3d(skel, fig=None, skel_id=None, skel_type=CMUSkeleton,
                    cam_loc=None, layout_kwargs=None, line_type=None, color = None, line_width=4,
                    marker_size=3, hoverlabel = None, Ext_trans = None, skel_name = None,
                    view=None, axes=None, low_lim=-7000, up_lim=7000, joint_names=None,
                    joint_parents=None, init_type=None):
    """
    Plotting function for canonicalized skeleton
    """

    # x, y, z of lines in the center/left/right body part
    clx, cly, clz = [], [], []
    llx, lly, llz = [], [], []
    rlx, rly, rlz = [], [], []
    # for individual joints 
    d_clx, d_cly, d_clz = [], [], []
    d_llx, d_lly, d_llz = [], [], []
    d_rlx, d_rly, d_rlz = [], [], []
    # orientation
    red_x, red_y, red_z = [], [], []
    green_x, green_y, green_z = [], [], []
    blue_x, blue_y, blue_z = [], [], []
    
    # -----------------------------------------------------------------------------
    x, y, z = list(skel[..., 0]), list(skel[..., 1]), list(skel[..., 2])
    if axes is not None:
        x_vec, y_vec, z_vec = axes[0].tolist(), axes[1].tolist(), axes[2].tolist()

    if init_type == "mirror19":
        joint_names= ['pelvis', 'right_hip', 'right_knee', 'right_ankle',  'left_hip',
                    'left_knee', 'left_ankle', 'neck', 'nose', 'right_eye', 
                    'right_ear', 'left_eye', 'left_ear', 
                    'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 
                    'right_elbow', 'right_wrist']
        joint_parents = ['pelvis', 'pelvis', 'right_hip', 'right_knee',  'pelvis', 
                    'left_hip', 'left_knee', 'pelvis', 'neck', 'nose', 
                    'right_eye', 'nose', 'left_eye', 
                    'neck','left_shoulder', 'left_elbow', 'neck',
                    'right_shoulder', 'right_elbow']

        idx = get_parent_idx(joint_names, joint_parents)
        verify_get_parent_idx(joint_names, joint_parents)
        joint_tree = np.array(idx)                    

    elif joint_names is not None and joint_parents is not None:
        idx = get_parent_idx(joint_names, joint_parents)
        verify_get_parent_idx(joint_names, joint_parents)
        joint_tree = np.array(idx)

    else:
        #use default alphapose init
        idx = get_parent_idx(skel_type.joint_names, skel_type.joint_parents)
        verify_get_parent_idx(skel_type.joint_names, skel_type.joint_parents)
        
        joint_tree = np.array(idx)
        joint_names = skel_type.joint_names
  
    
    root_id = skel_type.root_id

    l_joint_names = []
    r_joint_names = []
    c_joint_names = []
    
    # for bones
    for i, (j, name) in enumerate(zip(joint_tree, joint_names)):
        if "left" in name:
            lx, ly, lz = llx, lly, llz
        elif "right" in name:
            lx, ly, lz = rlx, rly, rlz
        else:
            lx, ly, lz = clx, cly, clz

        if "left_heel" in name:
            dlx, dly, dlz = d_llx, d_lly, d_llz
            l_joint_names.append(name)
        elif "right_heel" in name:
            dlx, dly, dlz = d_rlx, d_rly, d_rlz
            r_joint_names.append(name)
        else:
            dlx, dly, dlz = d_clx, d_cly, d_clz
            c_joint_names.append(name)

        lx += [x[i], x[j], None]
        ly += [y[i], y[j], None]
        lz += [z[i], z[j], None]

        dlx += [x[i]]
        dly += [y[i]]
        dlz += [z[i]]

    joint_names = [f"{i}_{name}" for i, name in enumerate(joint_names)]
    if skel_id is not None:
        joint_names = [f"{skel_id}_{name}" for name in joint_names]

    # Plot Points
    if skel_name == "alpha":
        points_center = go.Scatter3d(x=d_clx, y=d_cly, z=d_clz, mode="markers", marker=dict(size=marker_size),
                            line=dict(color="teal"),   
                            text=c_joint_names, opacity=1.0)

        points_left = go.Scatter3d(x=d_llx, y=d_lly, z=d_llz, mode="markers", marker=dict(size=marker_size),
                            line=dict(color="red"),   
                            text=l_joint_names, opacity=1.0)

        points_right = go.Scatter3d(x=d_rlx, y=d_rly, z=d_rlz, mode="markers", marker=dict(size=marker_size),
                            line=dict(color="red"),   
                            text=r_joint_names, opacity=1.0)

    else:
        points = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=marker_size),
                          line=dict(color="blue"),   
                          text=joint_names, opacity=1.0)
    # ----------------------------------------------------------------------- 
    # Plot lines
    color = "teal"; line_type = "solid"
    center_lines = go.Scatter3d(x=clx, y=cly, z=clz, mode="lines",
                                text = [hoverlabel],
                                line=dict(dash = line_type, color= color, width=line_width),
                                hoverinfo="all", opacity=1.0)
    color = "green"
    left_lines = go.Scatter3d(x=llx, y=lly, z=llz, mode="lines",
                              text = [hoverlabel],
                              line=dict(dash = line_type, color= color, width=line_width),
                              hoverinfo="all", opacity=1.0)
    color = "red"
    right_lines = go.Scatter3d(x=rlx, y=rly, z=rlz, mode="lines",
                               text = [hoverlabel],
                               line=dict(dash = line_type, color= color, width=line_width),
                               hoverinfo= "all", opacity=1.0)
    
    
    if skel_name == "alpha":
        data = [points_left, points_right, points_center, center_lines, left_lines, right_lines]
    else:
        data = [points, center_lines, left_lines, right_lines]

    if view == -1:
        eye_x,eye_y,eye_z=1.0,0.0,1.0 #side view
        cam_dict = dict(up=dict(x=0, y=-1, z=0), #stand straight
                eye=dict(x=eye_x, y=eye_y, z=eye_z), 
                )
    elif view == 0:
        eye_x,eye_y,eye_z=0.1,0.1,0.3 # front view
        cam_dict = dict(up=dict(x=0, y=-1, z=0), #stand straight
                eye=dict(x=eye_x, y=eye_y, z=eye_z), 
                )
    else: 
        cam_dict = dict(up=dict(x=0, y=1, z=0))

    # For equal scale along all axes
    layout = go.Layout(
    width=1024,
    height=1024, 
    scene = dict(
    camera = cam_dict,
        
        # define boundaries for hip-centered pose
        # not adding boundaries to the plot skews the visualization'''
        # If ratio is not the same for x,y,z, depth might be lost visually'''
        xaxis = dict(title='x axis', range = [low_lim, up_lim]), 
        yaxis = dict(title='y axis', range = [low_lim, up_lim]),
        zaxis = dict(title='z axis', range = [low_lim, up_lim]),
        aspectratio=dict(x=1, y=1, z=0.95),
            ))
    
    if fig is None: 
        fig = go.Figure(data=data, layout=layout)
    else:
        print("extra fig")
        for d in data:
            fig.add_trace(d)

    return fig
