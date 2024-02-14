# standard libraries
from ipaddress import _IPAddressBase
import torch

#ref loss functions:  https://github.com/zju3dv/EasyMocap/blob/64e0e48d2970b352cfc60ffd95922495083ef306/easymocap/pyfitting/lossfactory.py#L274

def gm(squared_res, sigma_squared):
    out = (sigma_squared * squared_res) / (sigma_squared + squared_res)
    return out.mean()

def l1_loss(res):
    return (torch.abs(res)).mean()

def l2_loss(res):
    return ((res)**2).mean()

def weighted_mse_loss(res, weight):
        return (weight * (res) ** 2).mean()
    
def weighted_l1_loss(res, weight):
        return (weight * torch.abs(res)).mean()

def loc_smooth_loss(kp3d_mm, **k_pipe_kwargs):
    """To smooth the velocity(rate of change)/acceleration of the 3D position"""
    chosen_frames = k_pipe_kwargs["chosen_frames"].to(k_pipe_kwargs["device"])
    frame_diff = chosen_frames[1:] - chosen_frames[:-1] 
    frame_2nd_diff = frame_diff[1:] - frame_diff[:-1] # ideal frame diff is 0
    # -----------------------------------------
    joint_vel = kp3d_mm[1:] - kp3d_mm[:-1]
    joint_acel = joint_vel[1:] - joint_vel[:-1]

    ideal = 0
    mask = torch.ones(len(frame_2nd_diff)).to(k_pipe_kwargs["device"])
    mask[frame_2nd_diff>ideal] = 0

    # l2 norm-> square, sum/mean, sqrt
    joint_acel = (joint_acel).pow(2.).mean(dim=(1,2)) * mask
    sm_loss = joint_acel.sum().pow(0.5)
    return sm_loss

def orient_smooth_loss(orient, **k_pipe_kwargs):
    """To smooth the velocity(rate of change)/acceleration of the bone orientation in 3D"""
    # TODO: Use proper angle diff instead
    chosen_frames = k_pipe_kwargs["chosen_frames"]
    frame_diff = chosen_frames[1:] - chosen_frames[:-1] # ideal frame diff is 1
    frame_2nd_diff = frame_diff[1:] - frame_diff[:-1] # ideal frame diff is 0
    # -----------------------------------------
    orient_vel = orient[1:] - orient[:-1]
    orient_acel = orient_vel[1:] - orient_vel[:-1]

    ideal = 0
    mask = torch.ones(len(frame_2nd_diff)).to(k_pipe_kwargs["device"])
    mask[frame_2nd_diff>ideal] = 0

    orient_acel = (orient_acel).pow(2.).mean(dim=(1,2,3)) * mask
    sm_loss = orient_acel.sum().pow(0.5)
    return sm_loss

def get_feet_loss(r_view_h, v_view_h=None, **k_pipe_kwargs):
    '''raw alphapose, mirror19, comb_set format: projecting feet to the ground - using reconstructed 3D feets'''
    if k_pipe_kwargs["skel_type"] == "alpha":
        rheel, lheel = 25, 24   
        rfoot, lfoot =  rheel, lheel
    elif k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"]:
        rank, lank = 11, 14
        rfoot, lfoot =  rank, lank
    elif k_pipe_kwargs["skel_type"]=="comb_set":
        rheel, lheel = 24, 21
        rfoot, lfoot = rheel, lheel

    """online lowest feet selection - depends on the reconstructed feets"""
    rfeet_select = list(map(lambda x: 0 if x[rfoot][1] > x[lfoot][1] else 1, r_view_h))
    rfeet = torch.stack(list(map(lambda x,y: y[rfoot] if x==0 else y[lfoot], rfeet_select, r_view_h))).to(k_pipe_kwargs["device"])
    vfeet = torch.stack(list(map(lambda x,y: y[lfoot] if x==0 else y[rfoot], rfeet_select, v_view_h))).to(k_pipe_kwargs["device"])

    # mirror position
    mirror_pos = (k_pipe_kwargs["p"] + k_pipe_kwargs["p_dash"])/2
    # vector towards feet of interests
    v_diff_vec = vfeet - mirror_pos
    r_diff_vec = rfeet - mirror_pos
    
    # scalar projection onto ground normal pointing up
    #B = r_diff_vec.shape[0]
    v_dist2gd = torch.bmm(v_diff_vec.view(k_pipe_kwargs["batch_size"],1,3), k_pipe_kwargs["normal_g"].view(k_pipe_kwargs["batch_size"],3,1)).view(k_pipe_kwargs["batch_size"])
    r_dist2gd = torch.bmm(r_diff_vec.view(k_pipe_kwargs["batch_size"],1,3), k_pipe_kwargs["normal_g"].view(k_pipe_kwargs["batch_size"],3,1)).view(k_pipe_kwargs["batch_size"])
    total_feet_loss =  r_dist2gd.mean() + v_dist2gd.mean() 

    # stay positive
    feet_loss = (total_feet_loss-0)**2 # enforce target 0
    return feet_loss

def bone_symmetry_loss(pose2d):
    ''' 
    using constrained bone symmetry on the 2D loss (soft)
    - we have 6 symmetric bones using DCPose skeleton structure
    ref: https://openaccess.thecvf.com/content/ACCV2020/papers/Cao_Anatomy_and_Geometry_Constrained_One-Stage_Framework_for_3D_Human_Pose_ACCV_2020_paper.pdf    
    
    '''

    left_up_arm = (pose2d[:, 5] - pose2d[:, 3])
    right_up_arm = torch.abs(pose2d[:, 6] - pose2d[:, 4])
    
    left_low_arm = (pose2d[:, 7] - pose2d[:, 5])
    right_low_arm = (pose2d[:, 8] - pose2d[:, 6])
    
    left_hip2shoulder = (pose2d[:, 9] - pose2d[:, 3])
    right_hip2shoulder = (pose2d[:, 10] - pose2d[:, 4])
    
    left_knee2hip = (pose2d[:, 11] - pose2d[:, 9])
    right_knee2hip = (pose2d[:, 12] - pose2d[:, 10])
    
    left_foot2knee = (pose2d[:, 13] - pose2d[:, 11])
    right_foot2knee = (pose2d[:, 14] - pose2d[:, 12])
    
    up_arm = l1_loss(left_up_arm - right_up_arm)
    low_arm = l1_loss(left_low_arm - right_low_arm)
    hip2shoulder = l1_loss(left_hip2shoulder - right_hip2shoulder)
    knee2hip = l1_loss(left_knee2hip - right_knee2hip)
    foot2knee = l1_loss(left_foot2knee - right_foot2knee)
    
    # 6 scalar differences
    sym_loss = (shoulder2neck + up_arm + low_arm + hip2shoulder + knee2hip + foot2knee)

    return sym_loss



def get2d_loss(res_real, res_virt, weight_real=None, weight_virt_flipped=None, **k_pipe_kwargs):
    #ref loss functions: https://github.com/zju3dv/EasyMocap/blob/64e0e48d2970b352cfc60ffd95922495083ef306/easymocap/pyfitting/lossfactory.py#L274
    loss_fn = k_pipe_kwargs["loss_fn"]

    if loss_fn == 'l1':
        loss = l1_loss(res_real) + l1_loss(res_virt)
 
    if loss_fn == 'l2':
        loss = l2_loss(res_real) + l2_loss(res_virt)
        
    if loss_fn == 'l2_weight':
        loss = weighted_mse_loss(res_real, weight_real) + weighted_mse_loss(res_virt, weight_virt_flipped)
        
    if loss_fn == 'l1_weight':
        loss = weighted_l1_loss(res_real, weight_real)  + weighted_l1_loss(res_virt, weight_virt_flipped)
        
    if loss_fn == 'Geman':
        sigma_squared = 200
        loss = gm(res_real, sigma_squared) + gm(res_virt, sigma_squared)
        
    return loss