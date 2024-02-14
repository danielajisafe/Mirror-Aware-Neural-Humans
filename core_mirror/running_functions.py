import sys
from tracemalloc import stop
import torch, json

# custom imports
from core_mirror.transforms import rotate_initial_pose
import core_mirror.util_skel as skel
from core_mirror.util_loading import alpha_to_hip_1st, hip_1st_to_alpha, hip_1st_to_mirror19, hip_1st_to_comset26, normalize_batch_normal, build_A_dash

from extras.kinematic_chain import KinematicChain
from core_mirror.transforms import baseline_to_h36m_uniform_3D, baseline_to_h36m_uniform_2D
from core_mirror.loss_factory import gm, l1_loss,l2_loss, weighted_mse_loss, weighted_l1_loss, bone_symmetry_loss, get2d_loss,\
                         loc_smooth_loss, orient_smooth_loss, get_feet_loss
from core_mirror.transforms import project_pose_batch, refine_g_normal, h36m_to_DCPose, flip_dcpose, flip
from extras.plotting import plotPoseOnImage, plot_multiple_views, plot15j_2d, plot15j_3d, plot15j_2d_no_image, add_bbox_in_image, plot15j_2d_uniform


@torch.no_grad()
def get_gradnorm(module):
    """ref: https://github.com/LemonATsu/A-NeRF/blob/eb16fe7e38a9f1688a7e785a286ad70c5f31d66f/core/trainer.py#L192"""
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

def correct_virt(flipvirt2d_fn, threshold, where_virt, weight2, proj2d_virt, rsh, lsh, REar, Nose, 
                 **k_pipe_kwargs):
    
    '''Flip virt reprojection based on estimate/detection position'''
    where_virt_proj_shoulder = (proj2d_virt[:,rsh,0] > proj2d_virt[:,lsh,0]).tolist();
    where_virt_proj_Ear_Nose = (proj2d_virt[:,REar,0] > proj2d_virt[:,Nose,0]).tolist()
    where_virt_proj = list(map(lambda diff,y,z: y if diff>threshold else z, torch.norm(proj2d_virt[:,rsh] - proj2d_virt[:,lsh], dim=-1).tolist(), where_virt_proj_shoulder, where_virt_proj_Ear_Nose))

    # recon now in some common format
    if k_pipe_kwargs["skel_type"] == "alpha":
        # alpha flip need use_mapper
        pose2d_out = list(map(lambda x,y,z: flipvirt2d_fn(z.unsqueeze(0),use_mapper=k_pipe_kwargs["use_mapper"]).squeeze(0) if x!=y else z, where_virt, where_virt_proj, proj2d_virt))
        w_out = list(map(lambda x,y,z: flipvirt2d_fn(z.unsqueeze(0),use_mapper=k_pipe_kwargs["use_mapper"]).squeeze(0) if x!=y else z, where_virt, where_virt_proj, weight2))
        
    else:
        # other flip expects common 
        pose2d_out = list(map(lambda x,y,z: flipvirt2d_fn(z.unsqueeze(0)).squeeze(0) if x!=y else z, where_virt, where_virt_proj, proj2d_virt))
        w_out = list(map(lambda x,y,z: flipvirt2d_fn(z.unsqueeze(0)).squeeze(0) if x!=y else z, where_virt, where_virt_proj, weight2))

    flipped_frames = list(map(lambda x,y: 1 if x!=y else 0, where_virt, where_virt_proj))
    proj2d_virt_up = torch.stack(pose2d_out)
    weight2 = torch.stack(w_out)

    return proj2d_virt_up, weight2, where_virt_proj, flipped_frames


def get2d(m_i, kc, hips, fewer,theta, eps, **k_pipe_kwargs):
    '''This function contains the 3d to 2d pipeline'''

    loss_dict = {}
    bool_params = k_pipe_kwargs["bool_params"]
    bf_unknown = k_pipe_kwargs["bf_unknown"]

    '''Building A_dash for fewer params'''
    if fewer or bool_params["n_m_single"]:
        (plane_d) = k_pipe_kwargs["A_dash_tuple"]
   
    if bool_params["bf_build"] and bool_params["bf_op"]: 
        
        '''build bone vectors'''
        # we need an initial skeleton of 26 joints.
        if k_pipe_kwargs["skel_type"] == "alpha": # 25 bones -> 14 symmetric bone factors

            # based on "hip-first" alphapose26 init | head-> nose, nose-> neck
            down_sym = bf_unknown[:, :6]; 
            face_sym = bf_unknown[:, 6:8] 
            up_sym = bf_unknown[:, 8:11] 
            others2 = bf_unknown[:, 11:13] 
            others1 = bf_unknown[:, 13:14] 
            
            bf_build = torch.cat([down_sym, down_sym, others2, face_sym,
                                    face_sym, others1, up_sym, up_sym], dim=1)


        elif k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"]: #18 bones-> 10 symmetric bone factors
            
            # based on "hip-first" mirror19 init
            down_sym = bf_unknown[:, :3]; 
            face_sym = bf_unknown[:, 3:5] 
            up_sym = bf_unknown[:, 5:8] 
            others1 = bf_unknown[:, 8:] 
            
            bf_build = torch.cat([down_sym, down_sym, others1, face_sym,
                                    face_sym, up_sym, up_sym], dim=1)

        elif k_pipe_kwargs["skel_type"]=="comb_set": #26 joints | 25 bones-> 14 symmetric bone factors

            # based on "hip-first" combset init | # head-> neck, nose-> neck
            down_sym = bf_unknown[:, :6]
            face_sym = bf_unknown[:, 6:8]
            up_sym = bf_unknown[:, 8:11]
            others2 = bf_unknown[:, 11:13] 
            others1 = bf_unknown[:, 13:14] 
            
            bf_build = torch.cat([down_sym, down_sym, others2, 
                                  face_sym, face_sym, others1,
                                  up_sym, up_sym 
                                  ], dim=1)
            
        else: # original H36m 17 pts at this point
            # 16 bones -> 10
            down_sym = bf_unknown[:, :3]; 
            up_sym = bf_unknown[:, 3:6] 
            others4 = bf_unknown[:, 6:] # 4 others

            bf_build = torch.cat([down_sym, down_sym, others4,
                                    up_sym, up_sym], dim=1); 
            import ipdb; ipdb.set_trace()

        bf_build = bf_build.to(k_pipe_kwargs["device"])
        kp3d_mm, orient, l2ws, bf_positive  = kc(theta, bone_factor=bf_build, **k_pipe_kwargs); 
        bf_out = bf_build
    else:
        bf_unknown = bf_unknown.to(k_pipe_kwargs["device"])
        kp3d_mm, orient, l2ws, bf_positive = kc(theta, bone_factor=bf_unknown, **k_pipe_kwargs);
        bf_out = bf_unknown
    
    """back to default ordering"""
    if k_pipe_kwargs["skel_type"] == "alpha":
        kp3d_mm = hip_1st_to_alpha(kp3d_mm, **k_pipe_kwargs)
    
    elif k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"]:
        kp3d_mm = hip_1st_to_mirror19(kp3d_mm, hip_1st_names=k_pipe_kwargs["joint_names"], \
        mirror_names=skel.joint_names_mirrored_human_19, **k_pipe_kwargs)

    elif k_pipe_kwargs["skel_type"]=="comb_set":
        kp3d_mm = hip_1st_to_comset26(kp3d_mm, hip_1st_names=k_pipe_kwargs["joint_names"], \
        comb_set_names=skel.joint_names_combset, **k_pipe_kwargs)
    
    '''Add temporal constraint on translation(location) and orientation'''
    loc_sm_loss = loc_smooth_loss(kp3d_mm, **k_pipe_kwargs)
    orient_sm_loss = orient_smooth_loss(orient, **k_pipe_kwargs)

    loss_dict["loc_sm_loss"] = loc_sm_loss
    loss_dict["orient_sm_loss"] = orient_sm_loss
    
    kp3d_m = (kp3d_mm/1000)

    '''Now we need to put our entire pipeline in (m), hence hip should now be in meters range'''
    kp3d_h = (kp3d_m + hips)
    
    # save and return quantities used
    if fewer:
        '''normalize avg_normal_m explicitly instead of using a normal loss''' 
        avg_normal_m = normalize_batch_normal(k_pipe_kwargs["avg_normal_m"])
        A_dash = build_A_dash(avg_normal_m, plane_d, k_pipe_kwargs["device"])
        
        ''' update n_m to what is being optimized'''
        n_m = avg_normal_m.repeat(k_pipe_kwargs["batch_size"], 1)
        # use on-the-fly calculation since you disable some gradients in Adash
        A_dash_dups = A_dash.repeat(k_pipe_kwargs["batch_size"], 1,1) # (1,4,4) -> (B,4,4)

    elif not fewer:
        if bool_params["n_m_single"]:
            '''optimizing n_m alone without plane_d'''
            avg_normal_m = normalize_batch_normal(k_pipe_kwargs["avg_normal_m"])
            A_dash = build_A_dash(avg_normal_m, plane_d, k_pipe_kwargs["device"])
            
            n_m = avg_normal_m.repeat(k_pipe_kwargs["batch_size"], 1)
            A_dash_dups = A_dash.repeat(k_pipe_kwargs["batch_size"], 1,1) # (1,4,4) -> (B,4,4)
        
        elif bool_params["A_dash"]:
            '''optimizing whole A_dash'''
            A_dash_dups = k_pipe_kwargs["A_dash"].repeat(k_pipe_kwargs["batch_size"], 1,1) # (1,4,4) -> (B,4,4)
            # only for later usage
            n_m = k_pipe_kwargs["avg_normal_m"]
        else:
            print("something not implemented.")
            import ipdb; ipdb.set_trace()
            
    K_dups = k_pipe_kwargs["K"].unsqueeze(0).repeat(k_pipe_kwargs["batch_size"], 1,1); 

    ''' canonical pose is r_view since its base camera'''
    proj2d_real_h36m, r_view_h, final_A = project_pose_batch(pose3d=kp3d_h, cord_transform=k_pipe_kwargs["A"], cam=K_dups, **k_pipe_kwargs); 
    '''using the same k matrix'''
    proj2d_virt_h36m, v_view_h, final_A_dash = project_pose_batch(pose3d=kp3d_h, cord_transform=A_dash_dups, cam=K_dups, **k_pipe_kwargs); 

    '''scalar projecting feet, to get distance to the ground'''
    feet_loss = get_feet_loss(r_view_h, v_view_h,**k_pipe_kwargs)
    loss_dict["feet_loss"] = feet_loss


    if k_pipe_kwargs["use_mapper"]:
        '''from INIT (h36m|h36m->alpha) to a common skeleton
        - needs to align here with order in data loader for loss computation'''
        if k_pipe_kwargs["skel_type"] == "dcpose":
            init_to_common_map_fn = skel.h36m_to_dc_common_fn

        elif k_pipe_kwargs["skel_type"] == "alpha":
            # init: alphapose structure (h36m extended); detection: alphapose
            init_to_common_map_fn = skel.alphapose_to_mirror_common_fn
            proj2d_real = init_to_common_map_fn(proj2d_real_h36m)
            proj2d_virt = init_to_common_map_fn(proj2d_virt_h36m)

        elif k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"]:
            # init: m19 init structure vs m19 GT2D (raw same as common map)
            proj2d_real = proj2d_real_h36m
            proj2d_virt = proj2d_virt_h36m

        elif k_pipe_kwargs["skel_type"]=="comb_set": # (raw same as common map)
            proj2d_real = proj2d_real_h36m
            proj2d_virt = proj2d_virt_h36m
     
    else:
        pass

    return proj2d_real, proj2d_virt, kp3d_m, kp3d_h, K_dups, r_view_h, v_view_h, n_m, plane_d, bf_out, orient, l2ws, bf_positive, loss_dict, final_A, final_A_dash
