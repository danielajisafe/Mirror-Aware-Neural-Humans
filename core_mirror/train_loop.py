import os,sys
import core_mirror.util_skel as skel
from matplotlib.style import use

# standard libraries
import cv2
import wandb
import uuid
import datetime
import torch, json
import numpy as np

from IPython import display
from tqdm import tqdm, trange
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# custom imports
from extras.compute_eval import batch_evaluate
from core_mirror.util_loading import calc_bone_length
from extras.utils import rot6d_to_rotmat, constrain_rot
from core_mirror.running_functions import get2d, correct_virt, get_gradnorm
from core_mirror.util_loading import alpha_to_hip_1st, hip_1st_to_alpha, save2pickle, get_joint_trees, mirror19_to_hip_1st, comset26_to_hip_1st

from extras.kinematic_chain import KinematicChain
from extras.skeleton_utils import get_bone_names, get_parent_idx, verify_get_parent_idx
from core_mirror.transforms import rotate_initial_pose, baseline_to_h36m_uniform_3D, baseline_to_h36m_uniform_2D
from core_mirror.loss_factory import gm, l1_loss,l2_loss, weighted_mse_loss, weighted_l1_loss, bone_symmetry_loss, get2d_loss, orient_smooth_loss, loc_smooth_loss
from core_mirror.transforms import project_pose_batch, refine_g_normal, h36m_to_DCPose, flip_dcpose, flip, flip_mirrorpose, flip_alphapose, flip_mirror19, flip_combset26
from extras.plotting import plotPoseOnImage, plot_multiple_views, plot15j_2d, plot15j_3d, plot15j_2d_no_image, add_bbox_in_image, plot15j_2d_uniform, plot2d_halpe26_mirror_common_2D


def new_kinematic_pipe(print_every=1000, test= None,**k_pipe_kwargs):
    """Everything here in a common format but, 
    Note: Alphapose - 26 for volumetric model, 25 only for evaluation purposes
          Mirror19 - 19 for optimization, 19 for evaluation purposes"""

    project_dir = k_pipe_kwargs["args"].project_dir 
    uniq_id = uuid.uuid4()

    device = k_pipe_kwargs["device"]
    view = k_pipe_kwargs["view"]
    N_J = k_pipe_kwargs["num_joints"]
    joint_names = k_pipe_kwargs["joint_names"]
    joint_parents = k_pipe_kwargs["joint_parents"]

    video_data = iter(k_pipe_kwargs["video_loader"])
    kp3d_m_house, kp3d_h_house, proj2d_real_house, proj2d_virt_up_house = [], [], [], []
    img_url_house, rot_mat_house, theta_house = [], [], []
    p_dash_2d_house, p_2d_house, N_2d_house, normal_end_2d_house = [],[],[],[]
    ground_end_2d_house, refine_ground_end_2d_house, otho_end_2d_house = [],[],[]
    l2ws_house = []
    flipped_frames = None


    if k_pipe_kwargs["skel_type"] == "dcpose":
        rsh, lsh = 4, 3
        rhip, lhip = rsh, lsh # 10, 9
        flipvirt2d_fn = flip_dcpose

    elif k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"] and k_pipe_kwargs["use_mapper"]:
        # when in mirror common 19
        hip = 8
        rsh, lsh = 2, 5
        rhip, lhip = 9, 12
        REar, Nose = 17, 0 
        flipvirt2d_fn = flip_mirror19

    elif k_pipe_kwargs["skel_type"] == "alpha":

        if k_pipe_kwargs["use_mapper"]:
            # when in mirror common 25
            hip = 8
            Rank = 11
            REar, Nose = 17, 0 
            rsh, lsh = 2, 5
            rhip, lhip = 9, 12

        # captures both options
        flipvirt2d_fn = flip_alphapose

    elif k_pipe_kwargs["skel_type"] == "comb_set"and k_pipe_kwargs["use_mapper"]:
        # when in combset, can be be seen as common too (as subset of a union of two skeleton joints)
        hip = 8
        rsh, lsh = 2, 5
        rhip, lhip = 9, 12
        REar, Nose = 17, 0 
        flipvirt2d_fn = flip_combset26

    # whole batch is used, hence single data iter
    for m_i in range(k_pipe_kwargs["n_mini_batch"]):

        data = next(video_data)
        theta_in = k_pipe_kwargs["theta_house"][m_i]
        hips = k_pipe_kwargs["hips_house"][m_i]
        image_urls  = data["image_urls"]

        data["p"], data["p_dash"] = data["p"].to(k_pipe_kwargs["device"]), data["p_dash"].to(k_pipe_kwargs["device"])
        p, p_dash = data["p"], data["p_dash"]

        if k_pipe_kwargs["use_mapper"]:
            # mirror19-GT2d or mirror25-alphapose
            est_2d_real, est_2d_virt  = data["real2d"].to(k_pipe_kwargs["device"]), data['virt2d'].to(k_pipe_kwargs["device"])
        else:
            est_2d_real, est_2d_virt = data["est_2d_real"].to(k_pipe_kwargs["device"]), data["est_2d_virt"].to(k_pipe_kwargs["device"])
        
        weight1, weight2 = data["c_score1"].to(k_pipe_kwargs["device"]), data["c_score2"].to(k_pipe_kwargs["device"])
        k_pipe_kwargs["chosen_frames"] = data["chosen_frames"]
        k_pipe_kwargs["p"], k_pipe_kwargs["p_dash"] = data["p"], data["p_dash"]

        # STARTING FROM HERE
        losses, store_real, store_virt = [], [], []

        # is real person backing the camera?
        where_real_shoulder = (est_2d_real[:,rsh,0] > est_2d_real[:,lsh,0]).tolist() 
        where_virt_shoulder = (est_2d_virt[:,rsh,0] > est_2d_virt[:,lsh,0]).tolist()  

        where_real_Ear_Nose = (est_2d_real[:,REar,0] > est_2d_real[:,Nose,0]).tolist() 
        where_virt_Ear_Nose = (est_2d_virt[:,REar,0] > est_2d_virt[:,Nose,0]).tolist()

        threshold = 0.
        where_real = list(map(lambda diff,y,z: y if diff>threshold else z, (est_2d_real[:,rsh,0] - est_2d_real[:,lsh,0]).tolist(), where_real_shoulder, where_real_Ear_Nose))#.tolist()  
        where_virt = list(map(lambda diff,y,z: y if diff>threshold else z, (est_2d_virt[:,rsh,0] - est_2d_virt[:,lsh,0]).tolist(), where_virt_shoulder, where_virt_Ear_Nose))#.tolist()  

        real_prop = round(1 - sum(where_real)/est_2d_real.shape[0], 2)
        print(f"% of issues in real m-batch - {real_prop} of {est_2d_real.shape[0]} frames")
        
        if k_pipe_kwargs["batch_size"] == 1:
            plot_id = k_pipe_kwargs["show_image"][0]
            image = mpimg.imread(image_urls[plot_id])
        else:
            plot_id = k_pipe_kwargs["show_image"][0]
            image = mpimg.imread(image_urls[k_pipe_kwargs["show_image"][0]])#; 
        
        B = k_pipe_kwargs["batch_size"]
        img_height, img_width,_ = image.shape
    
        # avoid division by zero
        eps = 1e-32 
        fewer = k_pipe_kwargs["bool_params"]["A_dash_fewer"]; 

        # kinematic chain
        kc = KinematicChain(k_pipe_kwargs["initial_pose3d"], use_rot6d=True, theta_shape = theta_in.shape)

        # Per-Frame Initial Global Orientation
        # ----------------------------------------------------------
        # Finding the optimal initial orientation Per Frame
        # ----------------------------------------------------------
        # Start of Search 
        # initials
        degree_err = torch.tensor([float("inf")]).repeat(theta_in.shape[0])
        degree_store = torch.tensor([0]).repeat(theta_in.shape[0]).tolist()

        for degree in [0,45,90,135,180,225,270,315]: 
            degree_batch = torch.tensor([degree]).repeat(theta_in.shape[0]).tolist()

            all = [False] * theta_in.shape[0] # False here means make a change
            # perturb initial position for a video/camera setup and continue everything as before
            theta_perturb = rotate_initial_pose(theta_in, all, degree=degree_batch, **k_pipe_kwargs) 
            # -------------------------------------------

            if k_pipe_kwargs["rotate_initial_flag"]:
                null = [None] * theta_in.shape[0]
                theta = rotate_initial_pose(theta_in, where_real, degree=null, **k_pipe_kwargs)   
            else:
                theta = theta_perturb

            """rehearsal projections"""
            # ---------------------------------------------
            proj2d_real, proj2d_virt, kp3d_m, kp3d_h, K_dups, r_view, v_view, n_m, plane_d, bf_build, orient, l2ws, bf_positive, loss_dict, final_A, final_A_dash = get2d(m_i, kc, hips, fewer,theta, eps, **k_pipe_kwargs)

            if k_pipe_kwargs["flip_virt_flag"]:
                '''Flip virt reprojection based on estimate positioning'''
                proj2d_virt_up, weight2, where_virt_proj, flipped_frames= correct_virt(flipvirt2d_fn, threshold,\
                                                    where_virt, weight2, proj2d_virt, rsh, lsh, REar, Nose, **k_pipe_kwargs)
 
            else:
                proj2d_virt_up = proj2d_virt
            #----------------------------------------------
            
            # now in common format
            # measure bone vectors (because they have directions than simply using locations)
            proj_bone_vectors= (proj2d_real[:, [rsh, rhip, REar], :]- proj2d_real[:, [lsh, lhip, Nose], :])
            est_bone_vectors = (est_2d_real[:, [rsh, rhip, REar], :] - est_2d_real[:, [lsh, lhip, Nose], :]) 
            
            resid_real = proj_bone_vectors - est_bone_vectors 
            frames_loss_real = torch.abs(resid_real) 

            # use only the error from more confident bone vectors (i.e > threshold)
            bv_conf = weight1[:, [rsh, rhip, REar], :] 
            assert B == bv_conf.shape[0], "batch size is not the same"

            def get_idx_above_thresd(tensor, thresh):
                """get indices of values greater than threshold
                ref: https://stackoverflow.com/a/50046640/12761745"""
                n = len(tensor)
                mask = torch.ones(n).to(device)
                inds = torch.nonzero((tensor >= thresh)*mask)
                return inds

            conf_threshold = 0.5
            f_loss_select  = list(map(lambda x,y: x[get_idx_above_thresd(y, conf_threshold)].mean(), frames_loss_real, bv_conf.view(B,-1)))
            frames_loss = torch.Tensor(f_loss_select)
            err_bool = frames_loss.cpu() < degree_err

            degree_err  = list(map(lambda x,y,z: y if x == True else z, err_bool, frames_loss.tolist(), degree_err.tolist()))
            degree_err = torch.Tensor(degree_err)
            degree_store = list(map(lambda x,y,z: y if x == True else z, err_bool, degree_batch, degree_store))

        print(f"optimal initial batch orientation- first 5: {degree_store[:5]}")
        # --------------------------------------------

        # MAIN PIPELINE
        '''Optimization Loop'''

        for iter_ in trange(k_pipe_kwargs["iterations"]):

            k_pipe_kwargs["optimizer_house"][m_i].zero_grad()

            # Per-Frame Correction
            # ---------------------------------------------
            all = [False] * theta_in.shape[0] 
            # perturb initial position for a video/camera setup and continue everything as before
            theta_perturb = rotate_initial_pose(theta_in, all, degree=degree_store, **k_pipe_kwargs) 
            # ---------------------------------------------

            if k_pipe_kwargs["rotate_initial_flag"]:
                null = [None] * theta_in.shape[0]
                theta = rotate_initial_pose(theta_in, where_real, degree=null, **k_pipe_kwargs)   
            else:
                theta = theta_perturb
            # ---------------------------------------------
            # is optimized rotation still orthogonal? ensure it to be
            rot_mat = rot6d_to_rotmat(theta.view(-1,6)).view(theta.shape[0], N_J,3,3)

            intend_identity = np.round(constrain_rot(rot_mat).detach().cpu().numpy())
            if np.all(intend_identity != torch.eye(3).view(1,1,3,3).repeat(theta.shape[0], N_J,1,1).detach().cpu().numpy()):
                print("rotation is optimizing but likely going beyond rotation to do also scaling")
                import ipdb; ipdb.set_trace()
            # -----------------------------------------
            # now in common format
            proj2d_real, proj2d_virt, kp3d_m, kp3d_h, K_dups, r_view_h, v_view_h, n_m, plane_d, bf_build, orient, l2ws, bf_positive, loss_dict, final_A, final_A_dash = get2d(m_i, kc,hips,fewer,theta, eps, **k_pipe_kwargs)
            
            if k_pipe_kwargs["flip_virt_flag"]:
                '''Flip virt reprojection based on estimate positioning
                    - Re-use first where_virt_proj '''
                
                proj2d_virt_up, weight2, where_virt_proj, flipped_frames = correct_virt(flipvirt2d_fn, threshold,\
                                                where_virt, weight2, proj2d_virt, rsh, lsh, REar, Nose, **k_pipe_kwargs)
                # any errors of misalignment?
                if iter_ == (k_pipe_kwargs["iterations"] - 1):
                    virt_prop = round(sum((torch.tensor(where_virt)!=torch.tensor(where_virt_proj)).tolist())/est_2d_virt.shape[0], 2)
                    print(f"Last Iter: % requiring standard virt flip - {virt_prop} of {est_2d_virt.shape[0]} frames")

            else:
                proj2d_virt_up = proj2d_virt
            
            # End of Per-Frame Correction
            #----------------------------------------------------------------------
            # real person now mostly aligned with virt estimate
            
            if iter_ == (k_pipe_kwargs["iterations"] - 1):
                where_virt_proj_new = (proj2d_virt_up[:,rsh,0] > proj2d_virt_up[:,lsh,0]) 
                fixed_virt_prop = round(sum((torch.tensor(where_virt).to(k_pipe_kwargs["device"]) !=where_virt_proj_new).tolist())/est_2d_virt.shape[0], 2)
                print(f"Last Iter: % of standard virt flip left - {fixed_virt_prop}")

            res1 = proj2d_real - est_2d_real
            res2 = proj2d_virt_up - est_2d_virt; 

            pose_loss = get2d_loss(res1, res2, weight1, weight2, **k_pipe_kwargs)

            #-----------------------------------------------------
            if k_pipe_kwargs["args"].loc_smooth_loss:
                loc_sm_scale = k_pipe_kwargs["loc_smooth_scale"]
                loc_sm_loss = loss_dict["loc_sm_loss"] 
            else:
                loc_sm_scale, loc_sm_loss = 0., 0.
    
            if k_pipe_kwargs["args"].orient_smooth_loss:
                orient_sm_scale = k_pipe_kwargs["orient_smooth_scale"]
                orient_sm_loss = loss_dict["orient_sm_loss"] 
            else:
               orient_sm_scale, orient_sm_loss = 0., 0.

            if k_pipe_kwargs["args"].feet_loss:
                feet_loss_scale = k_pipe_kwargs["feet_loss_scale"]
                f_loss = loss_dict["feet_loss"] 
            else:
               feet_loss_scale, f_loss = 0., 0.

            loc_loss = loc_sm_scale*(loc_sm_loss/100) 
            orient_loss = orient_sm_scale*orient_sm_loss 
            feetx_loss = feet_loss_scale*(f_loss*10) 
            
            # other constraints
            othog_value = k_pipe_kwargs["avg_normal_m"].view(-1) @ k_pipe_kwargs["n_g_single"].view(-1)
            othog_loss = (othog_value - 0)**2 # enforce target 0

            n_m_loss = (torch.norm(k_pipe_kwargs["avg_normal_m"]) - 1)**2 # enforce target 1
            n_g_loss = (torch.norm(k_pipe_kwargs["n_g_single"]) - 1)**2

            loss = pose_loss + loc_loss + orient_loss + feetx_loss + (othog_loss + n_m_loss + n_g_loss)

            loss.backward()
            losses.append(loss.item())
            wandb.log({'pose_loss':pose_loss, 'loc_loss': loc_loss,
                       'orient_loss': orient_loss, 'feet_loss': feetx_loss,
                       'othog_loss': othog_loss, 'n_m_loss': n_m_loss, 'n_g_loss': n_g_loss, 
                       'total loss': loss,
                       })


            k_pipe_kwargs["optimizer_house"][m_i].step()

            _, ax = plt.subplots()
            fig = plt.figure() 
            if iter_%print_every == 0:
                
                ''' 3D plot '''
                plt.imshow(image)

                true_xr, true_yr = est_2d_real[plot_id][:,0].detach().cpu().view(-1), est_2d_real[plot_id][:,1].detach().cpu().view(-1)
                true_xv, true_yv = est_2d_virt[plot_id][:,0].detach().cpu().view(-1), est_2d_virt[plot_id][:,1].detach().cpu().view(-1)
    
                xr, yr = proj2d_real[plot_id][:,0].detach().cpu().view(-1), proj2d_real[plot_id][:,1].detach().cpu().view(-1)
                xv, yv = proj2d_virt_up[plot_id][:,0].detach().cpu().view(-1), proj2d_virt_up[plot_id][:,1].detach().cpu().view(-1)

                if k_pipe_kwargs["skel_type"] == "alpha" or (k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"]) or \
                    k_pipe_kwargs["skel_type"] == "comb_set":
    
                    plot2d_halpe26_mirror_common_2D(xr,yr, image, plt, linestyle = "-", true=(true_xr, true_yr), plot_true=True)
                    plot2d_halpe26_mirror_common_2D(xv,yv, image, plt, linestyle = "--", true=(true_xv, true_yv), plot_true=True)
        

                # 3D quantities onto 2D (Mirror Normal and P_dash onto image)
                # ---------------- ground, normal, otho 2d plots------------------
                '''set desired mini-batch size for visualization'''
                p_dash_mini, p_mini = p_dash[(m_i*k_pipe_kwargs["batch_size"]):(m_i+1)*k_pipe_kwargs["batch_size"]], p[(m_i*k_pipe_kwargs["batch_size"]):(m_i+1)*k_pipe_kwargs["batch_size"]]
                n_g_mini = k_pipe_kwargs["normal_g"][(m_i*k_pipe_kwargs["batch_size"]):(m_i+1)*k_pipe_kwargs["batch_size"]]

                '''position of the mirror: midpoint of p and pdash'''
                N = (p + p_dash)/2 
                '''plotting the mirror normal from the mirror position'''
                normal_end = N + n_m
                '''project to 2d - uses batched dot product'''
                N_2d = torch.bmm(K_dups, N.view(B, 3, 1)); # -> (B, 3, 1)
                '''dividing by depth - the precision is slightly different in the 2nd decimal'''
                N_2d = torch.div(N_2d, N_2d[:, 2:3]); # -> (B, 3, 1)

                normal_end_2d =  torch.bmm(K_dups, normal_end.view(B, 3, 1))
                normal_end_2d = torch.div(normal_end_2d, normal_end_2d[:, 2:3])

                ground_end = N + n_g_mini
                ground_end_2d = torch.bmm(K_dups, ground_end.view(B, 3, 1)) 
                ground_end_2d = torch.div(ground_end_2d, ground_end_2d[:, 2:3]) 

                '''torch keeps at 4-point precision, numpy allows more'''
                otho = torch.cross(n_m, n_g_mini); 

                '''N point, n_m vector/direction'''
                otho_end = N + otho
                otho_end_2d = torch.bmm(K_dups, otho_end.view(B, 3, 1)) 
                otho_end_2d = torch.div(otho_end_2d, otho_end_2d[:, 2:3])

                ''' n_m and n_g_mini would not necessarily be othorgonal, use the otho between them
                to refine n_g_mini which we are not super sure of
                note: order matters, the reverse flips every x,y,z component by -1'''
                refine_ground = torch.cross(otho, n_m);

                '''In math: N is point, n_m is a vector/direction'''
                refine_ground_end = N + refine_ground
                refine_ground_end_2d = torch.bmm(K_dups, refine_ground_end.view(B, 3, 1))
                refine_ground_end_2d = torch.div(refine_ground_end_2d, refine_ground_end_2d[:, 2:3]) 
                
                '''Plotting 3D point in 2D looks okay'''
                # P_dash
                p_dash_2d = torch.bmm(K_dups, p_dash.view(B, 3, 1)) 
                p_dash_2d = torch.div(p_dash_2d, p_dash_2d[:, 2:3]) 

                # P
                p_2d = torch.bmm(K_dups, p.view(B, 3, 1)) 
                p_2d = torch.div(p_2d, p_2d[:, 2:3]) 
                
                plt.axis('off')
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

                if k_pipe_kwargs["clear_output"]:
                    display.clear_output(wait=True)

                def get_img_from_fig(fig, dpi=180):
                    #ref: https://stackoverflow.com/a/58641662/12761745
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=dpi)
                    buf.seek(0)
                    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    buf.close()
                    img = cv2.imdecode(img_arr, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img


                '''save the optimization progression in real-time'''
                result_dir = "/scratch/dajisafe/smpl/figures/images_to_video"
                if k_pipe_kwargs["interest_frame"] and k_pipe_kwargs["batch_size"] == 1:
                    name = [m_i if k_pipe_kwargs["batch_size"] == 1 else k_pipe_kwargs["show_image"][0]][0]
                    plt.savefig(result_dir + f"/real_time_{name}_%06d.jpg"%(iter_), dpi=150, bbox_inches='tight', pad_inches = 0)

                    # local saving
                    plt.savefig(result_dir + f"/real_time/{name}_%06d.jpg"%(iter_), dpi=150, bbox_inches='tight', pad_inches = 0)


                plot_img_np = get_img_from_fig(fig)
                wand_img = wandb.Image(plot_img_np, caption="Input image")
                wandb.log({"examples": wand_img})

                plt.show()
                # clear current figure
                plt.cla()
                plt.close(fig)
            plt.close('all')

        plt.close('all')
        # store per mini-batch
        kp3d_m_house.append(kp3d_m); kp3d_h_house.append(kp3d_h)
        proj2d_real_house.append(proj2d_real); proj2d_virt_up_house.append(proj2d_virt_up)
        img_url_house.append(image_urls)
        rot_mat_house.append(rot_mat)
        theta_house.append(theta)

        # projection quantities
        p_dash_2d_house.append(p_dash_2d), p_2d_house.append(p_2d), N_2d_house.append(N_2d)
        normal_end_2d_house.append(normal_end_2d), ground_end_2d_house.append(ground_end_2d)
        refine_ground_end_2d_house.append(refine_ground_end_2d), otho_end_2d_house.append(otho_end_2d)

        # store per mini-batch
        root_id = 0
        idx = get_parent_idx(joint_names, joint_parents)
        verify_get_parent_idx(joint_names, joint_parents)
        joint_trees = np.array(idx)

        if k_pipe_kwargs["skel_type"] == "alpha":
            kp3d_btw = alpha_to_hip_1st(kp3d_m, device=device)
        elif k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"]:
            kp3d_btw = mirror19_to_hip_1st(kp3d_m,skel.joint_names_mirrored_human_19, joint_names, **k_pipe_kwargs)
        elif k_pipe_kwargs["skel_type"]=="comb_set":
            kp3d_btw = comset26_to_hip_1st(kp3d_m,skel.joint_names_combset, joint_names, **k_pipe_kwargs)

        bone_length, mean_bl, std_bl, bv = calc_bone_length(kp3d_btw, root_id, joint_trees)

        # save each mini-batch
        '''only projections are now in common (if used) format, 
        all others are in h36m/alphapose default format'''

        to_pickle = [(f"theta", theta), ("initial_pose3d", k_pipe_kwargs["initial_pose3d"]), (f"kp3d", kp3d_m), (f"kp3d_h", kp3d_h), 
        (f"proj2d_real", proj2d_real), (f"proj2d_virt", proj2d_virt_up),
        (f"K_optim", k_pipe_kwargs["K"]) , (f"r_view_h", r_view_h), (f"v_view_h", v_view_h), (f"img_urls", image_urls),
        (f"est_2d_real", est_2d_real),  (f"est_2d_virt", est_2d_virt), ("bf_build", bf_build),
        ("optim_rotation3x3", rot_mat), ("optim_theta", theta), ("batch_size", k_pipe_kwargs["batch_size"]), 
        ("batch_no", m_i), ("b_orientation", orient), ("bone_length", bone_length),
        ("N3d", N), ("p3d", p), ("p_dash3d", p_dash),("n_m", n_m), ("plane_d", plane_d),  ("n_g_mini", n_g_mini), ("otho", otho),
        ("p_dash_2d", p_dash_2d), ("p_2d", p_2d), ("N_2d", N_2d),
        ("normal_end_2d", normal_end_2d), ("ground_end_2d", ground_end_2d),
        ("refine_ground_end_2d", refine_ground_end_2d), ("otho_end_2d", otho_end_2d),
        ("l2ws", l2ws), ("chosen_frames", k_pipe_kwargs["chosen_frames"]),
        ("bf_positive", bf_positive), 
        ("joint_names", joint_names), ("joint_parents", joint_parents),
        ("flipped_frames", flipped_frames), ("A_dash_tuple", k_pipe_kwargs["A_dash_tuple"]),
        ("A_dash", k_pipe_kwargs["A_dash"]), ("final_A", final_A),
        ("final_A_dash", final_A_dash), 
        ("avg_D", k_pipe_kwargs["avg_D"]), 
        ("degree_store", degree_store),
        ] 
        
        if k_pipe_kwargs["args"].rec_data:
            filename = k_pipe_kwargs["dir_path"] + f'/{k_pipe_kwargs["args"].seq_name}_{m_i}_{iter_}_{k_pipe_kwargs["iterations"]}iter_{k_pipe_kwargs["bool_params"]}_{uniq_id}.pickle'

        elif k_pipe_kwargs["args"].eval_data and not k_pipe_kwargs["args"].gt:
            filename = k_pipe_kwargs["dir_path"] + f'result_{view}_{m_i}_{k_pipe_kwargs["iterations"]}iter_{k_pipe_kwargs["bool_params"]}_{uniq_id}.pickle'
        
        elif k_pipe_kwargs["args"].net_data:
            pass
        elif k_pipe_kwargs["args"].eval_data and k_pipe_kwargs["args"].gt:
            filename = k_pipe_kwargs["dir_path"] + f'result_{view}_{m_i}_{k_pipe_kwargs["iterations"]}iter_{k_pipe_kwargs["bool_params"]}_{uniq_id}.pickle'
        else:
            filename = k_pipe_kwargs["dir_path"] + f'result_{view}_{m_i}_{k_pipe_kwargs["iterations"]}iter_{k_pipe_kwargs["bool_params"]}_{uniq_id}.pickle'
        save2pickle(filename, to_pickle)
        print(f"saved to {filename}")

        print()


    """compute eval just for getting right scale alone - only camera 3"""
    if int(k_pipe_kwargs["view"]) in [2,3,4,5,6,7] and k_pipe_kwargs["args"].print_eval:
        cam = int(k_pipe_kwargs["view"])
        eval_kwargs = k_pipe_kwargs
        eval_kwargs["result_id"] = uniq_id
        result, kp3d_c, gt3d_chosen_c = batch_evaluate(**eval_kwargs)
        
        result["loc_smooth_scale"] = loc_sm_scale

        wandb_dict= { 'cam': cam,
                    'loc_smooth_scale': loc_sm_scale,
                    'mpjpe_cam': result["mpjpe_in_mm"], 
                    'n_mpjpe_cam': result["n_mpjpe_in_mm"],
                    'pmpjpe_cam': result["pmpjpe_in_mm"],
                    'n_vals': result["No_of_evaluations"],
                    }
        if k_pipe_kwargs["skel_type"] == "alpha":
            wandb_dict['bl_accuracy'] = result["bl_acc"]
            
        wandb.log(wandb_dict)
        
    else:
        result = {}
                    
    # ---------------------------------------------------------------------------------------------
    """At the end of all optimization"""
    # calculate mean and standard deviation of all (Alpha in 26 joints)
    if len(kp3d_m_house) !=0:
        kp3d_m_house = torch.cat(kp3d_m_house)
        if k_pipe_kwargs["skel_type"] == "alpha":
            kp3d_btw = alpha_to_hip_1st(kp3d_m_house, device=device)

        elif k_pipe_kwargs["skel_type"]=="gt2d" and k_pipe_kwargs["mirror19"]:
            kp3d_btw = mirror19_to_hip_1st(kp3d_m_house,skel.joint_names_mirrored_human_19, joint_names, **k_pipe_kwargs)

        elif k_pipe_kwargs["skel_type"]=="comb_set":
            kp3d_btw = comset26_to_hip_1st(kp3d_m_house,skel.joint_names_combset, joint_names, **k_pipe_kwargs)
        bone_length, mean_bl, std_bl, bv = calc_bone_length(kp3d_btw, root_id, joint_trees)

        bones = get_bone_names(joint_names, joint_parents)

        # save some bonelength plot
        fig, ax1 = plt.subplots(1,1)
        n_bls  = bone_length.shape[1]
        x_range = torch.tensor(list(range(n_bls)), dtype=torch.float32)#.log()
        plt.errorbar(x_range, mean_bl.detach().cpu().numpy(),np.round(std_bl.detach().cpu().numpy()), linestyle='None', marker='^', capsize=3)
        plt.xlabel("Joint Names")
        plt.ylabel("BoneLength Mean-Std Error (m)")
        
        ax1.set_xticks(x_range)
        ax1.set_xticklabels(bones, minor=False, rotation=80)
 
        # save some bonefactor plot
        fig, ax2 = plt.subplots(1,1)
        n_bls  = bf_positive.view(-1).shape[0]
        x_range = torch.tensor(list(range(n_bls)), dtype=torch.float32)#.log()
        plt.errorbar(x_range, bf_positive.view(-1).detach().cpu().numpy(), linestyle='None', marker='^', capsize=3)
        plt.xlabel("Joint Names")
        plt.ylabel("Bone factor (no unit)")

        ax2.set_xticks(x_range)
        ax2.set_xticklabels(bones, minor=False, rotation=80)        
 
        num = kp3d_m_house.shape[0]
        print(f"Last unique identifier + camera id: {uniq_id}_cam_{view}")

        if int(k_pipe_kwargs["view"]) in [2,3,4,5,6,7] and k_pipe_kwargs["args"].print_eval:
            try:
                with open(project_dir + f"outputs/metrics.txt", "a") as myfile:
                    myfile.write(f"{uniq_id}_cam_{view}, r_prop:{real_prop}, v_prop {virt_prop}, fixed_v_prop {fixed_virt_prop} pmpjpe {result['pmpjpe_in_mm']} n_valid_frames {num} n_iterations {k_pipe_kwargs['iterations']}\n")
                    myfile.write(f"rec_eval_pts: {result['rec_eval_pts']}, gt_eval_pts: {result['gt_eval_pts']}\n")
            except:
                with open(project_dir + f"outputs/metrics.txt", "a") as myfile:
                    myfile.write(f"{uniq_id}_cam_{view}, r_prop:{real_prop} pmpjpe {result['pmpjpe_in_mm']} n_valid_frames {num} n_iterations {k_pipe_kwargs['iterations']}\n")
                    myfile.write(f"rec_eval_pts: {result['rec_eval_pts']}, gt_eval_pts: {result['gt_eval_pts']}\n")
        else:
            with open(project_dir + f"outputs/metrics.txt", "a") as myfile:
                if k_pipe_kwargs["args"].gt:
                    myfile.write(f"{uniq_id}_cam_{view}, r_prop:{real_prop}, v_prop {virt_prop}, fixed_v_prop {fixed_virt_prop} n_valid_frames {num} n_iterations {k_pipe_kwargs['iterations']}\n")
                else:
                    myfile.write(f"{uniq_id}_cam_{view}, r_prop:{real_prop}, v_prop {virt_prop}, fixed_v_prop {fixed_virt_prop} n_valid_frames {num} n_iterations {k_pipe_kwargs['iterations']} seq_name {k_pipe_kwargs['args'].seq_name}\n")
                    

    return image, kc, kp3d_m_house, kp3d_h_house, proj2d_real_house, proj2d_virt_up_house, K_dups, r_view_h, v_view_h, img_url_house, result, uniq_id
