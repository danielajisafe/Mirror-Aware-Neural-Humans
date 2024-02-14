
import os
import pdb
import time
import json
import torch
import torch.nn as nn
from typing import List

# custom imports 
from extras.theta import theta_params
from core_mirror.util_loading import increase_initial, normalize_batch_normal
from core_mirror.train_loop import new_kinematic_pipe
from core_mirror.mirror_geometry import generate_p_data, mirror_calibrate, mirror_calibrate_batch, mirror_operation, mirror_operation_batch,  visualize_sim


def new_run_setup(args, config_wb, show_image, clear_output, cam_matrix, video_dataset, video_loader, batch_size,
              normal_ground, train_dataset, print_every, iterations, bool_params, identity, dir_path, initial_flip_flag, flip_virt_flag, interest_frame,
              view=None,skel_type=None,use_mapper=None, mirror19=False):

    device = torch.device("cpu")
    n_mini_batch = len(video_dataset)//batch_size
    # ----------------------------
    '''Initial gt3d pose from h36m Camera 0'''
    if args.h36m_data_dir != 'None':

        sibling_no = 0
        initial_pose = train_dataset[0]['pose3d_siblings'][sibling_no][:, :]

        if skel_type == "alpha":
            # increase to 26 joints and convert to "hip-first"
            initial_pose, joint_names, joint_parents = increase_initial(initial_pose, init_type=skel_type, device=device)

        elif skel_type=="gt2d" and mirror19:
            # increase to 19 joints and convert to "hip-first"
            initial_pose, joint_names, joint_parents = increase_initial(initial_pose, init_type=skel_type, mirror19=mirror19, device=device)

        elif skel_type=="comb_set":
            # increase to 19 joints and convert to "hip-first"
            initial_pose, joint_names, joint_parents = increase_initial(initial_pose, init_type=skel_type, mirror19=mirror19, device=device)
    
    else: 
        if skel_type == "alpha":
            tpose_file = 'dataset/mirror_aware_h36m_adapt_tpose.json'
            
            if os.path.isfile(tpose_file):
                print("using accessed template pose...") ; time.sleep(3)  

                with open(tpose_file) as f:
                    tpose = json.load(f)
                initial_pose = torch.Tensor(tpose[skel_type]["initial_pose"])
                joint_names = tpose[skel_type]["joint_names"]
                joint_parents = tpose[skel_type]["joint_parents"]
            else:
                raise ValueError("\nTpose file is required. Reach out to dajisafe[at]cs.ubc.ca with subject title 'Mirror-Aware-Human Template Pose'")

        else:
            raise NotImplementedError

    scale = 1.
    # centered by hip
    initial_pose3d = (initial_pose - initial_pose[0:1, :]).to(device)
    initial_pose3d = scale * initial_pose3d

    # -----------------------------
    num_joints = initial_pose3d.shape[0] if len(initial_pose3d.shape)==2 else initial_pose3d.shape[1]
    theta = theta_params(batch_size, use_rot6D=True, start_zero=args.start_zero, num_joints=num_joints).to(device)

    theta_house = []
    for i in range(n_mini_batch):
        theta_mini = theta.clone().detach()
        theta_mini.requires_grad = True
        
        if args.disable_rot:
            "disable feet and face rotation"
            if skel_type=="alpha":
                # starts with template Alphapose26 hip-first
                nose, left_eye, right_eye = 14, 17,15
                right_foot, left_foot, right_heel, left_heel = 3,9, 4,10
                indices_to_mask = [nose, left_eye, right_eye, right_foot, left_foot, right_heel, left_heel] 
                
            elif skel_type=="gt2d" and mirror19:
                # starts with template mirror19 hip-first
                nose, left_eye, right_eye = 8, 11,9
                indices_to_mask = [nose, left_eye, right_eye]

            if skel_type=="comb_set":
                # starts with template combset hip-first
                nose, left_eye, right_eye = 14, 17,15
                right_foot, left_foot, right_heel, left_heel = 3,9, 4,10
                indices_to_mask = [nose, left_eye, right_eye, right_foot, left_foot, right_heel, left_heel] 

        elif args.disable_all_rot:
            if skel_type=="alpha":
                indices_to_mask = [i for i in range(26)]
            elif skel_type=="gt2d" and mirror19:
                indices_to_mask = [i for i in range(19)]
            elif skel_type=="combset":
                indices_to_mask = [i for i in range(26)]


        def theta_mask(grad):
            zero_mask = torch.ones_like(grad)
            '''disable these rotations, with hip-first in mind'''
            for ind in indices_to_mask:
                zero_mask[:,ind,:] = 0; 

            grad = grad * zero_mask
            return grad
 
        '''register hook only once - sets gradients permanently to 0'''
        if args.disable_rot or args.disable_all_rot:
            theta_mini.register_hook(theta_mask)
        theta_house.append(theta_mini)
    # ----------------------------

    '''for all'''
    a, normal_m, c, N, d, D, c_dash, n_g_rough = mirror_calibrate_batch(video_dataset.p, video_dataset.p_dash)  
    
    '''optimizing only a single mirror matrix for all pair of poses'''
    avg_normal_m = normal_m.mean(dim=0, keepdim=True)
    # normalize
    avg_normal_m = normalize_batch_normal(avg_normal_m)

    avg_D = D.mean(dim=0, keepdim=True)
    mirror_mat, plane_d = mirror_operation_batch(avg_D, avg_normal_m); 
    plane_d = plane_d.to(device)
    avg_normal_m, avg_D = avg_normal_m.to(device), avg_D.to(device) 

    A_dash = mirror_mat.to(device); 
    A_dash_tuple = None
    
    '''note: y negative (points up), xz positive 
    y positive (points down), xz negative '''

    if normal_ground[1] > 0:
        n_g_single = torch.Tensor((normal_ground * -1)).float().to(device)
    else: 
        n_g_single = torch.Tensor((normal_ground)).float().to(device)

    '''batch size of 1 is same as no batch option which is better''' 
    normal_g = n_g_single.view(1,3).repeat(len(video_dataset), 1)
    
    # Orientation and Position
    A = torch.cat([torch.cat([torch.eye(3), torch.Tensor([[0],[0],[0]])], axis=1), torch.Tensor([[0,0,0,1]])])
    A = A.unsqueeze(0).repeat(batch_size, 1,1).to(device); 

    print(f"cam_matrix: fx {cam_matrix[0,0]} fy {cam_matrix[1,1]}")
    print("ground plane normal+", normal_g[0])

    '''we might drop optimizing the camera, or optimize only 4 values of it'''
    K = torch.Tensor(cam_matrix).to(device)

    if bool_params["K_op"]: 
        K.requires_grad = True

        zero_mask = torch.zeros(3,3).to(device)
        '''only desired is updated'''
        zero_mask[0,0] = 1; zero_mask[1,1] = 1 
        zero_mask[0,2] = 1; zero_mask[1,2] = 1

        '''register hook only once - sets gradients permanently to 0'''
        K.register_hook(lambda grad: grad * zero_mask)

    # initialize hip as ankle + ground normal direction (which is p+g_normal)
    '''ankle + 1m - hips is not fixed for pose, based on ankles''' 
    hips = video_dataset.p.to(device) + (normal_g) 
    hips = hips.unsqueeze(1).to(device)

    # hip positions are different across minibatches (based on ankles)
    hips_house = [hips[(i*batch_size):(i+1)*batch_size].clone().detach().requires_grad_(True) for i in range(n_mini_batch)]
 
    # set conditional variables
    '''Bone factor multiplication'''
    if bool_params["bf_op"]: 
        if skel_type == "alpha":
            bf = torch.ones(1, 25, 1, 1)
        elif skel_type=="gt2d" and mirror19:
            bf = torch.ones(1, 18, 1, 1) 
        elif skel_type=="comb_set":
            bf = torch.ones(1, 25, 1, 1) 

        if bool_params["bf_build"]: 
            if skel_type == "alpha":
                bf = torch.ones(1,14, 1, 1) # 6x2+2x2+3x2+3 -> 25 bones
            elif skel_type=="gt2d" and mirror19:
                ''' 8 symmetric, 2 others '''
                bf = torch.ones(1, 10, 1, 1)# 8x2+2 -> 18 bones
            if skel_type == "comb_set":
                bf = torch.ones(1,14, 1, 1) # 6x2+2x2+3x2+3 -> 25 bones
        bf.requires_grad = True        

    #------------------------------------------------------------------
    criterion = nn.L1Loss()
    loss_fn = 'l1_weight' 

    '''base parameters'''
    parameters= [[theta_house[i], hips_house[i]] for i in range(n_mini_batch)]
         
    if bool_params["A_dash"]:
        if bool_params["A_dash_fewer"]:
            
            avg_normal_m.requires_grad = True; 
            plane_d.requires_grad = True

            A_dash_tuple = (plane_d); 
            A_dash = None

            for i in range(n_mini_batch):
                # should be same, hence gradients should flow back from all mini-batches
                parameters[i] += [avg_normal_m, plane_d]

        else:
            A_dash.requires_grad = True
            A_dash = A_dash.to(device)

            for i in range(n_mini_batch):
                # should be same, hence gradients should flow back from all mini-batches
                parameters[i] += [A_dash] 

    elif bool_params["n_m_single"]:
        avg_normal_m.requires_grad = True; 
        for i in range(n_mini_batch):
            # should be same, hence gradients should flow back from all mini-batches
            parameters[i] += [avg_normal_m]
            
            # only diff: plane_d is not optimized
            A_dash_tuple = (plane_d); 
            A_dash = None

    if bool_params["bf_op"]: 
        for i in range(n_mini_batch):
            # use a single copy across all mini-batches (since same person)
            # gradients should flow back from all mini-batches
            parameters[i] += [bf]
    else:
        bf = None

    # dropped
    if skel_type=="alpha":
        Rheel_RBT, Rheel_RST, Lheel_LBT, Lheel_LST = 4,5,10,11
        Nose_lEye, lEye_lEar, nose_rEye, rEye_rEar = 14,15,16,17
        bf_indices_to_mask  = [Rheel_RBT, Rheel_RST, Lheel_LBT, Lheel_LST,Nose_lEye, lEye_lEar, nose_rEye, rEye_rEar]
    
    elif skel_type=="gt2d" and mirror19:
        nose_rEye, rEye_rEar = 8,9
        Nose_lEye, lEye_lEar = 10,11
        bf_indices_to_mask = [nose_rEye, rEye_rEar,Nose_lEye, lEye_lEar]
    
    elif skel_type=="comb_set":
        bf_indices_to_mask = [None]

    def bf_mask(grad):
        zero_mask = torch.ones_like(grad)
        '''disable gradients for the bone lengths'''
        for ind in bf_indices_to_mask:
            zero_mask[:,ind,:,:] = 0; 
        grad = grad * zero_mask
        return grad

    '''sets gradients permanently to 0'''
    if args.disable_bf and bf is not None:
        bf.register_hook(bf_mask); pdb.set_trace()

    if bool_params["K_op"]: 
        for i in range(n_mini_batch):
            # use a single copy across all mini-batches (since camera is fixed)
            # gradients should flow back from all mini-batches
            parameters[i] += [K]

    if bool_params["n_g_single"]:
        n_g_single.requires_grad = True
        for i in range(n_mini_batch):
            parameters[i] += [n_g_single]

    '''optimization'''
    optimizer_house = [torch.optim.Adam(parameters[i], lr=1e-1) for i in range(n_mini_batch)]

    '''kwargs: keyword arguments'''
    k_pipe_kwargs = {"args": args, "eval_data": args.eval_data, "net_data": args.net_data,
                    "rec_data": args.rec_data, "det": args.det, "gt": args.gt,
                    "useGTfocal": args.useGTfocal, "alphapose": args.alphapose,
                    "mirror19": args.mirror19, 
                    "total_chosen_frames" : video_dataset.chosen_frames, 
                    "show_image": show_image, "clear_output" : clear_output, 
                    "batch_size": batch_size, "parameters" : parameters, 
                    "theta_house": theta_house, "video_loader": video_loader, 
                    "initial_pose3d": initial_pose3d, "iterations": iterations,
                    "optimizer_house": optimizer_house, "criterion" : criterion,
                    "hips_house": hips_house, "A":A, "A_dash": A_dash, 
                    "K": K, "avg_normal_m": avg_normal_m, "normal_g": normal_g, "n_g_single": n_g_single,
                    "loss_fn": loss_fn, "bool_params": bool_params, 
                    "device": device, "num_joints":num_joints,
                    "n_mini_batch":n_mini_batch, "identity": identity, 
                    "dir_path": dir_path, "rotate_initial_flag": initial_flip_flag,
                    "flip_virt_flag": flip_virt_flag, "A_dash_tuple": A_dash_tuple,
                    "bf_unknown": bf, "print_every": print_every, "view": view, 
                    "interest_frame": interest_frame,
                    "skel_type": skel_type, "use_mapper": use_mapper,
                    "mirror19": mirror19, "joint_names":joint_names, "joint_parents":joint_parents,
                    "loc_smooth_scale": config_wb.loc_smooth_scale, "orient_smooth_scale": config_wb.orient_smooth_scale,
                    "feet_loss_scale" : config_wb.feet_loss_scale, "avg_D":avg_D,
                    "offline_eval":args.offline_eval, "body15":args.body15, "apples14":args.apples14,
                    "comb_set":args.comb_set,
                    }

    print(f"n_g_single gradient *** {n_g_single.grad}")
    print(f"avg_normal_m gradient *** {avg_normal_m.grad}")
    image, kc_model, kp3d_house, kp3d_h_house, proj2d_real_house, proj2d_virt_house, K_optim, r_view, v_view, img_url_house, result, uniq_id = new_kinematic_pipe(**k_pipe_kwargs)
    
    print(f"n_g_single gradient *** {n_g_single.grad}, norm {torch.norm(n_g_single)}")
    print(f"avg_normal_m gradient *** {avg_normal_m.grad} norm {torch.norm(avg_normal_m)}")
    print(f"K gradient *** {K.grad}")
    return result, uniq_id