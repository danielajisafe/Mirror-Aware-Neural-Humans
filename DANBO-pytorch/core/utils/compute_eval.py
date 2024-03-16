# standard libraries
import cv2
import ipdb
import json
import glob
import torch
import numpy as np
from glob import glob
#import argparse
import sys
import platform
from tqdm import tqdm, trange
sys.path.append("../")
sys.path.append("/scratch/dajisafe/smpl/")
sys.path.append("/scratch/dajisafe/smpl/mirror_project_dir/")
#----------------------------
#import util_skel as skel
import matplotlib.pyplot as plt
#from IPython import display
#---------------------------------------

import sys

naye_local_dir = "/scratch/dajisafe/smpl/A_temp_folder/A-NeRF"
cc_local_dir = "/home/dajisafe/scratch/anerf_mirr/A-NeRF/core/utils"
sockeye_local_dir = "/scratch/st-rhodin-1/users/dajisafe/anerf_mirr/A-NeRF"
sys.path.append(cc_local_dir)
sys.path.append(sockeye_local_dir)
sys.path.append(naye_local_dir)

#ipdb.set_trace()

# custom imports
import core.utils.util_skel as skel
from core.utils.pmpjpe import pmpjpe, procrustes
from core.utils.evaluation import MPJPE, NMPJPE
from core.utils.extras import load_pickle, save2pickle
from core.utils.extras import alpha_to_hip_1st, hip_1st_to_alpha, calc_bone_length
from core.utils.skeleton_utils import get_bone_names, get_parent_idx, verify_get_parent_idx

def eval_opt_kp(kps,comb, rec_eval_pts, gt_eval_pts):

    view = comb.split("_cam_")[1]
    print(f"camera: {view} comb: {comb}")

    rec_eval_pts = np.array(rec_eval_pts)
    gt_eval_pts = np.array(gt_eval_pts)

    # 
    n_kps = kps.shape[0]
    test_idxs = np.where((rec_eval_pts/n_kps) >1)[0]
    stp_idx = test_idxs[0]
    rec_eval_pts = rec_eval_pts[:stp_idx]
    gt_eval_pts = gt_eval_pts[:stp_idx]

    print(f"no of excluded test idxs: {len(test_idxs)}")
    print(f"rec_eval_pts: {rec_eval_pts}")
    print(f"gt_eval_pts: {gt_eval_pts}")
    
    cluster = platform.node()
    if cluster.startswith('se'):
        project_dir = "/scratch/st-rhodin-1/users/dajisafe/anerf_mirr"
        extri_file = project_dir + '/extri.yml'
        filename = project_dir + f"/mirror_GT3D.pickle"

    elif cluster.startswith('naye'):
        project_dir = "/scratch/dajisafe/smpl/mirror_project_dir"
        extri_file = project_dir + '/authors_eval_data/extri.yml'
        filename = project_dir + f"/authors_eval_data/mirror_GT3D.pickle"

    else:
        print("where are we?")
        ipdb.set_trace()

    skel_type = "alpha"

    #import ipdb; ipdb.set_trace()

    #calib_filename = project_dir + f"/authors_eval_data/calib_data_alphapose/calib{view}_20000iter.pickle"
    #sorted_recon = sorted(glob(project_dir + f"/authors_eval_data/new_recon_results_no_gt2d/{view}/*{idt}.pickle"))       
    #calib_filename = project_dir + f"/authors_eval_data/calib_data_alphapose_GTfocal/calib{view}_20000iter_May_11.pickle"
    
    
    #import ipdb; ipdb.set_trace()
    "remember we have droplast option, hence reference only whats passed in here"
    #proj2d_real, proj2d_virt, img_recon = [], [], []
    #est_2d_real, est_2d_virt, batch_values = [], [], []
    #img_recon, kp3d = [], []

    
    # for filename in sorted_recon:
    #     from_pickle = load_pickle(filename)
        
    #     kp3d.extend(from_pickle["kp3d"]) 
    #     img_recon.extend(from_pickle["img_urls"])    
        #batch_size, batch_no = from_pickle["batch_size"], from_pickle["batch_no"]
        #batch_values.extend(list(range((batch_no*batch_size), (batch_no+1)*batch_size)))
    
    #all((np.array(img_recon)[1700]) == np.array(sorted(img_recon)[1700]))
    #import ipdb; ipdb.set_trace()
    # sort based target on sorting img_recon
    #kp3d_ = list(map(lambda x:x[1], sorted(zip(img_recon, kps))))
    #batch_values = list(map(lambda x:x[1], sorted(zip(img_recon, batch_values))))
    # now sort images
    #img_recon = sorted(img_recon)
    
    # stack together
    kps = torch.Tensor(kps)
    kp3d = hip_1st_to_alpha(kps)
    
    #kp3d = torch.stack(kp3d)
    #import ipdb; ipdb.set_trace()

    #from_pickle = load_pickle(calib_filename)
    #chosen_frames = from_pickle["chosen_frames"]

    #import ipdb; ipdb.set_trace()
    '''read GT 3D data'''
    sequence_length = 1800
    #read extri file
    gt3d_real_all = []
    #gt3d_virt_all = []
    #imgs_gt = []

    #rec_eval_pts, gt_eval_pts = [], []
    #img_recon_names = [img.split("/")[-1] for img in img_recon]

    #import ipdb; ipdb.set_trace()
    # for index in trange(sequence_length):
    #     img_id_6 = f"{index:06d}" # 6 leading zeros
    #     #img_id_8 = f"{index:08d}"
    #     jsonfile = project_dir + f"/A-NeRF/data/mirror/keypoints3d/{img_id_6}.json"
    #     with open(jsonfile, 'r') as f:
    #         keypoints3d = json.load(f)
    #     gt3d_real_all.append(torch.Tensor(keypoints3d[0]['keypoints3d']))
    #     #gt3d_virt_all.append(torch.Tensor(keypoints3d[1]['keypoints3d']))
        

    # #import ipdb; ipdb.set_trace()
    # '''We would evaluate at the camera-coords level using the given REAL 3D pose. why only real? :)'''
    # gt3d_real_all = torch.stack(gt3d_real_all, dim=0).detach().cpu()


    
    from_pickle = load_pickle(filename)
    gt3d_real_all = from_pickle["gt3d_real_all"]

    # to_pickle = [("gt3d_real_all", gt3d_real_all)]
    # save2pickle(filename, to_pickle)

    

    #import ipdb; ipdb.set_trace()
    gt3d_chosen = gt3d_real_all[gt_eval_pts]
    kp3d_chosen = kp3d[rec_eval_pts].detach().cpu()
    # ---------------------------------------------wwww

    assert gt3d_chosen.shape[0] == kp3d_chosen.shape[0], "Eval GT 3D does not match the size of Eval Recon 3D"

    #import ipdb; ipdb.set_trace()
    # optimize on dcpose/mirror but eval on mirror format (No mirror to dc)
    # if skel_type == "dcpose":
    #     # unlikely dcpose 3D
    #     ans = input("do you have mirror2dc for dc common evaluation? y/n")
    #     if ans == "y":
    #         h36m_to_common_map_fn = skel.h36m_to_dc_common
    #         #gt3d_chosen = gt3d_chosen[:, skel.mirror_to_dc_common ,0:3] #Not yet
    #     else:
    #         h36m_to_common_map_fn = skel.h36m_to_mirror_common_fn
    #         gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror15 ,0:3] #drop confidence score
        
    if skel_type == "alpha" or (skel_type=="gt2d" and kwargs["mirror19"]):
        # keep at 25 for now
        h36m_to_common_map_fn = skel.alphapose_to_mirror_common_fn
        gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror25 ,0:3] #drop confidence score

    # elif skel_type == "gt2d" and not kwargs["mirror19"]:
    #     h36m_to_common_map_fn = skel.h36m_to_mirror_common_fn
    #     gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror15 ,0:3] #drop confidence score
    
    kp3d_final =  h36m_to_common_map_fn(kp3d_chosen)
    #import ipdb; ipdb.set_trace()
    
    # # TODO: Add feature for mirror19 recon, to update joints and parent names 
    # '''Calculate bone length error'''
    # # TODO: confirm kp_root_id?
    # kp_root_id = 8 # not 0?
    # kp_p_idx = get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
    # verify_get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
    # kp_joint_trees = np.array(kp_p_idx)
    
    # kp_bone_lengths, kp_mean_bl, kp_std_bl, kp_bv = calc_bone_length(kp3d_final, kp_root_id, kp_joint_trees)
    
    # gt_root_id = 8
    # gt_p_idx = get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
    # verify_get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
    # gt_joint_trees = np.array(gt_p_idx)
    # gt_bone_lengths, gt_mean_bl, gt_std_bl, gt_bv = calc_bone_length(gt3d_chosen, gt_root_id, gt_joint_trees)
    # #import ipdb; ipdb.set_trace()

    # # calc bone length error
    # eps = 1e-36
    # bl_error = abs(kp_mean_bl[:19] - gt_mean_bl[:19])/ (gt_mean_bl[:19] + eps)
    # bl_acc = torch.sum(bl_error < 0.20)/float(len(bl_error)) #%

    # bl_error = (kp_mean_bl[:19] - gt_mean_bl[:19])
    # bl_error = sum(bl_error**2)
    #import ipdb; ipdb.set_trace()

    
    # Extrinsics
    # 2) define which camera? 
    # ref: https://github.com/zju3dv/EasyMocap/blob/584ba2c1e85c626e90bbcfa6931faf8998c5ba84/easymocap/mytools/camera_utils.py#L13
    fs = cv2.FileStorage(extri_file, cv2.FILE_STORAGE_READ)
    Rot_3 = torch.Tensor(fs.getNode(f"Rot_{view}").mat()).view(1,3,3).detach().cpu()
    T_3 = torch.Tensor(fs.getNode(f"T_{view}").mat()).view(1,1,3).detach().cpu()

    B = gt3d_chosen.shape[0]
    #Tranform gt3D to Camera X coordinates
    #_ = input("No transform to camera since we evaluate in canonical space")
    gt3d_chosen = torch.bmm(gt3d_chosen, Rot_3.permute(0,2,1).repeat(B,1,1)) + T_3; #print("T_3", T_3)

    '''make both poses hip-centered to compare'''
    # if skel_type == "dcpose":
    #     ans = input("do you have mirror2dc for dc common evaluation? y/n")
    #     if ans == "y":
    #         hip_index = skel.joint_names_common_H36M_DCPose.index('pelvis')
    #     else:
    #         hip_index = skel.joint_names_mirrored_human_15.index('pelvis')
    
    if skel_type == "alpha" or (skel_type=="gt2d"):# and kwargs["mirror19"]):
        hip_index = skel.joint_names_common_Alphapose_n_Mirror.index('pelvis')
        
    # elif skel_type == "gt2d" and not kwargs["mirror19"]:
    #     hip_index = skel.joint_names_mirrored_human_15.index('pelvis')

    gt3d_chosen_c = gt3d_chosen - gt3d_chosen[:, hip_index:hip_index+1, :]
    kp3d_c = kp3d_final - kp3d_final[:, hip_index:hip_index+1, :]

    #drop keypoints in the feet (mirror authors excluded that) 
    gt3d_chosen_c = gt3d_chosen_c[:,:19,:]
    kp3d_c = kp3d_c[:,:19,:]

    #ipdb.set_trace()
    # apples-to-apples (Alpha-to-Mirr skel vs SMPL-to-Mirr skel comparison)
    resp = input("do you want apples-to-apples evaluation, yes or no? ")
    if resp == "yes":
        gt3d_chosen_c = gt3d_chosen_c[:,1:15,:]
        kp3d_c = kp3d_c[:,1:15,:]
        print("running apples-to-apples evaluation...")
    else:
        print("NOT running apples-to-apples evaluation...why?")
        #ipdb.set_trace()

    mpjpe = MPJPE(gt3d_chosen_c, kp3d_c)
    n_mpjpe = NMPJPE(gt3d_chosen_c, kp3d_c)
    err_pmpjpe = pmpjpe(gt3d_chosen_c.permute(0,2,1).detach().cpu().numpy(), kp3d_c.permute(0,2,1).detach().cpu().numpy())

    result = {'mpjpe_in_mm': round(mpjpe.item()*1000, 2),
              'n_mpjpe_in_mm': round(n_mpjpe.item()*1000, 2),
                'pmpjpe_in_mm': round(err_pmpjpe*1000, 2),
                'No_of_evaluations': f"{str(kp3d_c.shape[0])+'/18'}",
                 #'bl_acc': round(bl_acc.item()),
                 #'bl_error': bl_error,
                 'rec_eval_pts': rec_eval_pts,
                 'gt_eval_pts': gt_eval_pts
                 }

    print(result)

# # def batch_evaluate(args, step=100, view=int(args.view)):
# def batch_evaluate(step=100, **kwargs):
#     #import ipdb; ipdb.set_trace()
#     project_dir = "/scratch/dajisafe/smpl/mirror_project_dir"

#     idt = kwargs["result_id"]
#     view = kwargs["view"]
#     skel_type= kwargs["skel_type"]

#     # detections
#     if kwargs["eval_data"] and not kwargs["gt"] and not kwargs["alphapose"]:
        
#         #recon_path = project_dir + f'/authors_eval_data/new_recon_results_gt/{view}/result_{view}_*.pickle'
#         recon_path = project_dir + f'/authors_eval_data/new_recon_results_gt2d/{view}/*{idt}.pickle'
#         sorted_recon = sorted(glob(recon_path))

#         # get sequential indices and sort appropriately
#         #results_indices = list(map(lambda x: int(x.split("_{")[0].split("_")[-1]), results))
#         #sorted_recon = sort_B_via_A(results_indices, results)
#         calib_filename = project_dir + f"/authors_eval_data/calib_data/calib{view}_data_20000iter.pickle"
#         #import ipdb; ipdb.set_trace()

#     # gt 2D
#     elif kwargs["eval_data"] and kwargs["gt"] and not kwargs["useGTfocal"] and not kwargs["alphapose"]:
#         #recon_path = project_dir + f'/authors_eval_data/new_recon_results_gt/{view}/result_{view}_*.pickle'
#         recon_path = project_dir + f'/authors_eval_data/new_recon_results_gt2d/{view}/*{idt}.pickle'
#         sorted_recon = sorted(glob(recon_path))

#         # get sequential indices and sort appropriately
#         #results_indices = list(map(lambda x: int(x.split("_{")[0].split("_")[-1]), results))
#         #sorted_recon = sort_B_via_A(results_indices, results)
#         calib_filename = project_dir + f"/authors_eval_data/calib_data_with_GT2D/calib{view}_build_gt2d_20000iter.pickle"
#         #import ipdb;ipdb.set_trace()

#     # gt 2D and gt Focal
#     elif kwargs["eval_data"] and kwargs["gt"] and kwargs["useGTfocal"] and not kwargs["alphapose"]:
#         recon_path = project_dir + f'/authors_eval_data/new_recon_results_gt2d_gtfocal/{view}/*{idt}.pickle'
#         sorted_recon = sorted(glob(recon_path))

#         # get sequential indices and sort appropriately
#         #results_indices = list(map(lambda x: int(x.split("_{")[0].split("_")[-1]), results))
#         #sorted_recon = sort_B_via_A(results_indices, results)
#         calib_filename = project_dir + f"/authors_eval_data/calib_data_with_GT2D_GTfocal/calib{view}_build_gt2d_gtfocal_20000iter.pickle"
#         #stop
#         #import ipdb;ipdb.set_trace()

#     elif kwargs["eval_data"] and kwargs["alphapose"] and not kwargs["useGTfocal"]:
#         #import ipdb;ipdb.set_trace()
#         #8de086ca-b13b-4165-9780-94960f02dcde 
#         #calib_filename = project_dir + f"/authors_eval_data/calib_data_alphapose/calib{view}_20000iter.pickle"
#         sorted_recon = sorted(glob(project_dir + f"/authors_eval_data/new_recon_results_no_gt2d_no_gtfocal_May_11/{view}/*{idt}.pickle"))       
#         calib_filename = project_dir + f"/authors_eval_data/calib_data_alphapose/calib{view}_20000iter_May_11.pickle"

#     elif kwargs["eval_data"] and kwargs["alphapose"] and kwargs["useGTfocal"]:
#         #8de086ca-b13b-4165-9780-94960f02dcde
#         #calib_filename = project_dir + f"/authors_eval_data/calib_data_alphapose/calib{view}_20000iter.pickle"
#         sorted_recon = sorted(glob(project_dir + f"/authors_eval_data/new_recon_results_no_gt2d/{view}/*{idt}.pickle"))       
#         calib_filename = project_dir + f"/authors_eval_data/calib_data_alphapose_GTfocal/calib{view}_20000iter.pickle"
        
#     elif kwargs["rec_data"]:
#         pass


#     #import ipdb; ipdb.set_trace()
#     "remember we have droplast option, hence reference only whats passed in here"
#     kp3d, proj2d_real, proj2d_virt, img_recon = [], [], [], []
#     est_2d_real, est_2d_virt, batch_values = [], [], []
#     img_recon = []


#     for filename in sorted_recon:
#         from_pickle = load_pickle(filename)
        
#         kp3d.extend(from_pickle["kp3d"]) 
#         img_recon.extend(from_pickle["img_urls"])    
#         batch_size, batch_no = from_pickle["batch_size"], from_pickle["batch_no"]
#         batch_values.extend(list(range((batch_no*batch_size), (batch_no+1)*batch_size)))
    
#     #import ipdb; ipdb.set_trace()

#     #joint_names = from_pickle["joint_names"]
#     #joint_parents = from_pickle["joint_parents"]
#     #chosen_frames = from_pickle["chosen_frames"]

#     # sort based target on sorting img_recon
#     kp3d_ = list(map(lambda x:x[1], sorted(zip(img_recon, kp3d))))
#     batch_values = list(map(lambda x:x[1], sorted(zip(img_recon, batch_values))))
#     # now sort images
#     img_recon = sorted(img_recon)
#     # stack together
#     kp3d = torch.stack(kp3d_)
#     #import ipdb; ipdb.set_trace()

#     from_pickle = load_pickle(calib_filename)
#     chosen_frames = from_pickle["chosen_frames"]

#     #import ipdb; ipdb.set_trace()
#     '''read GT 3D data'''
#     sequence_length = 1800
#     #read extri file
#     extri_file = project_dir + '/authors_eval_data/extri.yml'
#     gt3d_real_all = []
#     gt3d_virt_all = []
#     #imgs_gt = []

#     rec_eval_pts, gt_eval_pts = [], []
#     img_recon_names = [img.split("/")[-1] for img in img_recon]

#     for index in trange(sequence_length):
#         img_id_6 = f"{index:06d}" # 6 leading zeros
#         img_id_8 = f"{index:08d}"
#         jsonfile = project_dir + f"/authors_eval_data/keypoints3d/{img_id_6}.json"
#         with open(jsonfile, 'r') as f:
#             keypoints3d = json.load(f)
#         gt3d_real_all.append(torch.Tensor(keypoints3d[0]['keypoints3d']))
#         gt3d_virt_all.append(torch.Tensor(keypoints3d[1]['keypoints3d']))
#         #imgs_gt.append(img_id_6)        

#         if index%step == 0 and index in batch_values and index in chosen_frames:
#             gt_eval_pts.append(index)
#             # find img_id_8 in img_recon
#             rec_eval_pts.append(img_recon_names.index(f"{img_id_8}.jpg"))

    
#     #import ipdb; ipdb.set_trace()
#     '''We would evaluate at the camera-coords level using the given REAL 3D pose. why only real? :)'''
#     gt3d_real_all = torch.stack(gt3d_real_all, dim=0)

#     assert len(gt_eval_pts) == len(rec_eval_pts), "gt_eval_pts and rec_eval_pts do not have the same length"
#     if len(gt_eval_pts) != 0:
#         print(f"You have {len(gt_eval_pts)}/18 points to evaluate")
#         print("******************************************************")
#         # print(f"gt_eval_pts: {gt_eval_pts}")
#         # print(f"rec_eval_pts: {rec_eval_pts}")

#     else:
#         print(f"You have 0/18 points to evaluate")
#         print("******************************************************")
#         return None

#     #import ipdb; ipdb.set_trace()
#     gt3d_chosen = gt3d_real_all[gt_eval_pts]
#     kp3d_chosen = kp3d[rec_eval_pts]
#     # ---------------------------------------------

#     assert gt3d_chosen.shape[0] == kp3d_chosen.shape[0], "Eval GT 3D does not match the size of Eval Recon 3D"

#     #import ipdb; ipdb.set_trace()
#     # optimize on dcpose/mirror but eval on mirror format (No mirror to dc)
#     if skel_type == "dcpose":
#         # unlikely dcpose 3D
#         ans = input("do you have mirror2dc for dc common evaluation? y/n")
#         if ans == "y":
#             h36m_to_common_map_fn = skel.h36m_to_dc_common
#             #gt3d_chosen = gt3d_chosen[:, skel.mirror_to_dc_common ,0:3] #Not yet
#         else:
#             h36m_to_common_map_fn = skel.h36m_to_mirror_common_fn
#             gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror15 ,0:3] #drop confidence score
        
#     elif skel_type == "alpha" or (skel_type=="gt2d" and kwargs["mirror19"]):
#         # keep at 25 for now
#         h36m_to_common_map_fn = skel.alphapose_to_mirror_common_fn
#         gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror25 ,0:3] #drop confidence score

#     # elif skel_type == "gt2d" and not kwargs["mirror19"]:
#     #     h36m_to_common_map_fn = skel.h36m_to_mirror_common_fn
#     #     gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror15 ,0:3] #drop confidence score
    
#     kp3d_final =  h36m_to_common_map_fn(kp3d_chosen)
#     #import ipdb; ipdb.set_trace()
    
#     # TODO: Add feature for mirror19 recon, to update joints and parent names 
#     '''Calculate bone length error'''
#     kp_root_id = 0
#     kp_p_idx = get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
#     verify_get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
#     kp_joint_trees = np.array(kp_p_idx)
#     #import ipdb; ipdb.set_trace()
#     kp_bone_lengths, kp_mean_bl, kp_std_bl, kp_bv = calc_bone_length(kp3d_final, kp_root_id, kp_joint_trees)
    
#     gt_root_id = 8
#     gt_p_idx = get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
#     verify_get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
#     gt_joint_trees = np.array(gt_p_idx)
#     gt_bone_lengths, gt_mean_bl, gt_std_bl, gt_bv = calc_bone_length(gt3d_chosen, gt_root_id, gt_joint_trees)

#     # calc bone length error
#     eps = 1e-36
#     bl_error = abs(kp_mean_bl[:19] - gt_mean_bl[:19])/ (gt_mean_bl[:19] + eps)
#     bl_acc = torch.sum(bl_error < 0.25)/float(len(bl_error)) #%
#     #import ipdb; ipdb.set_trace()

    
#     # Extrinsics
#     # 2) define which camera? 
#     # ref: https://github.com/zju3dv/EasyMocap/blob/584ba2c1e85c626e90bbcfa6931faf8998c5ba84/easymocap/mytools/camera_utils.py#L13
#     fs = cv2.FileStorage(extri_file, cv2.FILE_STORAGE_READ)
#     Rot_3 = torch.Tensor(fs.getNode(f"Rot_{kwargs['view']}").mat()).view(1,3,3)
#     T_3 = torch.Tensor(fs.getNode(f"T_{kwargs['view']}").mat()).view(1,1,3)

#     B = gt3d_chosen.shape[0]
#     #Tranform gt3D to Camera X coordinates
#     #_ = input("No transform to camera since we evaluate in canonical space")
#     gt3d_chosen = torch.bmm(gt3d_chosen, Rot_3.permute(0,2,1).repeat(B,1,1)) + T_3; #print("T_3", T_3)

#     '''make both poses hip-centered to compare'''
#     if skel_type == "dcpose":
#         ans = input("do you have mirror2dc for dc common evaluation? y/n")
#         if ans == "y":
#             hip_index = skel.joint_names_common_H36M_DCPose.index('pelvis')
#         else:
#             hip_index = skel.joint_names_mirrored_human_15.index('pelvis')
    
#     elif skel_type == "alpha" or (skel_type=="gt2d" and kwargs["mirror19"]):
#         hip_index = skel.joint_names_common_Alphapose_n_Mirror.index('pelvis')
        
#     elif skel_type == "gt2d" and not kwargs["mirror19"]:
#         hip_index = skel.joint_names_mirrored_human_15.index('pelvis')

#     gt3d_chosen_c = gt3d_chosen - gt3d_chosen[:, hip_index:hip_index+1, :]
#     kp3d_c = kp3d_final - kp3d_final[:, hip_index:hip_index+1, :]

#     #drop keypoints in the feet (mirror authors excluded that) 
#     gt3d_chosen_c = gt3d_chosen_c[:,:19,:]
#     kp3d_c = kp3d_c[:,:19,:]

#     mpjpe = MPJPE(gt3d_chosen_c, kp3d_c)
#     n_mpjpe = NMPJPE(gt3d_chosen_c, kp3d_c)
#     err_pmpjpe = pmpjpe(gt3d_chosen_c.permute(0,2,1).detach().numpy(), kp3d_c.permute(0,2,1).detach().numpy())

#     result = {'mpjpe_in_mm': round(mpjpe.item()*1000, 2),
#               'n_mpjpe_in_mm': round(n_mpjpe.item()*1000, 2),
#                 'pmpjpe_in_mm': round(err_pmpjpe*1000, 2),
#                 'No_of_evaluations': f"{str(kp3d_c.shape[0])+'/18'}",
#                  'bl_acc': round(bl_acc.item()),
#                  'rec_eval_pts': rec_eval_pts,
#                  'gt_eval_pts': gt_eval_pts
#                  }

#     print(result)
#     return result, kp3d_c, gt3d_chosen_c


# if __name__ == "__main__":
    
#     # Instantiate the parser
#     parser = argparse.ArgumentParser(description='Optional app description')
#     # Optional argument
#     parser.add_argument('--eval_data', help='Authors evaluation data', action= "store_true", default=False)
#     parser.add_argument('--net_data', help='internet data', action= "store_true", default=False)
#     parser.add_argument('--rec_data', help='our recorded data', action= "store_true", default=False)
#     parser.add_argument('--det', help='detection on eval data', action= "store_true", default=False)
#     parser.add_argument('--gt', help='gt annotation of eval data', action= "store_true", default=False)
#     parser.add_argument('--view', help='camera view')
#     parser.add_argument('--useGTfocal', help='gt focal length of eval data', action= "store_true", default=False)
#     parser.add_argument('--alphapose', help='alphapose detections', action= "store_true", default=False)
#     parser.add_argument('--mirror19', help='init starts as 19 keypoints', action= "store_true", default=False)
#     parser.add_argument('--skel_type', help='skeletion type')
#     parser.add_argument('--result_id', help='unique identifier for reconstruction results')

#     #parse
#     args = parser.parse_args()

#     kwargs = {"eval_data": args.eval_data, "net_data": args.net_data, "rec_data": args.rec_data, 
#             "view": int(args.view), "det": args.det, "gt": args.gt,
#             "useGTfocal": args.useGTfocal, "alphapose": args.alphapose, 
#             "mirror19": args.mirror19, 
#             "skel_type": args.skel_type, "result_id": args.result_id
#             }

#     result, kp3d_c, gt3d_chosen_c = batch_evaluate(**kwargs)
