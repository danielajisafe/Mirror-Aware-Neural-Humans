# standard libraries
import cv2
import pdb
import json
import glob
import torch
import numpy as np
from glob import glob
import argparse
import sys
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
#---------------------------------------


# custom imports
import core_mirror.util_skel as skel
from core_mirror.pmpjpe import pmpjpe, procrustes
from extras.evaluation import MPJPE, NMPJPE
from core_mirror.util_loading import load_pickle, save2pickle, sort_B_via_A
from core_mirror.util_loading import alpha_to_hip_1st, hip_1st_to_alpha, calc_bone_length
from extras.skeleton_utils import get_bone_names, get_parent_idx, verify_get_parent_idx



def batch_evaluate(step=100, **kwargs):
    """ The reconstructed kps and Gt kps that gets here are non hip-first ordered.
        They are in their original ordering or arrangement."""

    import ipdb 
    project_dir = "."
    idt = kwargs["result_id"]
    view = kwargs["view"]
    skel_type= kwargs["skel_type"]
    args = kwargs['args']
   
    if kwargs["offline_eval"]:

        if kwargs["eval_data"] and kwargs["alphapose"] and kwargs["useGTfocal"] and not kwargs["gt"] and not args.comb_set:
            # mirror human eval data
            sorted_recon = sorted(glob(project_dir + f"outputs/new_recon_results_no_gt2d/{view}/*{idt}.pickle"))       
            calib_filename = project_dir + f"outputs/calib_data_alphapose_GTfocal/calib{view}_20000iter.pickle"

        elif kwargs["rec_data"] and kwargs["alphapose"] and not kwargs["useGTfocal"] and not kwargs["gt"] and not args.comb_set:
            # internet and internal data set
            sorted_recon = sorted(glob(project_dir + f"outputs/new_recon_results_no_gt2d_no_gtfocal_May_11/{view}/*{idt}.pickle"))       
            calib_filename = project_dir + f"outputs/calib_data_alphapose/calib{view}_20000iter_May_11.pickle"

        # detections
        elif kwargs["eval_data"] and not kwargs["gt"] and not kwargs["alphapose"] and not args.comb_set:
            recon_path = project_dir + f'outputs/new_recon_results_gt2d/{view}/*{idt}.pickle'
            sorted_recon = sorted(glob(recon_path))
            calib_filename = project_dir + f"outputs/calib_data/calib{view}_data_20000iter.pickle"

        # gt 2D
        elif kwargs["eval_data"] and kwargs["gt"] and not kwargs["useGTfocal"] and not kwargs["alphapose"] and not args.comb_set:
            recon_path = project_dir + f'outputs/new_recon_results_gt2d/{view}/*{idt}.pickle'
            sorted_recon = sorted(glob(recon_path))
            calib_filename = project_dir + f"outputs/calib_data_with_GT2D/calib{view}_build_gt2d_20000iter.pickle"

        # gt 2D and gt Focal
        elif kwargs["eval_data"] and kwargs["gt"] and kwargs["useGTfocal"] and not kwargs["alphapose"] and not args.comb_set:
            day = "26"
            recon_path = project_dir + f'outputs/new_recon_results_gt2d_gtfocal_Jan{day}_2023/{view}/*{idt}.pickle'
            sorted_recon = sorted(glob(recon_path))
            calib_filename = project_dir + f"outputs/calib_data_with_GT2D_GTfocal_Jan{day}_2023/calib{view}_build_gt2d_gtfocal_20000iter.pickle"

        elif args.comb_set:
            recon_path = project_dir + f"outputs/new_recon_results_gt2d_plus_alphapose/{view}/*{idt}.pickle"
            sorted_recon = sorted(glob(recon_path))

        elif kwargs["rec_data"]:
            pass
  
    else:
        print("real-time evaluation")
        args = kwargs['args']
        sorted_recon = sorted(glob(f"{args.recon_dir}*{idt}.pickle"))

    "kind reminder: we have droplast option, hence reference only whats passed in here"
    kp3d, proj2d_real, proj2d_virt, img_recon = [], [], [], []
    est_2d_real, est_2d_virt, batch_values = [], [], []
    img_recon = []
    
    # for visualization or debugging
    kp3d_h = [] 
    proj2d_real, proj2d_virt = [], []

    for filename in sorted_recon:
        from_pickle = load_pickle(filename)
        
        kp3d.extend(from_pickle["kp3d"]) 
        img_recon.extend(from_pickle["img_urls"])    
        batch_size, batch_no = from_pickle["batch_size"], from_pickle["batch_no"] 
        batch_values.extend(list(range((batch_no*batch_size), (batch_no+1)*batch_size)))

        # for visualization or debugging
        kp3d_h.extend(from_pickle["kp3d_h"]) 
        proj2d_real.extend(from_pickle["proj2d_real"])
        proj2d_virt.extend(from_pickle["proj2d_virt"]) 
    
    if kwargs["offline_eval"]:
        # pick from any or last recon (usually 2000iter)
        chosen_frames = from_pickle["chosen_frames"].tolist()

    # sort based target on sorting img_recon
    kp3d_ = list(map(lambda x:x[1], sorted(zip(img_recon, kp3d))))
    batch_values = list(map(lambda x:x[1], sorted(zip(img_recon, batch_values))))

    # debug purposes
    kp3d_h_ = list(map(lambda x:x[1], sorted(zip(img_recon, kp3d_h))))
    proj2d_real_ = list(map(lambda x:x[1], sorted(zip(img_recon, proj2d_real))))
    proj2d_virt_ = list(map(lambda x:x[1], sorted(zip(img_recon, proj2d_virt))))

    # now sort images
    img_recon = sorted(img_recon)
    kp3d = torch.stack(kp3d_).detach().cpu()

    # debug purposes
    kp3d_h = torch.stack(kp3d_h_).detach().cpu()
    proj2d_real = torch.stack(proj2d_real_).detach().cpu()
    proj2d_virt = torch.stack(proj2d_virt_).detach().cpu()

    if not kwargs["offline_eval"] and "total_chosen_frames" in kwargs:
        chosen_frames = kwargs["total_chosen_frames"]
        
    '''read GT 3D data'''
    sequence_length = 1800
    #read extri file
    extri_file = project_dir + '/dataset/zju-m-seq1/extri.yml'
    gt3d_real_all = []
    gt3d_virt_all = []

    rec_eval_pts, gt_eval_pts = [], []
    img_recon_names = [img.split("/")[-1] for img in img_recon]

    for index in trange(sequence_length):
        img_id_6 = f"{index:06d}" # 6 leading zeros
        img_id_8 = f"{index:08d}"
        jsonfile = project_dir + f"/dataset/zju-m-seq1/keypoints3d/{img_id_6}.json"
        with open(jsonfile, 'r') as f:
            keypoints3d = json.load(f)
        gt3d_real_all.append(torch.Tensor(keypoints3d[0]['keypoints3d']))
        gt3d_virt_all.append(torch.Tensor(keypoints3d[1]['keypoints3d']))

        if index%step == 0 and index in batch_values and index in chosen_frames:
            gt_eval_pts.append(index)
            # find img_id_8 in img_recon
            rec_eval_pts.append(img_recon_names.index(f"{img_id_8}.jpg")) # index gives the local information, value gives the global relative information.

    '''We would evaluate at the camera-coords level using the given REAL 3D pose. As the real and virt is just one pose'''
    gt3d_real_all = torch.stack(gt3d_real_all, dim=0)
    '''remove this after: sanity check'''
    if skel_type == "gt2d" and kwargs["gt"] and not args.comb_set: 
        cam_id = view

        print("")
        """these are eval indices within GT2D-manual correction valid frames"""
        if cam_id==2:
            rec_gt_pts = [0, 100, 200, 300, 400, 494, 594, 694, 794, 894, 994, 1094, 1194, 1294, 1394]
            gt_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
            
        elif cam_id==3:
            rec_gt_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
            gt_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]

        elif cam_id==5:
            rec_gt_pts = [0, 100, 200, 288, 388, 488, 588, 688, 788, 888, 988, 1088]
            gt_pts = [0, 100, 200, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
        
        elif cam_id==6:
            rec_gt_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
            gt_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]

        elif cam_id==7:
            rec_gt_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1636]
            gt_pts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700]
    
        else:
            print("what are your eval idx?")
            pdb.set_trace()
        
        assert rec_gt_pts == rec_eval_pts, "is rec_eval_pts correct?"
        assert gt_pts == gt_eval_pts, "is gt_eval_pts correct?"

    # -------------------------------------------------------------------
    rec_eval_pts = np.array(rec_eval_pts)
    gt_eval_pts = np.array(gt_eval_pts)

    if args.protocol2 and (skel_type == "gt2d" and kwargs["gt"]):
        print("******Protocol II: *******.")

        gt_train_set_size = {"2":1415, "3":1620, "5":1240, "6":1620, "7":1563}
        n_kps = gt_train_set_size[str(view)]
        test_idxs = np.where((rec_eval_pts/n_kps) >1)[0]

        stp_idx = len(rec_eval_pts) if len(test_idxs)==0 else test_idxs[0] 
        rec_eval_pts = rec_eval_pts[:stp_idx]
        gt_eval_pts = gt_eval_pts[:stp_idx]

        print(f"no of excluded test idxs: {len(test_idxs)} - {(test_idxs)}")
        print(f"rec_eval_pts: {rec_eval_pts}")
        print(f"gt_eval_pts: {gt_eval_pts}")

    else:
        print("******Protocol I: ******")
    # -------------------------------------------------------------------

    assert len(gt_eval_pts) == len(rec_eval_pts), "gt_eval_pts and rec_eval_pts do not have the same length"
    if len(gt_eval_pts) != 0:
        print(f"You have {len(gt_eval_pts)}/18 points to evaluate")
        print("***********************************")

    else:
        print(f"You have 0/18 points to evaluate")
        print("***********************************")
        return None

    gt3d_chosen = gt3d_real_all[gt_eval_pts]
    kp3d_chosen = kp3d[rec_eval_pts]
    # ---------------------------------------------

    assert gt3d_chosen.shape[0] == kp3d_chosen.shape[0], "Eval GT 3D does not match the size of Eval Recon 3D"
    # optimize on dcpose/mirror but eval on mirror format (No mirror to dc)
    if skel_type == "dcpose":
        # unlikely dcpose 3D
        ans = input("do you have mirror2dc for dc common evaluation? y/n")
        if ans == "y":
            h36m_to_common_map_fn = skel.h36m_to_dc_common
        else:
            h36m_to_common_map_fn = skel.h36m_to_mirror_common_fn
            gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror15 ,0:3] #drop confidence score
        
    elif skel_type == "alpha":# or (skel_type=="gt2d" and kwargs["mirror19"]):
        # keep at 25 for now
        h36m_to_common_map_fn = skel.alphapose_to_mirror_common_fn
        gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror25 ,0:3] # drop confidence score

    elif skel_type == "gt2d" and kwargs["mirror19"]:
        h36m_to_common_map_fn = skel.Identity_map # keep at mirror19
        gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror19 ,0:3] # mirror25-to-mirror19 mapping # drop confidence score
    
    elif skel_type == "comb_set":
        h36m_to_common_map_fn = skel.Identity_map # keep at 26 (m19+alpha6+1head)
        gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror19 ,0:3] # mirror25-to-mirror19 mapping # drop confidence score
       
    # dcpose?
    elif skel_type == "gt2d" and not kwargs["mirror19"]:
        h36m_to_common_map_fn = skel.h36m_to_mirror_common_fn # 17-to-15 mapping
        gt3d_chosen = gt3d_chosen[:, skel.mirror_to_mirror15 ,0:3] # mirror25-to-body15 mapping # drop confidence score
    
    kp3d_final =  h36m_to_common_map_fn(kp3d_chosen)
        
    '''Calculate bone length error'''
    if skel_type == "alpha":
        kp_root_id = 0
        kp_p_idx = get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
        verify_get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
        kp_joint_trees = np.array(kp_p_idx)
        kp_bone_lengths, kp_mean_bl, kp_std_bl, kp_bv = calc_bone_length(kp3d_final, kp_root_id, kp_joint_trees)
        
        gt_root_id = 8
        gt_p_idx = get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
        verify_get_parent_idx(skel.joint_names_common_Alphapose_n_Mirror, skel.joint_parents_common_Alphapose_n_Mirror)
        gt_joint_trees = np.array(gt_p_idx)
        gt_bone_lengths, gt_mean_bl, gt_std_bl, gt_bv = calc_bone_length(gt3d_chosen, gt_root_id, gt_joint_trees)

        # calc bone length error
        eps = 1e-36
        bl_error = abs(kp_mean_bl[:19] - gt_mean_bl[:19])/ (gt_mean_bl[:19] + eps)
        bl_acc = torch.sum(bl_error < 0.25)/float(len(bl_error)) #%

    
    # Extrinsics
    # 2) define which camera? 
    # ref: https://github.com/zju3dv/EasyMocap/blob/584ba2c1e85c626e90bbcfa6931faf8998c5ba84/easymocap/mytools/camera_utils.py#L13
    fs = cv2.FileStorage(extri_file, cv2.FILE_STORAGE_READ)
    Rot_3 = torch.Tensor(fs.getNode(f"Rot_{kwargs['view']}").mat()).view(1,3,3)
    T_3 = torch.Tensor(fs.getNode(f"T_{kwargs['view']}").mat()).view(1,1,3)

    B = gt3d_chosen.shape[0]
    #Tranform gt3D to Camera X coordinates
    gt3d_chosen = torch.bmm(gt3d_chosen, Rot_3.permute(0,2,1).repeat(B,1,1)) + T_3


    '''make both poses hip-centered to compare'''
    if skel_type == "alpha": 
        hip_index = skel.joint_names_common_Alphapose_n_Mirror.index('pelvis')
        
    elif skel_type == "gt2d" and kwargs["mirror19"]:
        hip_index = skel.joint_names_mirrored_human_19.index('pelvis')

    elif skel_type == "comb_set":
        hip_index = skel.joint_names_combset.index('pelvis')
    
    elif skel_type == "gt2d" and not kwargs["mirror19"]:
        hip_index = skel.joint_names_mirrored_human_15.index('pelvis')

    gt3d_chosen_c = gt3d_chosen - gt3d_chosen[:, hip_index:hip_index+1, :]
    kp3d_c = kp3d_final - kp3d_final[:, hip_index:hip_index+1, :]

    if kwargs["apples14"]:
        joint_idxs = np.arange(1,15)
        print("running apples14 (S1) evaluation...")
    elif kwargs["body15"]:
        joint_idxs = np.arange(0,15)
        print("running body15 evaluation...")
    else:
        print("running 19-joints (S2) evaluation...why?") # (data authors excluded feet joints) 
        joint_idxs = np.arange(0,19)

    gt3d_chosen_c = gt3d_chosen_c[:,joint_idxs,:]
    kp3d_c = kp3d_c[:,joint_idxs,:]

    mpjpe = MPJPE(gt3d_chosen_c, kp3d_c)
    n_mpjpe = NMPJPE(gt3d_chosen_c, kp3d_c)
    err_pmpjpe = pmpjpe(gt3d_chosen_c.permute(0,2,1).detach().numpy(), kp3d_c.permute(0,2,1).detach().numpy())

    result = {  'mpjpe_in_mm': round(mpjpe.item()*1000, 2),
                'n_mpjpe_in_mm': round(n_mpjpe.item()*1000, 2),
                'pmpjpe_in_mm': round(err_pmpjpe*1000, 2),
                'No_of_evaluations': f"{str(kp3d_c.shape[0])+'/18'}",
                'rec_eval_pts': rec_eval_pts,
                'gt_eval_pts': gt_eval_pts
            }

    if skel_type == "alpha":
        result["bl_acc"] = round(bl_acc.item())

    print(result)
    return result, kp3d_c, gt3d_chosen_c


if __name__ == "__main__":
    
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # Optional argument
    parser.add_argument('--eval_data', help='Authors evaluation data', action= "store_true", default=False)
    parser.add_argument('--net_data', help='internet data', action= "store_true", default=False)
    parser.add_argument('--rec_data', help='our recorded data', action= "store_true", default=False)
    parser.add_argument('--det', help='detection on eval data', action= "store_true", default=False)
    parser.add_argument('--gt', help='gt annotation of eval data', action= "store_true", default=False)
    parser.add_argument('--offline_eval', help='offline evaluation post optimization', action= "store_true", default=False)
    parser.add_argument('--view', help='camera view')
    parser.add_argument('--useGTfocal', help='gt focal length of eval data', action= "store_true", default=False)
    parser.add_argument('--alphapose', help='alphapose detections', action= "store_true", default=False)
    parser.add_argument('--mirror19', help='init starts as 19 keypoints', action= "store_true", default=False)
    parser.add_argument('--skel_type', help='skeletion type')
    parser.add_argument('--result_id', help='unique idcompute_eval.pyentifier for reconstruction results')

    # evaluation related
    parser.add_argument('--body15', help='use common 15 joints between alphapose and Mirror skeleton - includes nose joint', action= "store_true", default=False)
    parser.add_argument('--apples14', help='use common 14 joints between SMPL, alphapose and Mirror skeleton', action='store_true', default=False)

    parser.add_argument('--protocol2', help='evaluate the trainset (90%) of valid frames', action='store_true', default=False)
    parser.add_argument('--comb_set', help='combined GT2D + alphapose feet', action='store_true', default=False)
    
    
    args = parser.parse_args()

    kwargs = {"eval_data": args.eval_data, "net_data": args.net_data, "rec_data": args.rec_data, 
            "view": int(args.view), "det": args.det, "gt": args.gt,
            "useGTfocal": args.useGTfocal, "alphapose": args.alphapose, 
            "mirror19": args.mirror19, 
            "skel_type": args.skel_type, "result_id": args.result_id,
            "offline_eval": args.offline_eval, "body15": args.body15, "apples14": args.apples14,
            "args": args
            }
    result, kp3d_c, gt3d_chosen_c = batch_evaluate(**kwargs)
