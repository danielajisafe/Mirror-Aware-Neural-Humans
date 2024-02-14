
import os
import pdb
import torch

import time
import wandb
import argparse
import datetime

from IPython import display
from numpy.core.numeric import identity
from core_mirror.dev import new_run_setup
from extras.DataLoader import video_loader                                                                                                                                                                                                                                                                                                                                     
from extras.DataLoader import h36m_dataset
from core_mirror.util_loading import load_pickle, save2pickle, update_img_paths


# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# dataset-related 
parser.add_argument('--project_dir', help='directory to project', default=".")
parser.add_argument('--eval_data', help='Authors evaluation data', action= "store_true", default=False)
parser.add_argument('--net_data', help='internet data', action= "store_true", default=False)
parser.add_argument('--rec_data', help='our recorded data', action= "store_true", default=False)
parser.add_argument('--h36m_data_dir', help='directory to processed h36m data', default="path/to/h36m/processed/")

parser.add_argument('--load_filename', help='load calibration file with 3D ankles, cam matrix etc')
parser.add_argument('--recon_dir', help='directory to store reconstructed 3D pose') 

parser.add_argument('--det', help='detections on eval data', action= "store_true", default=False)
parser.add_argument('--gt', help='gt annotation on eval data', action= "store_true", default=False)
parser.add_argument('--comb_set', help='gt annotation + alphapose detections on eval data', action= "store_true", default=False) 
parser.add_argument('--use_alpha_ankles', help='use alpha ankles instead of gt calibrated ankles', action= "store_true", default=False)

parser.add_argument('--loc_smooth_loss', help='smoothness constraints for joint poisitions over consecutive frames', action= "store_true", default=False)
parser.add_argument('--orient_smooth_loss', help='smoothness constraints for joint orientations over consecutive frames', action= "store_true", default=False)
parser.add_argument('--feet_loss', help='penalizing the feet to be closer to the ground', action= "store_true", default=False)
parser.add_argument('--view', help='camera view')
parser.add_argument('--seq_name', help='sequence id, actor or video name (if using internet or internal data)')
parser.add_argument('--infostamp', help='information stamp to save results to')

parser.add_argument('--inline_com', help='write an (inline) terminal comment')
parser.add_argument('--skel_type', help='skeleton type')
parser.add_argument('--num_workers', help='number of workers for data loader', default=12)

parser.add_argument('--useGTfocal', help='gt focal length of eval data', action= "store_true", default=False)
parser.add_argument('--opt_k', help='optimize focal length', action= "store_true", default=False)
parser.add_argument('--use_mapper', help='use mapper', action= "store_true", default=False)
parser.add_argument('--alphapose', help='init starts as alpha26, optimization on alpha25', action= "store_true", default=False)
parser.add_argument('--mirror19', help='init starts as mirror19, optimization on mirror19', action= "store_true", default=False)
parser.add_argument('--disable_rot', help='disable some rotation gradients i.e for the feet', action= "store_true", default=False)
parser.add_argument('--disable_all_rot', help='disable all rotation gradients', action= "store_true", default=False)
parser.add_argument('--start_zero', help='start rotation from 0 degree', action= "store_true", default=False)
parser.add_argument('--disable_bf', help='disable the gradients for bone factors', action= "store_true", default=False)
parser.add_argument('--print_eval', help='print pose evaluation results at the end of training', action= "store_true", default=False)
parser.add_argument('--alpha_feet_weight', help='scale the weight of the feet detections in GT2D+alphapose combined set', type=float, default=1.0)

parser.add_argument('--print_every', help='print every n steps ',  type=int, default=200)
parser.add_argument('--iterations', help='total no of iterations for training',  type=int, default=2000)

# evaluation-related
parser.add_argument('--offline_eval', help='offline evaluation post optimization', action= "store_true", default=False)
parser.add_argument('--body15', help='use common 15 joints between alphapose and mirror skeleton - includes nose joint', action= "store_true", default=False)
parser.add_argument('--apples14', help='use common 14 joints between SMPL, alphapose and mirror skeleton', action='store_true', default=False)
parser.add_argument('--protocol2', help='evaluate the trainset (90%) of valid frames', action='store_true', default=False)
    
# parse
args = parser.parse_args()
start = time.time()

project_dir = args.project_dir
# -------------------------------------------------------Parse Arguments -----------------------------------------------------------------------------
if args.eval_data and not args.gt and not args.alphapose and not args.comb_set:
    # dc pose
    view = args.view # (2,3,4,5,6,7)
    recon_dir = project_dir + f"outputs/new_recon_results_no_gt2d/{view}/"    

    load_filename = project_dir + f"outputs/calib_data/calib{view}_data_20000iter.pickle"
    save_filename = project_dir + f"outputs/calib_data/calib{view}_data_20000iter.pickle"
    identity = view

elif args.eval_data and args.gt and not args.alphapose and not args.comb_set:
    view = args.view 
    # GT 2D 
    json_file = project_dir + f"outputs/annots/combined/{view}/{view}.json"
    image_directory = project_dir + f'outputs/images/frames/{view}_mp4'
    
    '''choose to use GT 2d alone or with GT focal'''
    if args.useGTfocal:

        day = "26"
        load_filename = project_dir + f"outputs/calib_data_with_GT2D_GTfocal_Jan{day}_2023/calib{view}_build_gt2d_gtfocal_20000iter.pickle"
        recon_dir = project_dir + f"outputs/new_recon_results_gt2d_gtfocal_Jan{day}_2023/{view}/"   
    else:    
        load_filename = project_dir + f"outputs/calib_data_with_GT2D/calib{view}_build_gt2d_20000iter.pickle"
        recon_dir = project_dir + f"outputs/new_recon_results_gt2d/{view}/"   

elif args.eval_data and args.alphapose and args.useGTfocal and not args.gt and not args.comb_set:
    # mirror human eval data
    view = args.view 
    load_filename = project_dir + f"outputs/calib_data_alphapose_GTfocal/calib{view}_{args.infostamp}.pickle"
    recon_dir = project_dir + f"outputs/new_recon_results_no_gt2d/{view}/" 
    print(f"using: {load_filename}")
    time.sleep(3)

elif args.rec_data and args.alphapose and not args.useGTfocal and not args.gt and not args.comb_set:
    # internet and internal data set
    view = args.view 
    load_filename = project_dir + f"outputs/calib_data_alphapose/calib{view}_{args.seq_name}_{args.infostamp}.pickle"
    recon_dir = project_dir + f"outputs/recon_results_no_gt2d_no_gtfocal/{view}/"

elif args.eval_data and args.alphapose and not args.useGTfocal and not args.comb_set:
    view = args.view 

    load_filename = project_dir + f"outputs/calib_data_alphapose/calib{view}_20000iter_May_11.pickle"
    recon_dir = project_dir + f"outputs/new_recon_results_no_gt2d_no_gtfocal_May_11/{view}/" 

elif args.comb_set:
    """ combined gt2D (19 joints) and alphapose detections (6 feet joints) """
    view = args.view 
    # GT 2D 
    json_file = project_dir + f"outputs/annots/combined/{view}/{view}.json"
    image_directory = project_dir + f'outputs/images/frames/{view}_mp4'
    
    '''choose to use GT 2d alone or with GT focal'''
    if args.useGTfocal:
        day = "26"
        load_filename = project_dir + f"outputs/calib_data_with_GT2D_GTfocal_Jan{day}_2023/calib{view}_build_gt2d_gtfocal_20000iter.pickle"

    load_filename2 = project_dir + f"outputs/calib_data_alphapose_GTfocal/calib{view}_20000iter.pickle"
    recon_dir = project_dir + f"outputs/new_recon_results_gt2d_plus_alphapose/{view}/" 
    print(f"using: {load_filename}\n{load_filename2}")
    time.sleep(3)
    

else:
    raise ValueError("please choose a dataset among {--eval_data, --net_data, --rec_data}. For eval data, add --gt or --useGTfocal or --det flag")

os.makedirs(recon_dir, exist_ok=True)
# -------------------------------------------------------Parameters -----------------------------------------------------------------------------
# optimize either n_m_single only (recommended) | A_dash + A_dash_fewer -> + plane_d | Adash + n_m_single

bool_params = {
                 "A_dash": False, "A_dash_fewer": False, "K_op": args.opt_k, # modify
                #-----------------------------------------------------------------
                "bf_op": True, "bf_build":True, "n_g_single": True, # base case
                "n_m_single": True
                }

show_image = [0,0] 
clear_save_dir = True
clear_output = True
interest_frame = False
use_mapper  = args.use_mapper 
skel_type= args.skel_type 
rotate_initial_flag = False 
flip_virt_flag = True 
sub_sample = None

tag = "loc_smooth_scale"
loc_values = [0.22] 
# created for ablation studies
pmpjpe_results, bl_acc_results = [], []

for idx, val in enumerate(loc_values): 
    loc_smooth_scale = val
    orient_smooth_scale = 1.0
    feet_loss_scale = 1.0

    """all batch allows global coherency SPIN notes"""
    batch_size  = "all"
    print_every, iterations = args.print_every, args.iterations 

    if not bool_params["K_op"]:
        time.sleep(3)
        print("you are not optimizing camera...")

    # -------------------------------------------------
    # update with your username
    username = "username"

    # 1️⃣ Start a new run, tracking config metadata
    wandb.init(project="mirror-aware-human", mode="disabled",
    entity=username, config={
        "loc_smooth_scale": loc_smooth_scale,
        "orient_smooth_scale": orient_smooth_scale,
        "feet_loss_scale": feet_loss_scale,

        "comment": args.inline_com,
        "eval_data": args.eval_data,
        "net_data": args.net_data,
        "alphapose": args.alphapose,
        "det": args.det,
        "view": args.view,
        "gt": args.gt,
        "useGTfocal": args.useGTfocal,
        "use_mapper": args.use_mapper,
        "disable_rot": args.disable_rot,
        "disable_all_rot": args.disable_all_rot,
        "start_zero": args.start_zero,
        "use_mapper": args.use_mapper,
        "disable_bf": args.disable_bf,
        "batch_size": batch_size,
        "bool_params": bool_params,
    })
    config_wb = wandb.config

    # -------------------------------------------------------Initial pose-----------------------------------------------------------------------------
    if args.h36m_data_dir != 'None':
        h36m_subj = ['S1']
        train_dataset = h36m_dataset(folders = h36m_subj, split_flag = 'train set', args=args)
    else:
        train_dataset = None
    real_feets, virt_feets = None, None
    # ---------------------------------------------- Load/run calibration ------------------------------------------------------------
    import pdb
    if os.path.isfile(load_filename):
        if args.comb_set:
            args.load_filename = [load_filename, load_filename2]
            from_pickle_a = load_pickle(load_filename)
            from_pickle_b = load_pickle(load_filename2)
        else:
            args.load_filename = load_filename
            from_pickle = load_pickle(load_filename)

        print(f"***** Using the calibration from {args.load_filename} *****")
        args.recon_dir = recon_dir

        
        if args.eval_data and not args.gt and not args.alphapose and not args.comb_set:
            print("mirror evaluation data set with dcpose")
            ankles, cam_matrix, pseudo_2d_real_select, pseudo_2d_virt_select, normal_ground = from_pickle
            # update image paths
            new_path = f"dataset/zju-m-seq1/images/{view}/"
            update_img_paths(pseudo_2d_real_select, new_path)
            update_img_paths(pseudo_2d_virt_select, new_path)

        elif args.rec_data and args.alphapose and not args.useGTfocal and not args.gt and not args.comb_set:
            print("internet or internal data set with alphapose")
            ankles, cam_matrix, pseudo_2d_real_select = from_pickle["ankles"], from_pickle["cam_matrix"], from_pickle["pseudo_2d_real_select"]
            pseudo_2d_virt_select, normal_ground, chosen_frames = from_pickle["pseudo_2d_virt_select"], from_pickle["normal_ground"], from_pickle["chosen_frames"]
            dropped_frames = from_pickle["dropped_frames"] if "dropped_frames" in from_pickle.keys() else None
            drop_len = len(dropped_frames) if dropped_frames!=None else 0
            print(f"chosen_frames: {len(chosen_frames)} | total frames {len(chosen_frames) + drop_len}")

        elif args.eval_data and args.alphapose and args.useGTfocal and not args.comb_set:
            print("mirror evaluation data set with alphapose and GT focal")
            ankles, cam_matrix, pseudo_2d_real_select = from_pickle["ankles"], from_pickle["cam_matrix"], from_pickle["pseudo_2d_real_select"]
            pseudo_2d_virt_select, normal_ground, chosen_frames = from_pickle["pseudo_2d_virt_select"], from_pickle["normal_ground"], from_pickle["chosen_frames"]
            dropped_frames = from_pickle["dropped_frames"] if "dropped_frames" in from_pickle.keys() else None

        elif args.eval_data and args.gt and args.useGTfocal and not args.comb_set:
            print("mirror evaluation data set with GT2D and GT focal")
            ankles, cam_matrix, pseudo_2d_real_select = from_pickle["ankles"], from_pickle["cam_matrix"], from_pickle["pseudo_2d_real_select"]
            pseudo_2d_virt_select, normal_ground, chosen_frames = from_pickle["pseudo_2d_virt_select"], from_pickle["normal_ground"], from_pickle["chosen_frames"]

        elif args.comb_set:
            print("mirror evaluation data set with combined GT2D and Alphapose")
            # GT2D
            ankles_a, cam_matrix_a, pseudo_2d_real_select_a = from_pickle_a["ankles"], from_pickle_a["cam_matrix"], from_pickle_a["pseudo_2d_real_select"]
            pseudo_2d_virt_select_a, normal_ground_a, chosen_frames_a = from_pickle_a["pseudo_2d_virt_select"], from_pickle_a["normal_ground"], from_pickle_a["chosen_frames"]
            # Alphapose
            ankles_b, cam_matrix_b, pseudo_2d_real_select_b = from_pickle_b["ankles"], from_pickle_b["cam_matrix"], from_pickle_b["pseudo_2d_real_select"]
            pseudo_2d_virt_select_b, normal_ground_b, chosen_frames_b = from_pickle_b["pseudo_2d_virt_select"], from_pickle_b["normal_ground"], from_pickle_b["chosen_frames"]
            # selection
            cam_matrix = cam_matrix_a
            normal_ground = normal_ground_a

        else:
            print("hhmmn")

    # -------------------------------------------------------Video Loader-------------------------------------------------------------------     

    if args.eval_data and args.gt and not args.comb_set:
        video_dataset = video_loader(pseudo_2d_real_select=pseudo_2d_real_select, pseudo_2d_virt_select=pseudo_2d_virt_select, json_path=json_file, chosen_frames=chosen_frames,
        ankles=ankles, real_feets=real_feets, virt_feets=virt_feets, skel_type=skel_type,use_mapper=use_mapper, mirror19=args.mirror19)

    elif args.eval_data and args.alphapose and not args.comb_set:
        #alphapose
        video_dataset = video_loader(pseudo_2d_real_select=pseudo_2d_real_select, pseudo_2d_virt_select=pseudo_2d_virt_select, chosen_frames=chosen_frames,
        dropped_frames=dropped_frames,
        ankles=ankles, real_feets=real_feets, virt_feets=virt_feets, skel_type=skel_type,use_mapper=use_mapper)

    elif args.rec_data and args.alphapose and not args.comb_set:
        #alphapose
        video_dataset = video_loader(pseudo_2d_real_select=pseudo_2d_real_select, pseudo_2d_virt_select=pseudo_2d_virt_select, chosen_frames=chosen_frames,
        dropped_frames=dropped_frames,
        ankles=ankles, real_feets=real_feets, virt_feets=virt_feets, skel_type=skel_type,use_mapper=use_mapper)

    elif args.comb_set:
        """"combined GT2D with feet joints from alphapose detections"""
        list_a = [pseudo_2d_real_select_a, pseudo_2d_virt_select_a, ankles_a, chosen_frames_a]
        list_b = [pseudo_2d_real_select_b, pseudo_2d_virt_select_b, ankles_b, chosen_frames_b]
        comb_set = (list_a, list_b)
        video_dataset = video_loader(use_mapper=use_mapper,comb_set=comb_set, use_alpha_ankles=args.use_alpha_ankles,
                                    alpha_feet_weight=args.alpha_feet_weight)


    indices_train = list(range(0, len(video_dataset)))
    video_set = torch.utils.data.Subset(video_dataset, indices_train)

    if batch_size == "all":
        batch_size = len(video_dataset)

    if args.num_workers !=0:
        print(f"No of cpu workers: {args.num_workers}")
        time.sleep(3)

    # define the dataset loader (batch size, shuffling, ...)
    video_loader_var = torch.utils.data.DataLoader(video_set, batch_size = batch_size, num_workers=int(args.num_workers), pin_memory=False, shuffle=False, drop_last=False)
    
    # -------------------------------------------------------run optimization-------------------------------------------------------------------
    result, uniq_id = new_run_setup(args, config_wb, show_image, clear_output, cam_matrix, video_dataset, video_loader_var,
    batch_size, normal_ground, train_dataset, print_every, iterations, bool_params, identity, recon_dir, rotate_initial_flag, flip_virt_flag,interest_frame, view, skel_type=skel_type,use_mapper=use_mapper, mirror19=args.mirror19)
    
    #------------------------------------------------------------------------------------

end = time.time()
secs = end - start
time_taken = str(datetime.timedelta(seconds=secs))
print(f"time taken: {time_taken}")
