import os
import cv2
import time
import json
import torch
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import core_mirror.internal_calib.ransac_newfocal as ransac_newfocal
import core_mirror.internal_calib.util as util
from core_mirror.util_loading import load_pickle, save2pickle
from core_mirror.internal_calib.util import determine_foot, determine_head

start = time.time()
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Optional argument
parser.add_argument('--useGTfocal', help='use gt 2d + gt focal to reconstruct 3d ankle', action="store_true", default=False)
parser.add_argument('--useGT', help='use gt 2d to reconstruct 3d ankle', action="store_true", default=False)
parser.add_argument('--mirror19', help='use selected 19 kps from 25 mirror joints', action="store_true", default=False)
parser.add_argument('--camera_id', help='camera view')
parser.add_argument('--p_dir', help='project directory')
parser.add_argument('--seq_name', help='sequence id, actor or video name (if using internet or internal data)')
parser.add_argument('--infostamp', help='information stamp to save results to')

parser.add_argument('--pickle_path', help='path to calibration (focal+ground normal) pickle')
parser.add_argument('--ankles_path', help='path to ankles (3D) if not provided in first pickle')
parser.add_argument('--output_dir', help='path to output results')

parser.add_argument('--json_file', help='path to json detection file')
parser.add_argument('--image_directory', help='image directory')
parser.add_argument('--skel_type', help='skeleton type') 
parser.add_argument('--split_already', help='Real vs Virt already sorted', action="store_true", default=False)
# parse
args = parser.parse_args()


def temp_main_house(jsonfile = None, image_directory=None, useGTfocal=False, 
                    camera_id=None, skel_type=None, mirror19=False, split_already=None,  pickle=None):

    import pdb
    intri_file = f"dataset/zju-m-seq1/intri.yml"
    
    if skel_type == "dcpose":
        # thorax, lshoulder, rshoulder = 1, 5, 6
        # lhip, rhip, lknee = 11, 12, 13
        # rknee, lfoot, rfoot = 14, 15, 16
        # head_ind = 2

        print("using dcpose format for alphapose detections\n")
        time.sleep(3)
        # update with Alphapose
        hip = 19
        thorax, lshoulder, rshoulder = 18, 5, 6
        lhip, rhip, lknee = 11, 12, 13
        rknee, lfoot, rfoot = 14, 24, 25
        lw, rw = 9, 10
        head_ind = 17 #2


    elif skel_type == "gt2d" and mirror19:
        hip = 8 
        thorax, lshoulder, rshoulder = 1, 5, 2
        lhip, rhip, lknee = 12, 9, 13
        rknee, lfoot, rfoot = 10, 14, 11
        lw, rw = 7, 4
        ear1 = 17 
        head_ind = ear1

    else:
        print("please what skeleton structure is been used?")
        pdb.set_trace()

    # ease of access
    joint_indices = {"hip":hip, "thorax":thorax, "lshoulder":lshoulder, 
                    "rshoulder":rshoulder, "lhip":lhip, "rhip":rhip, 
                    "lknee":lknee, "rknee":rknee, "lfoot":lfoot, 
                    "rfoot":rfoot, "lw":lw, "rw":rw, "head_ind":head_ind}

    if useGTfocal and camera_id is not None:
        # read intri file
        fs = cv2.FileStorage(intri_file, cv2.FILE_STORAGE_READ)
        true_camera = np.array(fs.getNode(f"K_{camera_id}").mat())

    with open(jsonfile, 'r') as f:
        datastore = json.load(f)
    ext = "jpg"
    filename_list = glob(image_directory + f"/*.{ext}")
    if len(filename_list) == 0:
        ext = "png"
        filename_list = glob(image_directory + f"/*.{ext}")

    image_path_name = ""
    frame_interval = 1 
    all_filename = sorted(filename_list) 
    filename_20 = sorted(filename_list)[0::frame_interval]

    first_image_id = all_filename[0].split("/")[-1][:-4]
    image_name = f"/{first_image_id}"+f".{ext}"
    num_points = 3 

    # average human height assumption (in metres)
    h = 1.6
    
    def select_indices(datastore, filename, calibrate, angle_filter_video, confidence, skel_type=None):
        print("selecting indices")
        
        image_index = []
        confidence = confidence 
        angle_filter_video = angle_filter_video 

        '''objective: once you access the confidence of real ankle, select the 
        corresponding virtual ankle to maintain consistency with corresponding pairs'''

        store_dict = dict()
        for fi, im_name in enumerate(filename):
            cnt = 0
            temp = []

            # we would get only 2 results per image
            for i in range(len(datastore["Info"])):
                img_id_a = int((datastore["Info"][i]["image_path"]).split("/")[-1][:-4])
                img_id_b = int((image_path_name + im_name).split("/")[-1][:-4])
                if img_id_a != img_id_b:
                    continue

                tmp = datastore["Info"][i]["keypoints"]
                tmp = np.array(tmp).reshape(-1,3)

                # Thorax
                ppl_u1 = (tmp[thorax][0])
                ppl_v1 = (tmp[thorax][1])

                # Head
                if skel_type == "dcpose":
                    ppl_u2, ppl_v2, head_score = determine_head(tmp, head_ind, lw, rw)
                elif skel_type == "gt2d":
                    ppl_u2, ppl_v2, head_score = determine_head(tmp, ear1, lw, rw, special=False)

                # L and R Shoulder
                ppl_u5=(tmp[lshoulder][0])
                ppl_v5=(tmp[lshoulder][1])

                ppl_u6=(tmp[rshoulder][0])
                ppl_v6=(tmp[rshoulder][1])

                # LHip
                ppl_u11=(tmp[lhip][0])
                ppl_v11=(tmp[lhip][1])

                # RHip
                ppl_u12=(tmp[rhip][0])
                ppl_v12=(tmp[rhip][1])

                # LKnee
                ppl_u13=(tmp[lknee][0])
                ppl_v13=(tmp[lknee][1])

                # RKnee
                ppl_u14=(tmp[rknee][0])
                ppl_v14=(tmp[rknee][1])

                # LFoot
                ppl_u15=(tmp[lfoot][0])
                ppl_v15=(tmp[lfoot][1])

                # RFoot
                ppl_u16=(tmp[rfoot][0])
                ppl_v16=(tmp[rfoot][1])

                # not explicitly taking the midpoint as ankle position
                ankle_x, ankle_y, _ = determine_foot(tmp,rfoot,lfoot)
                ankle_center = np.array([ankle_x, ankle_y])
                head = np.array([ppl_u2, ppl_v2])
                pose_height = np.linalg.norm(head - ankle_center)

                knee_ankle_left = np.array([ppl_u13, ppl_v13]) - np.array([ppl_u15, ppl_v15])
                knee_ankle_right = np.array([ppl_u14, ppl_v14]) - np.array([ppl_u16, ppl_v16])
                knee_hip_left = np.array([ppl_u13, ppl_v13]) - np.array([ppl_u11, ppl_v11])
     
                knee_hip_right = np.array([ppl_u14, ppl_v14]) - np.array([ppl_u12, ppl_v12])
                knee_angle_left = util.angle_between(knee_ankle_left, knee_hip_left)
                knee_angle_right = util.angle_between(knee_ankle_right, knee_hip_right)

                hip_knee_left = np.array([ppl_u11, ppl_v11]) - np.array([ppl_u13, ppl_v13])
                hip_knee_right = np.array([ppl_u12, ppl_v12]) - np.array([ppl_u14, ppl_v14])
                hip_shoulder_left = np.array([ppl_u11, ppl_v11]) - np.array([ppl_u5, ppl_v5])
                hip_shoulder_right = np.array([ppl_u12, ppl_v12]) - np.array([ppl_u6, ppl_v6])

                hip_angle_left = util.angle_between(hip_knee_left, hip_shoulder_left)
                hip_angle_right = util.angle_between(hip_knee_right, hip_shoulder_right)
                ankle_dist = np.linalg.norm(np.array([ppl_u15, ppl_v15]) - np.array([ppl_u16, ppl_v16]))/pose_height

                neck_angle = util.angle_between(np.array([ppl_u2, ppl_v2]) - np.array([ppl_u1, ppl_v1]), ankle_center - np.array([ppl_u2, ppl_v2]))
                angle_right = np.abs(knee_angle_right - np.pi) + np.abs(hip_angle_right - np.pi)
                angle_left = np.abs(knee_angle_left - np.pi) + np.abs(hip_angle_left - np.pi)
                pose_thresh = np.abs(neck_angle - np.pi) + min([angle_right, angle_left])
                angle = pose_thresh

                # not explicitly taking the midpoint as ankle position
                ankle_x, ankle_y, _ = determine_foot(tmp,rfoot,lfoot)
                if skel_type == "dcpose":
                    head_x, head_y, _ = determine_head(tmp, head_ind, lw, rw)
                elif skel_type == "gt2d":
                    head_x, head_y, _ = determine_head(tmp, ear1, lw, rw, special=False)

                # for calibration use ankle filtering
                if calibrate and angle > angle_filter_video:
                    continue

                '''I assume that real should show up first based on datastore'''
                if calibrate and tmp[rfoot][2] < confidence and tmp[lfoot][2] < confidence and head_score < confidence: # 2 is head       
                    print(f"rfoot {tmp[rfoot][2]} lfoot {tmp[lfoot][2]} head_score {head_score}")
                    continue
                cnt+=1
                image_index.append(i)
                temp.append(i)
                        
            '''save (frame no relative to selected ): (#of detections, indices relative to datastore) relationship'''
            store_dict[fi] = (cnt, temp)

        if calibrate:
            print(f" calibrate?: {calibrate} - how many selected with confidence {confidence} angle_filter_video {angle_filter_video}?", len(image_index))
        else:
            print(f" calibrate?: {calibrate} - how many selected with confidence {confidence} angle_filter_video {angle_filter_video}?", len(image_index))

        return image_index, store_dict

    print("len datastore", len(datastore['Info']))
    image_index_calibrate, store_dict_short = select_indices(datastore, filename_20, True, angle_filter_video = 1.5, confidence= 0.7, skel_type=skel_type)
    image_index, store_dict = select_indices(datastore, all_filename, False, angle_filter_video = np.inf, confidence= 0.0, skel_type=skel_type)

    ###########################################################################################   

    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Ground plane overlay ax1')
    img = mpimg.imread(image_directory + image_name)
    print("img", img.shape)
    imgplot = ax1.imshow(img)

    threshold = None
    threshold_cos = None
    scale = 1.0 

    # 0 means no plot, 1 means 1 plot, 2 means all plots
    drawPlot = 1
    threshold = 0.3 
    threshold_cos = 0.15 
    persons = None
    termination_cond = 20000 

    ####################################################
    print(len(image_index), "total allowed people")
    print(len(image_index_calibrate), "number of people used to calibrate")
    
    # ---------------------------------------------------------------------
    # using the cam matrix to reconstruct 3d ankles for all others
    # ---------------------------------------------------------------------

    if useGTfocal and camera_id is not None:
        print("************* Using the GT focal length -> Find best depth Z *************")
        normal, depth_Z, focal_predicted, cam_matrix, cam_inv, ankleworld, ppl3d_x, ppl3d_y, ppl3d_z, global_best_inlier, curent_best, global_best_error, global_best_error_cos, total_average_error, val, val1 = ransac_newfocal.ransac_best_ankle_np_lstq(datastore, joint_indices, image_index_calibrate, termination_cond, img.shape[1], img.shape[0], num_points,\
            threshold, threshold_cos, h, person_list = persons, skel_type=skel_type, useGTfocal=useGTfocal, true_cam=true_camera)

        print("cam_matrix GT focal", cam_matrix)
        print("ankleworld calculated from GT focal", ankleworld)
        print("depth_Z calculated from GT focal", depth_Z)

        true_cam_inv = np.linalg.inv(true_camera)
        assert(true_cam_inv.shape == cam_inv.shape)
        cam_matrix = true_camera
        cam_inv = true_cam_inv

    elif not useGTfocal:
        print("************* Using predicted focal length and ankleworld from single-view calibration *************")

        if pickle is not None:
            from_pickle = load_pickle(pickle)
        if 'normal' in from_pickle:
            normal = from_pickle["normal"] 
        elif 'ground_normal' in from_pickle:
            normal = from_pickle["ground_normal"] 
        else:
            print("missing ground normal")
            pdb.set_trace()

        if 'cam_matrix' in from_pickle:
            cam_matrix = from_pickle["cam_matrix"]
        elif 'focal' in from_pickle:
            focal = from_pickle["focal"]
            cam_matrix = np.eye(3)
            cam_matrix[0,0] = cam_matrix[1,1] = focal
            # assume principal point to be in image center
            cam_matrix[0,2] = img.shape[1]//2
            cam_matrix[1,2] = img.shape[0]//2
        else:
            print("missing focal length")
            pdb.set_trace()

        if 'depth_Z' in from_pickle:
                depth_Z = from_pickle["depth_Z"]
        else:
            depth_Z = None

        if args.ankles_path == "" or args.ankles_path == None: 
            # no seperate pickle file given for ankles
            if 'ankleworld' in from_pickle:
                anks = from_pickle["ankleworld"] 
            else:
                print("missing ankleworld")
                pdb.set_trace()
            
        # data type
        ankleworld = np.array(anks) if type(anks)==list else anks
        cam_matrix = np.array(cam_matrix) if type(cam_matrix)==list else cam_matrix
        cam_inv = np.linalg.inv(cam_matrix) 
       
        print("cam_matrix from pred", cam_matrix)
        print("ankleworld from pred", ankleworld)
        print("depth_Z from pred", depth_Z)
        
    # --------------------------------------------------------------------------------
    # reconstruct the 3d ankles for all frames (need pred/GT focal, pred normal and a single pred init ankle)
    # --------------------------------------------------------------------------------
    no_detections = list(map(lambda x : x[0],  list(store_dict.values())))
    assert len(all_filename) == len(no_detections), "each frame must have minimum of 1 detection"

    pseudo_2d_real_select = {'Info': []}
    pseudo_2d_virt_select = {'Info': []}
        
    chosen_frames = []   
    dropped_frames = [] 
    real_feets = []
    virt_feets = []

    # Store Foot and Head info for all 
    ppl_u, ppl_v = [], []
    for frame_no, detect_no in enumerate(no_detections):
        '''at this point: each frame has minimum of 1 detection n_detections >= n_frames '''
        # invalid frames are dropped
        if detect_no != 2: 
            dropped_frames.append(frame_no)
            continue

        chosen_frames.append(frame_no)
        datastore_index_1 = store_dict[frame_no][1][0]; 
        datastore_index_2 = store_dict[frame_no][1][1]; 

        d1_store = np.array(datastore["Info"][datastore_index_1]["keypoints"]).reshape(-1,3)
        d2_store = np.array(datastore["Info"][datastore_index_2]["keypoints"]).reshape(-1,3)
        # poses
        pose2d1 = torch.Tensor(datastore["Info"][datastore_index_1]["keypoints"])
        pose2d2 = torch.Tensor(datastore["Info"][datastore_index_2]["keypoints"])
        '''hip-neck norm "sorting" - we do it once here and do it again in dataloader'''
        first_norm = torch.norm(pose2d1[thorax] - pose2d1[hip], dim=-1)
        sec_norm = torch.norm(pose2d2[thorax] - pose2d2[hip], dim=-1)

        if first_norm < sec_norm:
            # poses
            real_pose, virt_pose = pose2d2, pose2d1
            # real ankles
            ankle_x, ankle_y, decide = determine_foot(d2_store,rfoot,lfoot,flag="real", decide=None,iter=frame_no)
            ppl_u.append(ankle_x)
            ppl_v.append(ankle_y)
            real_feets.append(decide)

            if decide == None:
                print("are we in-between decisions?")
                pdb.set_trace()
            # virt ankles
            ankle_x, ankle_y, v_decide = determine_foot(d1_store,rfoot,lfoot, decide,flag="virt", decide=decide,iter=frame_no)
            ppl_u.append(ankle_x)
            ppl_v.append(ankle_y)
            virt_feets.append(v_decide)


        else:
            # poses
            real_pose, virt_pose = pose2d1, pose2d2
            # real ankles
            ankle_x, ankle_y, decide = determine_foot(d1_store,rfoot,lfoot,flag="real", decide=None, iter=frame_no)
            ppl_u.append(ankle_x)
            ppl_v.append(ankle_y)
            real_feets.append(decide)
            # virt ankles
            ankle_x, ankle_y, v_decide = determine_foot(d2_store,rfoot,lfoot, flag="virt", decide=decide,iter=frame_no)
            ppl_u.append(ankle_x)
            ppl_v.append(ankle_y)
            virt_feets.append(v_decide)

        pseudo_2d_real_select['Info'].append({'image_path' : all_filename[frame_no], 'keypoints' : real_pose})
        pseudo_2d_virt_select['Info'].append({'image_path' : all_filename[frame_no], 'keypoints' : virt_pose})

    person_world = util.plane_ray_intersection_np(ppl_u, ppl_v, cam_inv, normal, ankleworld)

    # extract 3d points
    others3d_x, others3d_y, others3d_z = person_world[0], person_world[1], person_world[2]
    ankles = torch.Tensor(np.stack([others3d_x, others3d_y, others3d_z], axis=1))
    print("all ankles", ankles.shape); 
     
    #----------------------------------------------------------------
    # PLOTTINGS HERE
    #----------------------------------------------------------------
    end = time.time()
    print(f"{(end - start):.2f} seconds")
    return ankles, cam_matrix, pseudo_2d_real_select['Info'], pseudo_2d_virt_select['Info'], normal, chosen_frames, dropped_frames, depth_Z, real_feets, virt_feets
           

if __name__ == "__main__":
    view = args.camera_id
    pred_focal_path = args.pickle_path
        
    output = temp_main_house(jsonfile=args.json_file,
    image_directory=args.image_directory, useGTfocal=args.useGTfocal, camera_id=int(args.camera_id), skel_type=args.skel_type,
    mirror19=args.mirror19, split_already=args.split_already, pickle=pred_focal_path)
    ankles, cam_matrix, pseudo_2d_real_select, pseudo_2d_virt_select, normal_ground, chosen_frames, dropped_frames, depth_Z, real_feets, virt_feets = output

    # save to pickle
    to_pickle = dict(ankles=ankles, cam_matrix=cam_matrix, pseudo_2d_real_select=pseudo_2d_real_select, 
            pseudo_2d_virt_select=pseudo_2d_virt_select, normal_ground=normal_ground, 
            chosen_frames=chosen_frames, dropped_frames=dropped_frames, depth_Z=depth_Z, real_feets=real_feets, 
            virt_feets=virt_feets)
    
    tuples = [("ankles",ankles), ("cam_matrix", cam_matrix), ("pseudo_2d_real_select", pseudo_2d_real_select), 
            ("pseudo_2d_virt_select", pseudo_2d_virt_select), ("normal_ground", normal_ground), 
            ("chosen_frames", chosen_frames), ("dropped_frames", dropped_frames), ("depth_Z", depth_Z), ("real_feets", real_feets),
            ("virt_feets", virt_feets)]
    
    # use GT2D
    if args.useGT:
        if args.useGTfocal and args.skel_type == "gt2d":
            day = "26"
            save_filename = args.output_dir + f"/calib_data_with_GT2D_GTfocal_Jan{day}_2023/calib{view}_build_gt2d_gtfocal_20000iter.pickle"
        elif not args.useGTfocal and args.skel_type == "gt2d": 
            save_filename = args.output_dir + f"/calib_data_with_GT2D/calib{view}_build_gt2d_20000iter.pickle"
    
    # alphapose 2D
    if not args.useGT:
        # mirror human eval data set
        if args.useGTfocal and args.skel_type == "dcpose":
            save_filename = args.output_dir + f"/calib_data_alphapose_GTfocal/calib{view}_{args.infostamp}.pickle"
        # internet and internal data set
        if not args.useGTfocal and args.skel_type == "dcpose":
            save_filename = args.output_dir + f"/calib_data_alphapose/calib{view}_{args.seq_name}_{args.infostamp}.pickle"
    
    os.makedirs("/".join(save_filename.split("/")[:-1]), exist_ok=True)
    save2pickle(save_filename, tuples) 
    print(f"saved to {save_filename}\n")
