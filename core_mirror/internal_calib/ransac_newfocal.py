
import math
import torch
import numpy as np
import random as rand

import math
import scipy.special
import core_mirror.internal_calib.util as util
import core_mirror.internal_calib.calibration_newfocal as calibration_newfocal
from core_mirror.internal_calib.util import determine_foot, determine_head
dtype = torch.float64

def ransac_best_ankle_np_lstq(datastore, joint_indices, image_index, termination_cond, img_width, img_height, 
                            num_points = 3, threshold = 50, threshold_cos = 0.1, h = 1.6, conf_flag = False,
                             person_list = None, inlier_ratio = 0.1, p_success = 0.99, skel_type=None,
                             useGTfocal=False, true_cam=None):
    
    thorax, lshoulder, rshoulder = joint_indices['thorax'], joint_indices['lshoulder'], joint_indices['rshoulder']
    lhip, rhip, lknee = joint_indices['lhip'], joint_indices['rhip'], joint_indices['lknee']
    rknee, lfoot, rfoot = joint_indices['rknee'], joint_indices['lfoot'], joint_indices['rfoot']
    lw, rw = joint_indices['lw'], joint_indices['rw']
    
    if skel_type == "dcpose":
        head_ind = joint_indices['head_ind']

    elif skel_type == "gt2d":
        ear1 = joint_indices['head_ind']

    if termination_cond is None:
        termination_cond = int(math.log(1.0 - p_success)/math.log(1.0 - inlier_ratio**num_points))
    
    total_comb = int(scipy.special.comb(len(image_index),num_points))
    samples = set()
    tries = 0
    while len(samples) < termination_cond and len(samples) < total_comb:
        samples.add(tuple(sorted(rand.sample(image_index, num_points))))
        tries += 1
        

    point_set = list(samples)
    comb_index = np.array(range(0, len(point_set)))
    global_best_inlier_error = np.inf
    global_best_inlier = 0
    global_best_error = np.inf
    global_best_error_cos = np.inf
    
    best_head_ground_error_3d = np.inf
    total_average_error = np.inf
    total_average_error_cos = np.inf

    error_array = []
    focal_length_array = []
    inlier_array = []
    inlier_head_array = []

    normal_array = []
    z_array = []
    curent_best = None
    persons = None
    t1 = img_width/2.0
    t2 = img_height/2.0
   
    normal_best = None
    depth_Z_best = None
    focal_predicted_best = None
    cam_matrix_best = None
    cam_inv_best = None
    ankleworld_best = None
    ppl3d_x_best = None
    ppl3d_y_best = None
    ppl3d_z_best = None
    
    rank_array = []
    prod_array = []
    
    np.random.shuffle(comb_index)
    for iterater in range(0, len(point_set)):
        randCombo = comb_index[iterater]
        ankle_u = []
        ankle_v = []
        head_u = []
        head_v = []

        if person_list is None:
            persons = np.array(point_set[randCombo])[:num_points] 
        else:
            persons = person_list

        ankles = []
        heads = []
        for ppl in persons:
            ankle_x, ankle_y, _ = determine_foot(datastore["Info"][ppl]["keypoints"],rfoot,lfoot)

            if skel_type == "dcpose":
                    head_x, head_y, _ = determine_head(datastore["Info"][ppl]["keypoints"], head_ind, lw, rw)
            elif skel_type == "gt2d":
                head_x, head_y, _ = determine_head(datastore["Info"][ppl]["keypoints"], ear1, lw, rw, special=False)
                    
            ankle_u.append(ankle_x)
            ankle_v.append(ankle_y)
            head_u.append(head_x)
            head_v.append(head_y)
            
        au = np.array(ankle_u)
        av = np.array(ankle_v)
        aw = np.ones(3)
        hu = np.array(head_u)
        hv = np.array(head_v)
        hw = np.ones(3)

        normal, depth_Z, focal_predicted, cam_matrix = calibration_newfocal.calibration_focalpoint_lstq(num_points, hv, av, hu, au, h, t1, t2, true_cam=true_cam, useGTfocal=useGTfocal)

        if useGTfocal and true_cam is not None:
            cam_matrix = true_cam
            cam_inv = np.linalg.inv(cam_matrix)
            ankleworld = (cam_inv @ [au[0], av[0], aw[0]]) * np.absolute(depth_Z[0])
        else:
            cam_inv = np.linalg.inv(cam_matrix)
            ankleworld = (cam_inv @ [au[0], av[0], aw[0]]) * np.absolute(depth_Z[0])

        if focal_predicted is None:
            continue
        if focal_predicted <= 0.0:
            continue

        if normal[1] < 0: # ENFORCING POSITIVE Y
            normal[0] = -1*normal[0]
            normal[1] = -1*normal[1]
            normal[2] = -1*normal[2]

        #COMPUTE PLANE LINE INTERSECTION
        ppl_u = []
        ppl_v = []
        ppl_w = np.ones(len(image_index))

        ppl_head_u = []
        ppl_head_v = []
        ppl_head_w = np.ones(len(image_index))

        ppl3d_x = []
        ppl3d_y = []
        ppl3d_z = [] 

        inlier_count = 0
        error_list = []
        error_list_cos = []
        inlier_error_list = []
        inlier_error_list_cos = []
        head_ground_error_3d_array = []
        notUpsideDown = True

        if focal_predicted_best is None:
            normal_best = normal
            depth_Z_best = depth_Z
            focal_predicted_best = focal_predicted
            cam_matrix_best = cam_matrix
            cam_inv_best = cam_inv
            ankleworld_best = ankleworld

        ankle1_conf = []
        ankle2_conf = []
        head_conf = []

        for ppl in image_index:
            ankle_x, ankle_y, _ = determine_foot(datastore["Info"][ppl]["keypoints"],rfoot,lfoot)
            if skel_type == "dcpose":
                head_x, head_y, head_score = determine_head(datastore["Info"][ppl]["keypoints"], head_ind, lw, rw)
            elif skel_type == "gt2d":
                head_x, head_y, head_score = determine_head(datastore["Info"][ppl]["keypoints"], ear1, lw, rw, special=False)

            ppl_u.append(ankle_x)
            ppl_v.append(ankle_y)
            ppl_head_u.append(head_x)
            ppl_head_v.append(head_y)
            ankle1_conf.append(datastore["Info"][ppl]["keypoints"][lfoot][2])
            ankle2_conf.append(datastore["Info"][ppl]["keypoints"][rfoot][2])

            if skel_type == "dcpose":
                head_conf.append(head_score)
            elif skel_type == "gt2d":
                # use ear1 score
                head_conf.append(head_score)

        # how you get your Z
        person_world = util.plane_ray_intersection_np(ppl_u, ppl_v, cam_inv, normal, ankleworld)

        ppl3d_x = person_world[0]
        ppl3d_y = person_world[1]
        ppl3d_z = person_world[2]

        ankle_ppl_2d_3comp = cam_matrix @ person_world
        ankle_ppl_2d = np.zeros((2, person_world.shape[1]))

        ankle_ppl_2d[0] = np.divide(ankle_ppl_2d_3comp[0], ankle_ppl_2d_3comp[2])
        ankle_ppl_2d[1] = np.divide(ankle_ppl_2d_3comp[1], ankle_ppl_2d_3comp[2])

        head_ppl_3d = np.array(person_world) - np.transpose(np.tile(normal*h, (person_world.shape[1], 1)))
        head_ppl_2d = np.zeros((2, person_world.shape[1]))

        head_ppl_2d_3comp = cam_matrix @ head_ppl_3d

        head_ppl_2d[0] = np.divide(head_ppl_2d_3comp[0], head_ppl_2d_3comp[2])
        head_ppl_2d[1] = np.divide(head_ppl_2d_3comp[1], head_ppl_2d_3comp[2])

        head_dcpose = np.stack((ppl_head_u, ppl_head_v))
        ankle_dcpose = np.stack((ppl_u, ppl_v))

        head_vect_pred = head_ppl_2d - ankle_ppl_2d
        head_vect_ground = head_dcpose - ankle_dcpose
        error_cos = np.ones(person_world.shape[1]) - util.matrix_cosine(np.transpose(head_vect_pred), np.transpose(head_vect_ground))

        error_norm = np.linalg.norm(head_ppl_2d - head_dcpose, axis=0)/np.linalg.norm(head_vect_ground, axis=0)

        error_list = error_norm
        error_list_cos = error_cos

        error_norm_array = error_norm < threshold 
        error_cos_array = error_cos < threshold_cos
        notUpsideDown = np.array(head_vect_pred[1]) < 0
        selected_persons = np.in1d(image_index, persons)
        inlier_array = (error_norm_array & error_cos_array & notUpsideDown) | selected_persons

        inlier_count = list(inlier_array).count(True)
        if conf_flag:
            average_conf = np.add(np.add(ankle1_conf, ankle2_conf), head_conf)/3.0
            inlier_count =  np.sum(inlier_array.astype(int)*average_conf)

        inlier_index = np.where(inlier_array == True)
        inlier_error_list = error_norm[inlier_index]
        inlier_error_list_cos = error_cos[inlier_index]

        if ppl3d_x_best is None:
            ppl3d_x_best = ppl3d_x 
            ppl3d_y_best = ppl3d_y
            ppl3d_z_best = ppl3d_z

        average_error = np.average(np.array(error_list))
        average_inlier_error = np.average(np.array(inlier_error_list))

        average_error_cos = np.average(np.array(error_list_cos))
        average_inlier_error_cos = np.average(np.array(inlier_error_list_cos))

        error_array.append(average_error)
        focal_length_array.append(focal_predicted)

        normal_array.append(normal)
        z_array.append(depth_Z)

        if global_best_inlier == 0:
                global_best_error = average_inlier_error
                global_best_error_cos = average_inlier_error_cos
                total_average_error = average_error
                total_average_error_cos = average_error_cos

                curent_best = persons
                global_best_inlier = inlier_count

                normal_best = normal
                depth_Z_best = depth_Z
                focal_predicted_best = focal_predicted
                cam_matrix_best = cam_matrix
                cam_inv_best = cam_inv
                ankleworld_best = ankleworld
                ppl3d_x_best = ppl3d_x
                ppl3d_y_best = ppl3d_y
                ppl3d_z_best = ppl3d_z

        if inlier_count > global_best_inlier:
                global_best_error = average_inlier_error
                global_best_error_cos = average_inlier_error_cos
                total_average_error = average_error
                total_average_error_cos = average_error_cos

                curent_best = persons
                global_best_inlier = inlier_count

                normal_best = normal
                depth_Z_best = depth_Z
                focal_predicted_best = focal_predicted
                cam_matrix_best = cam_matrix
                cam_inv_best = cam_inv
                ankleworld_best = ankleworld
                ppl3d_x_best = ppl3d_x
                ppl3d_y_best = ppl3d_y
                ppl3d_z_best = ppl3d_z
        elif inlier_count == global_best_inlier:

            if global_best_error > average_inlier_error and global_best_error_cos > average_inlier_error_cos:
                    
                    global_best_error = average_inlier_error
                    global_best_error_cos = average_inlier_error_cos
                    total_average_error = average_error
                    total_average_error_cos = average_error_cos

                    curent_best = persons
                    global_best_inlier = inlier_count

                    normal_best = normal
                    depth_Z_best = depth_Z
                    focal_predicted_best = focal_predicted
                    cam_matrix_best = cam_matrix
                    cam_inv_best = cam_inv
                    ankleworld_best = ankleworld
                    ppl3d_x_best = ppl3d_x
                    ppl3d_y_best = ppl3d_y
                    ppl3d_z_best = ppl3d_z
                    
    return normal_best, depth_Z_best, focal_predicted_best, cam_matrix_best, cam_inv_best, ankleworld_best, ppl3d_x_best, ppl3d_y_best, ppl3d_z_best, global_best_inlier, curent_best, global_best_error, global_best_error_cos, total_average_error, total_average_error_cos, len(image_index)

