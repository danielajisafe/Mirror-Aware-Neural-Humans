

import pdb
import json
import torch
import numpy as np
from glob import glob
from tqdm import tqdm, trange

# custom imports
from extras.utils import rotate_hip

def drop_bystanders_non_eval(seq_name=None, file_dir=None, detect_type=None):
    '''Internet and Internal data: exclude all by-standers in the scene,
    by keeping only the top 2 persons with the longest hip-to-neck distance'''
    
    if seq_name==None and detect_type==None:
        detection_path = f"{file_dir}/Subj3_"
        frame_dir = 'visualai/images/'
    else:
        detection_path = f"{file_dir}/{seq_name}/detections/{detect_type}/"
        frame_dir = f'{file_dir}/{seq_name}/frames/'

    with open(f'{detection_path}alphap2dcpose_with_bystanders.json', 'r') as f:
        detect_with_bystanders = json.load(f)
        
    storehouse = {'Info': []}
    data = detect_with_bystanders['Info']

    alpha_root_id = 19
    alpha_neck_id = 18
    
    uniq_image_paths = list(sorted(set(map(lambda x: x['image_path'], data))))
    
    for img_path in tqdm(uniq_image_paths): 

        img_extract_with_bys = list(filter(lambda x: (img_path == x['image_path']), data))
        
        # drop outside by-standers/passers-by or indoor passers-by for internet video
        if len(img_extract_with_bys) > 2:
            kps_with_bys = list(map(lambda x: x['keypoints'], img_extract_with_bys))
            kps_with_bys = np.array(kps_with_bys)
            kps_with_bys = kps_with_bys.reshape(-1,26,3)

            root = kps_with_bys[:,alpha_root_id,:][:,:2]
            neck = kps_with_bys[:,alpha_neck_id,:][:,:2]

            dist = np.sqrt(np.sum((root-neck)**2, axis=1))

            # top k with longest hip-to-neck distance
            k=2
            min_idxs = np.argpartition(dist, -k)[-k:]

            for min_id in min_idxs:
                '''Notes: goal is to drop by-standers here, but clash/wrong assignment is possible for top 2 persons left'''

                temp = {}
                img_id = int(img_extract_with_bys[min_id]['image_path'].split('/')[-1][:-4])
                img = f"{img_id:08d}"
                ext = img_extract_with_bys[min_id]['image_path'].split('/')[-1][-4:]
                temp['image_path'] = frame_dir + img + ext

                temp['keypoints'] = kps_with_bys[min_id].tolist()
                storehouse['Info'].append(temp)
                
        else:
           # simple store even if detect is 1 or 2
            storehouse['Info'] = storehouse['Info'] + img_extract_with_bys
            
            
    out_file = f'{detection_path}alphap2dcpose.json'
    with open(out_file, 'w') as f:
        json.dump(storehouse, f)
    print(f"by-standers are dropped and data saved to {out_file}\n")



def drop_bystanders_mirr_eval(cam_id, filepath=None):
    '''MirrorEval Dataset (video 2-7): using the provided mirror data (now combined), exclude all by-standers in the scene'''
    
    # all frame anotations combined into single json per camera
    with open(f'dataset/zju-m-seq1/annots/combined/{cam_id}/{cam_id}.json', 'r') as f:
        mirr_GT2D = json.load(f)
    
    if filepath==None:
        filepath = f"dataset/authors_eval_data/alphapose_data/{cam_id}_detect/"
        frame_dir = f"dataset/authors_eval_data/images/frames/{cam_id}_mp4/"
    else:
        filepath =f"{filepath}/Cam3_"
        frame_dir = f"dataset/zju-m-seq1/images/{cam_id}/"

    with open(f'{filepath}alphap2dcpose_with_bystanders.json', 'r') as f:
        alphap2dcpose_with_bystanders = json.load(f)

    storehouse = {'Info': []}
    data = mirr_GT2D['Info']
    detect = alphap2dcpose_with_bystanders['Info']

    mirr_root_id = 8
    alpha_root_id = 19
    uniq_image_paths = list(sorted(set(map(lambda x: x['image_path'], data))))

    for img_path in tqdm(uniq_image_paths): 

        # extract image and its dictionary content
        img_extract_mirror = list(filter(lambda x: (img_path == x['image_path'].replace('', '')), data))
        img_extract_with_bys = list(filter(lambda x: (img_path == x['image_path'].replace('', '')), detect))

        kps_with_bys_ = list(map(lambda x: x['keypoints'], img_extract_with_bys))
        kps_with_bys_ = np.array(kps_with_bys_)
        kps_with_bys = kps_with_bys_.reshape(-1,26,3)
        root_with_bys = kps_with_bys[:,alpha_root_id,:][:,:2]

        for i in range(len(img_extract_mirror)):
            temp = {}
            kps = np.array(img_extract_mirror[i]['keypoints'])[:,:2]               
            root = kps[mirr_root_id:mirr_root_id+1]

            '''Notes: goal is to drop by-standers, clash/wrong assignment is possible 
            for top 2 persons left'''
            
            # l2-norm
            dist = np.sqrt(np.sum((root-root_with_bys)**2, axis=1))
            min_id = dist.argmin()
            
            img_id = int(img_extract_with_bys[min_id]['image_path'].split('/')[-1][:-4])
            img = f"{img_id:08d}"
            ext = img_extract_with_bys[min_id]['image_path'].split('/')[-1][-4:]
            temp['image_path'] = frame_dir + img + ext

            temp['keypoints'] = kps_with_bys[min_id].tolist()
            storehouse['Info'].append(temp)


        # special case 
        # if only 1 top person exists, add the next closest person
        if len(img_extract_mirror)==1:
            #ipdb.set_trace()
            temp = {}
            
            temp_idxs = [i for i in range(len(dist))]
            temp_idxs.remove(min_id)
            temp_idxs = np.array(temp_idxs)
            
            dist_left = dist[temp_idxs]
            # track the indices of distances 
            min_val = dist_left.min()
            min_id = np.where(dist == min_val)
            min_id = int(min_id[0]) if len(min_id)>0 else int(min_id)
            
            # drop the min_id
            # calc. argmin and use this as closest next person
            
            img_id = int(img_extract_with_bys[min_id]['image_path'].split('/')[-1][:-4])
            img = f"{img_id:08d}"
            ext = img_extract_with_bys[min_id]['image_path'].split('/')[-1][-4:]
            temp['image_path'] = frame_dir + img + ext

            temp['keypoints'] = kps_with_bys[min_id].tolist()
            storehouse['Info'].append(temp)

        elif len(img_extract_mirror)>2:
            pdb.set_trace()

    out_file = f'{filepath}alphap2dcpose.json'
    with open(out_file, 'w') as f:
        json.dump(storehouse, f)
    print(f"by-standers are dropped and data saved to {out_file}\n")



def alpha_detections_to_dcpose_form_non_eval(seq_name=None, file_dir=None, detect_type=None):
    '''Internet and Internal data: goal is to re-arrange the contents of the json file to match dcpose content structure. 
    Note:  - the kps still remains Alphapose 26 keypoints (Not dcpose)
           - no by-standers'''

    if seq_name==None and detect_type==None:
        detection_path = f"{file_dir}/Subj3_"
        frame_dir = 'visualai/images/Subj3/'
    else:
        detection_path = f"{file_dir}/{seq_name}/detections/{detect_type}/"
        frame_dir = f'{file_dir}/{seq_name}/frames/'

    with open(f'{detection_path}alphapose-results.json', 'r') as f:
        alpharaw = json.load(f)
        
    storehouse = {'Info': []}
    data = alpharaw

    uniq_image_paths = list(sorted(set(map(lambda x: x['image_id'], data))))
    for img_path in tqdm(uniq_image_paths): 

                # extract image and its dictionary content
                img_extract = list(filter(lambda x: (img_path == x['image_id']), data))

                for i in range(len(img_extract)):
                    temp = {}
                    kps = np.array(img_extract[i]['keypoints']).T.reshape(-1,3)
                    kps = kps.tolist()

                    img_id = int(img_extract[i]['image_id'][:-4])
                    ext = img_extract[i]['image_id'][-4:]
                    img = f"{img_id:08d}"

                    temp['image_path'] = frame_dir + img + ext
                    temp['keypoints'] = kps
                    storehouse['Info'].append(temp)

    # no by-standers
    out_file = f'{detection_path}alphap2dcpose_with_bystanders.json'
    with open(out_file, 'w') as f:
        json.dump(storehouse, f)
    print(f"converted to dcpose structure and data saved to {detection_path}alphap2dcpose_with_bystanders.json")

def alpha_detections_to_dcpose_form_mirr_eval(cam_id=None, filepath=None):
    '''MirrorEval Dataset (video 2-7): re-arrange the contents of the json file to match dcpose content structure. 
    Note: the kps still remains Alphapose 26 keypoints (Not dcpose)'''

    if filepath!=None:
        json_file =f"{filepath}/Cam3_alphapose-results.json"
        frame_dir = f"dataset/zju-m-seq1/images/{cam_id}/"
    else:
        filepath = f"dataset/authors_eval_data/alphapose_data/{cam_id}_detect"
        json_file = f'{filepath}/alphapose-results.json'
        frame_dir = f'dataset/authors_eval_data/images/frames/{cam_id}_mp4/'

    with open(f'{json_file}', 'r') as f:
        alpharaw = json.load(f)
        
    storehouse = {'Info': []}
    data = alpharaw

    uniq_image_paths = list(sorted(set(map(lambda x: x['image_id'], data))))
    for img_path in tqdm(uniq_image_paths): 

                # extract image and its dictionary content
                img_extract = list(filter(lambda x: (img_path == x['image_id']), data))

                for i in range(len(img_extract)):
                    temp = {}
                    kps = np.array(img_extract[i]['keypoints']).T.reshape(-1,3)
                    kps = kps.tolist()

                    img_id = int(img_extract[i]['image_id'][:-4])
                    ext = img_extract[i]['image_id'][-4:]
                    img = f"{img_id:08d}"

                    temp['image_path'] = frame_dir + img + ext
                    temp['keypoints'] = kps
                    storehouse['Info'].append(temp)

    json_file_with_bystanders = json_file.replace("alphapose-results", "alphap2dcpose_with_bystanders")
    out_file = f'{json_file_with_bystanders}'

    with open(out_file, 'w') as f:
        json.dump(storehouse, f)
    print(f"converted to dcpose structure and data saved to {out_file}")
    

def clean_function(pseudo_real_2dpose, pseudo_virt_2dpose, p, p_dash, skel_type, mirror19=False):
    """correct flipped virt and real in raw format"""
    # drop confidence
    pseudo_real = pseudo_real_2dpose[:,:,:2]
    pseudo_virt = pseudo_virt_2dpose[:,:,:2]

    if skel_type=="alpha": # raw alphapose
        neck,hip = 18,19
    elif skel_type=="gt2d" and mirror19: # raw mirror19
        neck,hip = 1,8
    elif skel_type=="comb_set": #raw combset
        neck,hip = 1,8

    B, N_J,_ = pseudo_real.shape
    # pose
    real_norm = torch.norm(pseudo_real[:, neck] - pseudo_real[:, hip], dim=-1)
    virt_norm = torch.norm(pseudo_virt[:, neck] - pseudo_virt[:, hip], dim=-1)
    
    real_pose_seq = list(map(lambda x,y,a,b: b if (x<y) else a, real_norm, virt_norm, pseudo_real_2dpose, pseudo_virt_2dpose))
    virt_pose_seq = list(map(lambda x,y,a,b: b if (y<x) else a, real_norm, virt_norm, pseudo_real_2dpose, pseudo_virt_2dpose))

    pseudo_real_2dpose_up = torch.stack(real_pose_seq)
    pseudo_virt_2dpose_up = torch.stack(virt_pose_seq)

    # ankles
    real_p_seq = list(map(lambda x,y,a,b: b if (x<y) else a, real_norm, virt_norm, p, p_dash))
    virt_p_dash_seq = list(map(lambda x,y,a,b: b if (y<x) else a, real_norm, virt_norm, p, p_dash))

    p_up = torch.stack(real_p_seq)
    p_dash_up = torch.stack(virt_p_dash_seq)

    return pseudo_real_2dpose_up, pseudo_virt_2dpose_up, p_up, p_dash_up
    

def rotate_initial_pose(theta, where, degree=None, **k_pipe_kwargs):
    '''Rotate initial pose for where it does not align well with current detection'''
    out = list(map(lambda x,y, deg: rotate_hip(x.unsqueeze(0), deg, **k_pipe_kwargs).squeeze(0) if y==False else x, theta, where, degree))
    theta = torch.stack(out)
    return theta 


def project_pose_batch(pose3d, cord_transform, cam, fewer=False, **k_pipe_kwargs):
    '''Apply coordinate/rigid-body transformation and project from 3D to 2D
    args:
        pose3d shape: (B,17,3)
        cord_transform: (4,4)
        cam: (3,3)
    return:
        posed3d_: (B,17,3)
    '''
    
    if fewer == True: # double check
        import ipdb; ipdb.set_trace()
    
    B = pose3d.shape[0]; 
    
    batch_zeros = torch.zeros(B, 3, 1).to(k_pipe_kwargs["device"])
    K_3x4 = torch.cat([cam, batch_zeros], dim=2); 
    P_3x4 = torch.bmm(K_3x4, cord_transform); 
    pose3d_homo = torch.cat((pose3d, pose3d.new_ones(1).expand(*pose3d.shape[:-1], 1)), 2)

    other_view = torch.bmm(cord_transform, pose3d_homo.permute(0,2,1)); 
    other_view = other_view[:,0:3, :].permute(0,2,1)
    
    # tranform and project to image plane
    proj2d_homo = torch.bmm(P_3x4, pose3d_homo.permute(0,2,1)).permute(0,2,1); 
    # normalize projection with depth Z
    proj2d_ = torch.div(proj2d_homo[:,:,:2],  proj2d_homo[:,:,2:3]) ; 

    return proj2d_, other_view, cord_transform


def refine_g_normal(batch_size, p, p_dash, n_m, n_g, K):
    
    B = batch_size
    
    # ---------------- ground, normal, otho 2d plots------------------
    '''position of the mirror: midpoint of p and pdash'''
    N = (p + p_dash)/2 
    '''plotting the mirror normal from the mirror position'''
    normal_end = N + n_m

    '''project to 2d - uses batched dot product'''
    N_2d = torch.bmm(K, N.view(B, 3, 1)); # -> (B, 3, 1)

    '''dividing by depth - the precision is slightly different in the 2nd decimal'''
    N_2d = torch.div(N_2d, N_2d[:, 2:3]); # -> (B, 3, 1)


    normal_end_2d =  torch.bmm(K, normal_end.view(B, 3, 1)) 
    normal_end_2d = torch.div(normal_end_2d, normal_end_2d[:, 2:3])

    ground_end = N + n_g
    ground_end_2d = torch.bmm(K, ground_end.view(B, 3, 1)) 
    ground_end_2d = torch.div(ground_end_2d, ground_end_2d[:, 2:3]) 

    '''torch keeps at 4-point precision, numpy allows more'''
    otho = torch.cross(n_m, n_g); 

    '''N point, n_m vector/direction'''
    otho_end = N + otho
    otho_end_2d = torch.bmm(K, otho_end.view(B, 3, 1)) 
    otho_end_2d = torch.div(otho_end_2d, otho_end_2d[:, 2:3]) 

    ''' n_m and n_g would not necessarily be othorgonal, use the otho between them
    to refine n_g which we are not super sure of
    note: order matters, the reverse flips every x,y,z component by -1'''
    refine_ground = torch.cross(otho, n_m);

    '''In math: N is point, n_m is a vector/direction'''
    refine_ground_end = N + refine_ground
    refine_ground_end_2d = torch.bmm(K, refine_ground_end.view(B, 3, 1)) 
    refine_ground_end_2d = torch.div(refine_ground_end_2d, refine_ground_end_2d[:, 2:3]) 

    '''Update ground normal estimate'''
    n_g = refine_ground
    return n_g, (N_2d, normal_end_2d, ground_end_2d, otho_end_2d, refine_ground_end_2d)
    # --------------------------------------------------------------


def h36m_to_DCPose(h_pose, flip=False):
    '''map (3d reconstructions or 2d reprojections) in h36m indices to DCPose indices 
    
    Args:
    h_pose: (B, 17, ..)
    
    return:
    dc_pose: (B, 15, ..)
    '''
    '''2 joints in h36m needs to be removed to match up with DCPose
        - exclude hip (0) and spine (7) from h36m'''
    
    indices = [i for i in range(17) if i not in [0, 7]]
    pose_select = h_pose[:, indices, :]
    
    '''re-order joints in h36m to correspond with DCPose - (RH-LH)'''
    ordering = torch.tensor([7,6,8,9,12,10,13,11,14,3,0,4,1,5,2]) # H36m to DCPose
    dc_pose = torch.index_select(pose_select, 1, ordering)

    return dc_pose


def flip_dcpose(pose):
    '''flip left joints to right and vice-versa - dccommon-15 format'''
        
    mirror_pose = torch.zeros_like(pose)
    
    # joint_symmetry_dc = [[0,0],[7,7],[8,8],[9,9],[1,4],[2,5],[3,6],[10,13],[11,14],[12,15]]
    joint_symmetry_dc = [[0,0],[1,1],[2,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]] #for DCPose
    # skip the lhip and rhip bones
    # joint_symmetry_dc = [[0,0],[1,1],[2,2],[3,4],[5,6],[7,8],[9,9],[10,10],[11,12],[13,14]] #for DCPose
    
    for pair in joint_symmetry_dc:
        mirror_pose[:, pair[0]] = pose[:, pair[1]]
        mirror_pose[:, pair[1]] = pose[:, pair[0]]
        

    return mirror_pose

def flip_mirrorpose(pose):
    '''flip left joints to right and vice-versa - mirror15 format'''
        
    mirror_pose = torch.zeros_like(pose)
    joint_symmetry_dc = [[0,0],[1,1],[8,8],[5,2],[6,3],[7,4],[12,9],[13,10],[14,11]]

    for pair in joint_symmetry_dc:
        mirror_pose[:, pair[0]] = pose[:, pair[1]]
        mirror_pose[:, pair[1]] = pose[:, pair[0]]
        
    return mirror_pose

# TODO: Make the flip functions more general, using only names, no magic nos
# Identify flip symmetry automatically
def flip_mirror19(pose):
    "In common mirror19 format"
    mirror_pose = torch.zeros_like(pose)
    joint_symmetry_dc = [[0,0],[1,1],[8,8],[5,2],[6,3],[7,4],[12,9],[13,10],[14,11],
                         [16,15],[18,17]]

    for pair in joint_symmetry_dc:
        mirror_pose[:, pair[0]] = pose[:, pair[1]]
        mirror_pose[:, pair[1]] = pose[:, pair[0]]
    return mirror_pose

def flip_combset26(pose):
    "In common26 format"

    pelvis,rhip,lhip,rkn = 8,9,12,10
    lkn,rank,lank,rheel = 13,11,14,24
    lheel,rBtoe,lBtoe,rStoe = 21,22,19,23
    lStoe,neck,nose,reye = 20,1,0,15,
    leye,rear,lear,head = 16,17,18,25
    rsh,lsh,relb,lelb = 2,5,3,6
    rwrist,lwrist = 4,7

    mirror_pose = torch.zeros_like(pose)
    joint_symmetry_dc = [[pelvis,pelvis],[rhip,lhip],[rkn,lkn],[rank,lank],
                         [rheel,lheel],[rBtoe,lBtoe],[rStoe,lStoe],[neck,neck],
                         [nose,nose],[reye,leye],[rear,lear],[head,head],
                         [rsh,lsh],[relb,lelb],[rwrist,lwrist]]

    for pair in joint_symmetry_dc:
        mirror_pose[:, pair[0]] = pose[:, pair[1]]
        mirror_pose[:, pair[1]] = pose[:, pair[0]]
    return mirror_pose

def flip_alphapose(pose, use_mapper=False):
    '''flip left joints to right and vice-versa - alphapose format'''
        
    mirror_pose = torch.zeros_like(pose)
    if use_mapper:
        # alpha now in mirror common
        joint_symmetry_dc = [[0,0],[1,1],[8,8],[15,16],[17,18],[2,5],[3,6],[4,7],[9,12],
                            [10,13],[11,14],[24,21],[23,20],[22,19]]
    else:
        # alpha still in raw state
        joint_symmetry_dc = [[0,0],[17,17],[18,18],[19,19],[1,2],[3,4],[5,6],[7,8],[9,10],
                            [11,12],[13,14],[15,16],[24,25],[20,21],[22,23]]

    for pair in joint_symmetry_dc:
        mirror_pose[:, pair[0]] = pose[:, pair[1]]
        mirror_pose[:, pair[1]] = pose[:, pair[0]]
        
    return mirror_pose
    

def flip_h36m(pose):

    '''flip left joints to right and vice-versa - h36m format - is it left or right first? confirm :) '''
        
    mirror_pose = torch.zeros_like(pose)
    #joint_symmetry_dc = [[0,0],[7,7],[8,8],[9,9],[1,4],[2,5],[3,6],[10,13],[11,14],[12,15]]
    joint_symmetry_dc = [[0,0],[1,4],[2,5],[3,6],[14,11],[15,12],[16,13],[7,7],[8,8],[9,9],[10,10]] #for DCPose
    #skip the lhip and rhip bones
    #joint_symmetry_dc = [[0,0],[1,1],[2,2],[3,4],[5,6],[7,8],[9,9],[10,10],[11,12],[13,14]] #for DCPose
    
    for pair in joint_symmetry_dc:
        mirror_pose[:, pair[0]] = pose[:, pair[1]]
        mirror_pose[:, pair[1]] = pose[:, pair[0]]
        
    return mirror_pose


def flip(pose):

    '''flip left joints to right and vice-versa - Uniform setup'''
        
    mirror_pose = torch.zeros_like(pose)
    joint_symmetry_dc = [[0,0],[7,7],[8,8],[9,9],[1,4],[2,5],[3,6],[10,13],[11,14],[12,15]]
    #joint_symmetry_dc = [[0,0],[1,1],[2,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]] for DCPose
    
    for pair in joint_symmetry_dc:
        mirror_pose[:, pair[0]] = pose[:, pair[1]]
        mirror_pose[:, pair[1]] = pose[:, pair[0]]

    return mirror_pose
    
    
def baseline_to_h36m_uniform_3D(b_pose, flip=False):
    '''map 3d pose in baseline indices to h36m indices
    for (15 joints) evaluation purposes
    
    Args:
    b_pose: (B, 25, 4)
    
    return:
    h_pose: (B, 16, 3)
    '''
    # drop unrequired joints, empty joints and accuracy column
    head = (b_pose[:, 17:18, :] + b_pose[:, 18:19, :])/2
    b_pose = torch.cat([b_pose[:, :15, :], head], dim=1); 
    b_pose = b_pose[:, :, 0:3]
    
    '''re-order joints to correspond with h36m'''

    # head_missing = 15 # setting it to the nose or 15 would be right eye,16 left eye)
    neck_with_offset = 1 # note the definition of neck differes, but this is the best we can do
    '''head included'''
    ordering = torch.tensor([8,9,10,11,12,13,14,neck_with_offset,0,15,5,6,7,2,3,4]) # baseline ordering to H36m
    #ordering = torch.tensor([8,9,10,11,12,13,14,neck_with_offset,0,head_missing,5,6,7,2,3,4]) #baseline ordering to H36m
    h_pose_nospine = torch.index_select(b_pose, 1, ordering)

    return h_pose_nospine
    
    
    
    
    
def baseline_to_h36m_uniform_2D(b_pose, flip=False):
    '''map 3d pose in baseline indices to h36m indices
    for (15 joints) evaluation purposes
    
    Args:
        b_pose: (B, 25, 3)
    return:
        h_pose: (B, 16, 2)
    '''
    
    # drop unrequired joints, empty joints and accuracy column
    head = (b_pose[:, 17:18, :] + b_pose[:, 18:19, :])/2; 
    b_pose = torch.cat([b_pose[:, :15, :], head], dim=1)
    b_pose = b_pose[:, :, 0:2]
    
    '''re-order joints to correspond with h36m'''

    #head_missing = 15 # setting it to the nose or 15 would be right eye,16 left eye)
    neck_with_offset = 1 # note the definition of neck difference
    '''head included'''
    ordering = torch.tensor([8,9,10,11,12,13,14,neck_with_offset,0,15,5,6,7,2,3,4]) #baseline ordering to H36m
    #ordering = torch.tensor([8,9,10,11,12,13,14,neck_with_offset,0,head_missing,5,6,7,2,3,4]) #baseline ordering to H36m
    h_pose_nospine = torch.index_select(b_pose, 1, ordering)
    
    return h_pose_nospine