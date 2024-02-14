# Utility script for loading and saving, etc

# standard imports
import json
import ipdb
import torch
import pickle5 as pickle
import numpy as np

def build_A_dash(avg_normal_m, plane_d, device=None):

    n1,n2,n3 = avg_normal_m[:,0:1], avg_normal_m[:,1:2], avg_normal_m[:,2:3];

    A_dash = torch.cat([
    torch.stack([(1 - (2*n1**2)), (-(2*n1*n2)), (-2*n1*n3), (-(2*n1*plane_d))], dim = 1).view(1, 4), 
    torch.stack([-(2*n1*n2), (1 - (2*n2**2)), -(2*n2*n3), -(2*n2*plane_d)], dim = 1).view(1, 4),
    torch.stack([-(2*n1*n3), -(2*n2*n3), (1 - (2*n3**2)), -(2*n3*plane_d)], dim = 1).view(1, 4),
    torch.stack([torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.ones(1)], dim = 1).view(1, 4).to(device),
    ], dim = 0)
    
    A_dash = A_dash.to(device)

    return A_dash


def normalize_batch_normal(n):
    assert len(n.shape)==2, "shape is not 2-dim or size 2"
    eps=1e-36
    a_norm = torch.norm(n, dim =1).view(-1,1)  + eps
    n_m = torch.div(n, a_norm)
    return n_m



def detect_flip(est_2d, skel_type):
    """alpha mirror common - 25
    For outlier detection, using hip is better"""
    
    if skel_type == "alpha":
        assert est_2d.shape[1] == 25, "Format should be alpha mirror common."
        hip_joint = 8
    raise Exception("skeleton not defined.")
    
    joint_vel = (est_2d[1:] - est_2d[:-1])
    joint_vel = joint_vel[:, hip_joint]
    closeness = torch.norm(joint_vel, dim=1)
    return closeness

def get_joint_trees(bones, joint_names, root_id=0):
    joint_trees = np.zeros(shape=len(joint_names))
    root_id = 0
    joint_trees[0] = root_id # hip connected to hip
    for bone in bones:
        joint_trees[joint_names.index(bone[1])] = joint_names.index(bone[0])
    return joint_trees

def calc_bone_length(poses, root_id, joint_trees):
    '''manually calculate 
    - bone length, mean bone length, and std deviation 
      across a sequence of poses
    - assumes hip-first ordering format
    '''

    children_trans = torch.cat([poses[:, :root_id], poses[:, root_id+1:]], dim=1)[..., None]
    parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
    parent_trans = poses[:, parent_ids, :, None]
    bv = children_trans - parent_trans 

    bl = torch.norm(bv, dim=2).squeeze(2)
    mean_bl = bl.mean(0)
    std_bl = bl.std(0)
    return bl, mean_bl, std_bl, bv


def increase_initial(h_pose, init_type, mirror19=False, device=None):
    ''' - create 26 or 19 joints from h36m initial 17 joints
        - Re-order so hip starts first based on Kinematic chain
    args:
        h_pose: (17,3)
        initial: (26,3) or (19,3)
    '''
    # Non-symmetric Pose
    Nose, Leye = h_pose[9], h_pose[9]+torch.tensor([-50,-30,-30])
    Reye, Lear, Rear = h_pose[9]+torch.tensor([50,-30,-30]), h_pose[9]+torch.tensor([-80,0,-80]), h_pose[9]+torch.tensor([80,0,-80])
    Lsh, Rsh = h_pose[11],h_pose[14]
    Lelb, Relb, Lw, Rw = h_pose[12],h_pose[15],h_pose[13],h_pose[16]

    LHip, RHip, Lkn, Rkn = h_pose[4],h_pose[1],h_pose[5],h_pose[2]
    Lank, Rank, Head, Neck = h_pose[6],h_pose[3],h_pose[10],h_pose[8]

    Hip, LBToe, RBToe, LSToe = h_pose[0],h_pose[6]+torch.tensor([20,50,70]),h_pose[3]+torch.tensor([-20,50,70]),h_pose[6]+torch.tensor([-20,50,70])
    RSToe, LHeel, RHeel = h_pose[3]+torch.tensor([20,50,70]), h_pose[6]+torch.tensor([20,50,-30]),h_pose[3]+torch.tensor([20,50,-30])

    # stack 26 joints
    random = torch.stack([Nose, Leye, Reye, Lear, Rear, Lsh, Rsh, Lelb, Relb, Lw, Rw,
                    LHip, RHip, Lkn, Rkn, Lank, Rank, Head, Neck, Hip, LBToe, 
                    RBToe, LSToe, RSToe, LHeel, RHeel]).to(device)

    # re-order so hip starts first (similar to h36m structure)
    initial_26 = alpha_to_hip_1st(random, device=device)
    initial_26 = initial_26.detach().cpu()


    if init_type == "alpha":
        # ----------------------------------------------
        # Symmetric Pose starts from Non-Sym Pose - alphapose26 init
        # ----------------------------------------------

        initial_26_c = initial_26-initial_26[0]
        Hip = initial_26_c[0] +torch.tensor([0,-10,0])

        Head = initial_26_c[19]; Head[0] = Hip[0]
        Neck = initial_26_c[13]; Neck[0] = Hip[0]
        Lsh = initial_26_c[20]
        Lelb = initial_26_c[21]
        Lw = initial_26_c[22]

        Nose = initial_26_c[14]; Nose[0] = Hip[0]
        Leye  = initial_26_c[17]
        Lear = initial_26_c[18]

        LHip = initial_26_c[7]
        Lkn = initial_26_c[8] +torch.tensor([-10,0,0])
        Lank = initial_26_c[9] +torch.tensor([-30,0,0])
        LHeel = initial_26_c[10] +torch.tensor([-30,0,0])
        LBToe = initial_26_c[11] +torch.tensor([-30,0,0])
        LSToe = initial_26_c[12] +torch.tensor([-30,0,0])

        # Flip side of the body
        Reye  = Leye * torch.tensor([-1,1,1]) 
        Rear = Lear * torch.tensor([-1,1,1]) 

        Rsh = Lsh * torch.tensor([-1,1,1]) 
        Relb = Lelb * torch.tensor([-1,1,1]) 
        Rw = Lw * torch.tensor([-1,1,1]) 

        Rank = Lank * torch.tensor([-1,1,1]) 
        RHip = LHip * torch.tensor([-1,1,1]) 
        Rkn = Lkn * torch.tensor([-1,1,1]) 
        RBToe = LBToe * torch.tensor([-1,1,1]) 
        RHeel = LHeel * torch.tensor([-1,1,1]) 
        RSToe = LSToe * torch.tensor([-1,1,1]) 

        j_list = [Hip,RHip,Rkn,Rank,RHeel,RBToe,RSToe,LHip,Lkn,Lank,LHeel,LBToe,LSToe,
                                Neck,Nose,Reye,Rear,Leye,Lear,Head,Lsh,Lelb,Lw,Rsh,Relb,Rw,
                                ]
        hip_first = torch.stack(j_list)

        # new: head-> nose, nose-> neck
        joint_names = ['pelvis','right_hip','right_knee','right_ankle', 
                    'right_heel', 'right_big_toe','right_small_toe', 'left_hip',
                    'left_knee','left_ankle','left_heel', 'left_big_toe', 
                    'left_small_toe','neck','nose', 'right_eye',
                    'right_ear', 'left_eye','left_ear', 'head', 
                    'left_shoulder','left_elbow','left_wrist', 'right_shoulder',
                    'right_elbow', 'right_wrist']

        joint_parents = ['pelvis','pelvis','right_hip','right_knee',
                        'right_ankle', 'right_heel', 'right_heel', 'pelvis',
                        'left_hip','left_knee','left_ankle','left_heel',
                        'left_heel', 'pelvis','neck','nose', 
                        'right_eye','nose','left_eye','nose',
                        'neck','left_shoulder', 'left_elbow','neck',
                        'right_shoulder','right_elbow']


    elif (init_type == "gt2d" and mirror19) or init_type == "comb_set":
        # ----------------------------------------------
        # Sym Pose starts from Non-Sym Pose - mirrorpose19 init
        # ----------------------------------------------
        initial_26_c = initial_26-initial_26[0]
        Hip = initial_26_c[0]

        Lsh = initial_26_c[20];  Lsh[0] = Lsh[0]-20; Lsh[1] = Lsh[1]+30; Hip[2] = Lsh[2]
        Neck[0] = Hip[0]; Neck[2] = Hip[2]; Neck[1] = Lsh[1]

        Lelb = initial_26_c[21]
        Lw = initial_26_c[22]

        Nose = initial_26_c[14]; Nose[0] = Hip[0]; 
        Leye  = initial_26_c[17]
        Lear = initial_26_c[18]

        LHip = initial_26_c[7]; LHip[1] = Hip[1]; LHip[2] = Hip[2]
        Lkn = initial_26_c[8] +torch.tensor([-10,0,0]); Lkn[2] = Hip[2]+40
        Lank = initial_26_c[9] +torch.tensor([-30,0,0]);Lank[2] = Hip[2] +15


        # Flip side of the body
        Reye  = Leye * torch.tensor([-1,1,1]) 
        Rear = Lear * torch.tensor([-1,1,1]) 

        Rsh = Lsh * torch.tensor([-1,1,1]) 
        Relb = Lelb * torch.tensor([-1,1,1]) 
        Rw = Lw * torch.tensor([-1,1,1]) 

        Rank = Lank * torch.tensor([-1,1,1]) 
        RHip = LHip * torch.tensor([-1,1,1])
        Rkn = Lkn * torch.tensor([-1,1,1]) 

        # stack 19 joints
        j_list = [Hip,RHip,Rkn,Rank,LHip,Lkn,Lank,
                                Neck,Nose,Reye,Rear,Leye,Lear,Lsh,Lelb,Lw,Rsh,Relb,Rw,
                                ]

        joint_names= ['pelvis', 'right_hip', 'right_knee', 'right_ankle',  
                    'left_hip', 'left_knee', 'left_ankle', 'neck', 
                    'nose', 'right_eye', 'right_ear', 'left_eye', 
                    'left_ear', 'left_shoulder', 'left_elbow', 'left_wrist', 
                    'right_shoulder', 'right_elbow', 'right_wrist']

        joint_parents = ['pelvis', 'pelvis', 'right_hip', 'right_knee',  
                        'pelvis', 'left_hip', 'left_knee', 'pelvis', 
                        'neck', 'nose', 'right_eye', 'nose', 
                        'left_eye', 'neck','left_shoulder', 'left_elbow', 
                        'neck', 'right_shoulder', 'right_elbow']

    
        if init_type == "comb_set":

            #  new definitions but joints re-used from mirror19 and alpha above
            Head = initial_26_c[19]; Head[0] = Hip[0]

            LHeel = initial_26_c[10] +torch.tensor([-30,0,0])
            LBToe = initial_26_c[11] +torch.tensor([-30,0,0])
            LSToe = initial_26_c[12] +torch.tensor([-30,0,0])

            # Flip joints to other side of the body
            RBToe = LBToe * torch.tensor([-1,1,1]) 
            RHeel = LHeel * torch.tensor([-1,1,1]) 
            RSToe = LSToe * torch.tensor([-1,1,1]) 

            # stack mirror19+6alpha+1head joints
            j_list =    [Hip,RHip,Rkn,Rank, 
                        RHeel,RBToe,RSToe, LHip,
                        Lkn,Lank,LHeel,LBToe,
                        LSToe,Neck,Nose,Reye,
                        Rear,Leye,Lear,Head,
                        Lsh,Lelb,Lw,Rsh,
                        Relb,Rw,
                    ]

            # new: head-> neck, nose-> neck
            joint_names= ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 
                        'right_heel', 'right_big_toe', 'right_small_toe', 'left_hip',
                        'left_knee', 'left_ankle', 'left_heel','left_big_toe', 
                        'left_small_toe', 'neck', 'nose', 'right_eye', 
                        'right_ear', 'left_eye', 'left_ear', 'head',
                        'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 
                        'right_elbow', 'right_wrist']

            joint_parents = ['pelvis', 'pelvis', 'right_hip', 'right_knee', 
                            'right_ankle', 'right_heel', 'right_heel', 'pelvis', 
                            'left_hip', 'left_knee', 'left_ankle','left_heel',
                            'left_heel', 'pelvis', 'neck', 'nose', 
                            'right_eye', 'nose', 'left_eye', 'neck',
                            'neck', 'left_shoulder', 'left_elbow', 'neck',
                            'right_shoulder', 'right_elbow']

        hip_first = torch.stack(j_list).to(device)

    else:
        raise Exception("structure not defined")

    return hip_first, joint_names, joint_parents

def comset26_to_hip_1st(h_pose, hip_1st_names, comb_set_names, device=None, **k_pipe_kwargs):
    '''Re-order comset26 so "hip starts first" based on Kinematic chain'''
    # re-order so hip starts first (similar to h36m structure)

    order = [hip_1st_names.index(name) for name in comb_set_names]
    ordering = torch.tensor(order).to(device) 

    if device.type=='cuda' and not h_pose.is_cuda:
        h_pose = h_pose.to(device) 

    N_J = len(ordering)
    if h_pose.shape[0]==N_J: 
        new_pose = torch.index_select(h_pose, 0, ordering).to(device) 
    elif h_pose.shape[1]== N_J:
        new_pose = torch.index_select(h_pose, 1, ordering).to(device) 
    return new_pose


def hip_1st_to_comset26(h_pose, hip_1st_names, comb_set_names, device=None, **k_pipe_kwargs):
    '''Re-order "hip-first comset26" to comset26 ordering'''
    # re-order so hip starts first (similar to h36m structure)

    order = [hip_1st_names.index(name) for name in comb_set_names]
    ordering = torch.tensor(order).to(device) 

    N_J = len(order)
    if h_pose.shape[0]==N_J: 
        new_pose = torch.index_select(h_pose, 0, ordering).to(device)  
    elif h_pose.shape[1]== N_J:
        new_pose = torch.index_select(h_pose, 1, ordering).to(device)  
    return new_pose

def alpha_to_hip_1st(h_pose, device=None):
    '''Re-order alphapose so "hip starts first" based on Kinematic chain'''
    # re-order so hip starts first (similar to h36m structure)
    ordering = torch.tensor([19,12,14,16,25,21,23,11,13,15,24,20,22,
                                18,0,2,4,1,3,17,5,7,9,6,8,10]).to(device) 

    if device is not None:
        if device.type=='cuda' and not h_pose.is_cuda:
            h_pose = h_pose.to(device) 

    N_J = len(ordering)
    if h_pose.shape[0]==N_J: 
        new_pose = torch.index_select(h_pose, 0, ordering).to(device) 
    elif h_pose.shape[1]== N_J:
        new_pose = torch.index_select(h_pose, 1, ordering).to(device) 

    return new_pose

def hip_1st_to_alpha(h_pose, **k_pipe_kwargs):
    '''Re-order "hip-first 26 skeleton" to standard alphapose ordering'''
    ordering = torch.tensor([14,17,15,18,16,20,23,21,24,22,25,
                            7,1,8,2,9,3,19,13,0,11,5,12,6,10,4]).to(k_pipe_kwargs["device"])
    N_J = len(ordering)
    if h_pose.shape[0]==N_J: 
        new_pose = torch.index_select(h_pose, 0, ordering).to(k_pipe_kwargs["device"])
    elif h_pose.shape[1]== N_J:
        new_pose = torch.index_select(h_pose, 1, ordering).to(k_pipe_kwargs["device"])
    return new_pose


def hip_1st_to_mirror19(h_pose, hip_1st_names, mirror_names, **k_pipe_kwargs):
    '''Re-order "hip-first mirror19" to standard mirror19 ordering'''
    order = [hip_1st_names.index(name) for name in mirror_names]
    ordering = torch.tensor(order).to(k_pipe_kwargs["device"])

    N_J = len(order)
    if h_pose.shape[0]== N_J: #(N_J, 3)
        new_pose = torch.index_select(h_pose, 0, ordering).to(k_pipe_kwargs["device"])
    elif h_pose.shape[1]== N_J: #(B, N_J, 3)
        new_pose = torch.index_select(h_pose, 1, ordering).to(k_pipe_kwargs["device"])

    return new_pose

def mirror19_to_hip_1st(h_pose, mirror_names, hip_1st_names, **k_pipe_kwargs):
    '''Re-order mirror19 to "hip-first mirror19"'''
    order = [mirror_names.index(name) for name in hip_1st_names]
    ordering = torch.tensor(order).to(k_pipe_kwargs["device"])

    N_J = len(order)
    if h_pose.shape[0]==N_J:
        new_pose = torch.index_select(h_pose, 0, ordering).to(k_pipe_kwargs["device"])
    elif h_pose.shape[1]== N_J:
        new_pose = torch.index_select(h_pose, 1, ordering).to(k_pipe_kwargs["device"])
    
    return new_pose
    


def sort_B_via_A(listA, listB):
    '''
    Interest: ListB
    sort list B based on sorting list A'''
    sorted_tuple = sorted(zip(listA, listB)) 
    sorted_B = list(map(lambda x:x[1], sorted_tuple))
    return sorted_B

def update_img_paths(data, new_path):
    ''' In-Place Update
    args:
        data: list of dictionaries with image_path, keypoints etc
        new_path: up till last folder before image name'''

    for dict_ in data:
        dict_["image_path"] = new_path+""+dict_["image_path"].split("/")[-1]

def save2pickle(filename, tuples):
    '''
    args:
        filename: Name to save the resulting pickle to
        tuples: list of tuples to save to pickle file
    '''
    to_pickle = dict(tuples)
    with open(f'{filename}', 'wb') as handle:
        pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    '''
    args:
        filename: saved filename to load pickle from
    '''
    with open(f'{filename}', 'rb') as handle:
        from_pickle = pickle.load(handle)
    return from_pickle


def run_DCPose(video_urls, save_folder)-> None:
    '''Runs off-the-shelf DcPose on list of video urls
    args:
        video_urls: List of video paths
        save_folder: LOcation to save dcpose results (frames and json files)'''

    pass
    print("DcPose Processing is Done.")
