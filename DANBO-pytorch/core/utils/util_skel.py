'''
Definition - 
Different Dataset - h36m, MPI-3DHP, DCPose, Mirror, SMPL etc
'''

joint_names_alpha_pose = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
    'right_hip','left_knee','right_knee','left_ankle','right_ankle','head', 
    'neck', 'pelvis','left_big_toe', 'right_big_toe', 'left_small_toe','right_small_toe','left_heel', 'right_heel']

# no head
joint_names_mirrored_human = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
    "left_elbow", "left_wrist", "pelvis",
    "right_hip","right_knee","right_ankle","left_hip","left_knee","left_ankle","right_eye","left_eye","right_ear",
    "left_ear","left_big_toe","left_small_toe","left_heel","right_big_toe","right_small_toe","right_heel"]

joint_parents_mirrored_human = [
    "neck", "pelvis", "neck", "right_shoulder", "right_elbow", "neck",
    "left_shoulder", "left_elbow", "pelvis",
    "pelvis","right_hip","right_knee","pelvis","left_hip","left_knee","nose","nose","right_eye",
    "left_eye","left_heel","left_heel","left_ankle","right_heel","right_heel","right_ankle"]


joint_names_mirrored_human_15 = joint_names_mirrored_human[:15]
joint_names_mirrored_human_19 = joint_names_mirrored_human[:19]
joint_names_mirrored_human_25 = joint_names_mirrored_human[:]

mirror19_hip_first_names = ['pelvis', 'right_hip', 'right_knee', 'right_ankle',  'left_hip',
                    'left_knee', 'left_ankle', 'neck', 'nose', 'right_eye', 
                    'right_ear', 'left_eye', 'left_ear', 
                    'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 
                    'right_elbow', 'right_wrist']

# TODO: update variable name
joint_names_alpha_hip_first = ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 
        'right_heel', 'right_big_toe', 'right_small_toe', 'left_hip',
        'left_knee', 'left_ankle',  'left_heel', 'left_big_toe', 
        'left_small_toe', 'neck', 'nose', 'right_eye', 
        'right_ear', 'left_eye', 'left_ear', 'head', 
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 
        'right_elbow', 'right_wrist',
    ]

alpha_hip_first_bones = [('pelvis', 'right_hip'), ('right_hip', 'right_knee'),('right_knee', 'right_ankle'),('right_ankle', 'right_heel'),
        ('right_heel', 'right_big_toe'), ('right_heel', 'right_small_toe'), ('pelvis', 'left_hip'), ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'), ('left_ankle', 'left_heel'), ('left_heel', 'left_big_toe'), ('left_heel', 'left_small_toe'),
        ('pelvis', 'neck'), ('neck', 'nose'), ('nose', 'right_eye'), ('right_eye', 'right_ear'),
        ('nose', 'left_eye'), ('left_eye', 'left_ear'), ('nose', 'head'), ('neck', 'left_shoulder'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist')]

joint_names_common_Alphapose_n_Mirror = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
    "left_elbow", "left_wrist", "pelvis",
    "right_hip","right_knee","right_ankle","left_hip","left_knee","left_ankle","right_eye","left_eye","right_ear",
    "left_ear","left_big_toe","left_small_toe","left_heel","right_big_toe","right_small_toe","right_heel"
]

joint_parents_common_Alphapose_n_Mirror = [
    "neck", "pelvis", "neck", "right_shoulder", "right_elbow", "neck",
    "left_shoulder", "left_elbow", "pelvis",
    "pelvis","right_hip","right_knee","pelvis","left_hip","left_knee","nose","nose","right_eye",
    "left_eye","left_heel","left_heel","left_ankle","right_heel","right_heel","right_ankle"]


mirror_symmetry = [["left_shoulder", "right_shoulder"], 
                                  ["left_elbow", "right_elbow"], ["left_wrist", "right_wrist"],["left_hip","right_hip"]
                                  , ["left_knee", "right_knee"], ["left_ankle", "right_ankle"]]


joint_names_dcpose = [ "nose", "neck", "head", "unknown_joint_1", "pelvis", # "unknown_joint_2",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", 
                "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",  
                "right_ankle"]
              
joint_names_H36M = ["pelvis","right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
               "left_ankle", "spine1", "neck", "nose", "head", "left_shoulder", "left_elbow", 
               "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]


joint_names_common_H36M_DCPose = ["pelvis","nose", "neck", "head", "left_shoulder", "right_shoulder", 
                                  "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def alphapose_to_mirror_common_fn(h_pose):
    pose = h_pose[:, alphapose_to_mirror_25, :]
    return pose

def  h36m_to_mirror_common_fn(h_pose):
    pose = h_pose[:, h36m_to_mirror15, :]
    return pose

def  h36m_to_dc_common_fn(pose):
    pose = pose[:, h36m_to_dc_common, :]
    return pose

def Identity_map(pose):
    return pose 


def flip(joint_names, symmetry):
    joint_names_new = joint_names.copy()
    for pair in symmetry:
        if pair[0] not in joint_names or pair[1] not in joint_names:
            continue
        p0 = joint_names.index(pair[0])
        p1 = joint_names.index(pair[1])
        joint_names_new[p1] = joint_names[p0]
        joint_names_new[p0] = joint_names[p1]
    return joint_names_new

joint_names_common_H36M_DCPose_mirrored = flip(joint_names_common_H36M_DCPose, mirror_symmetry)
#joint_names_common_H36M_DCPose_mirrored = joint_names_common_H36M_DCPose

# h36m
bones_h36m_right = [["pelvis","right_hip"],["right_hip", "right_knee"],["right_knee", "right_ankle"],
                   ["neck","right_shoulder"],["right_shoulder","right_elbow"], ["right_elbow","right_wrist"]]
bones_h36m_left = [["pelvis","left_hip"],["left_hip", "left_knee"],["left_knee", "left_ankle"],
                   ["neck","left_shoulder"],["left_shoulder","left_elbow"], ["left_elbow","left_wrist"]]
bones_h36m_center = [["pelvis","spine1"],["spine1","neck"],["nose","neck"],["nose","head"]]
bones_h36m = bones_h36m_right + bones_h36m_left + bones_h36m_center
bones_h36m_grouped = [bones_h36m_right,bones_h36m_left,bones_h36m_center]
bones_h36m_grouped_indices = [[[joint_names_H36M.index(n1),joint_names_H36M.index(n2)] for (n1,n2) in group] for group in bones_h36m_grouped]

# dc
bones_dc_right = [["right_shoulder","right_hip"],["right_hip", "right_knee"],["right_knee", "right_ankle"],
                   ["neck","right_shoulder"],["right_shoulder","right_elbow"], ["right_elbow","right_wrist"]]
bones_dc_left = [["left_shoulder","left_hip"],["left_hip", "left_knee"],["left_knee", "left_ankle"],
                   ["neck","left_shoulder"],["left_shoulder","left_elbow"], ["left_elbow","left_wrist"]]
bones_dc_center = [["neck","nose"],["nose","head"]]
bones_dc_grouped = [bones_dc_right,bones_dc_left,bones_dc_center]
bones_dc_grouped_indices = [[[joint_names_common_H36M_DCPose.index(n1),joint_names_common_H36M_DCPose.index(n2)] for (n1,n2) in group] for group in bones_dc_grouped]

# mirror
bones_mirror_right = [["pelvis","right_hip"],["right_hip", "right_knee"],["right_knee", "right_ankle"],
                   ["neck","right_shoulder"],["right_shoulder","right_elbow"], ["right_elbow","right_wrist"]]
bones_mirror_left = [["pelvis","left_hip"],["left_hip", "left_knee"],["left_knee", "left_ankle"],
                   ["neck","left_shoulder"],["left_shoulder","left_elbow"], ["left_elbow","left_wrist"]]
bones_mirror_center = [["pelvis","neck"],["neck","nose"]]
bones_mirror = bones_mirror_right + bones_mirror_left + bones_mirror_center
bones_mirror_grouped = [bones_mirror_right,bones_mirror_left,bones_mirror_center]
bones_mirror_grouped_indices = [[[joint_names_mirrored_human.index(n1),joint_names_mirrored_human.index(n2)] for (n1,n2) in group] for group in bones_mirror_grouped]


h36M32_to_H36M17 = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

# mapping functions
'''mother on the left, subset on the right, find the corresponding subset name in mother'''
h36m_to_mirror15 = [joint_names_H36M.index(name) for name in joint_names_mirrored_human_15]
mirror_to_mirror15 = [joint_names_mirrored_human.index(name) for name in joint_names_mirrored_human_15]

h36m_to_dc_common = [joint_names_H36M.index(name) for name in joint_names_common_H36M_DCPose]
dc_to_dc_common = [joint_names_dcpose.index(name) for name in joint_names_common_H36M_DCPose]
dc_flip = [joint_names_common_H36M_DCPose.index(name) for name in joint_names_common_H36M_DCPose_mirrored]

alphapose_to_mirror_25 = [joint_names_alpha_pose.index(name) for name in joint_names_common_Alphapose_n_Mirror]
mirror_to_mirror25 = [joint_names_mirrored_human.index(name) for name in joint_names_common_Alphapose_n_Mirror]
hip_first_to_mirror_25 = [joint_names_alpha_hip_first.index(name) for name in joint_names_common_Alphapose_n_Mirror]

mirror_to_mirror19 = [joint_names_mirrored_human.index(name) for name in joint_names_mirrored_human_19]
