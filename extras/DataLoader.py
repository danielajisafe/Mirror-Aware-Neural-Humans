# standard import
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import numpy as np
import skimage.io as io
import json, cv2
import torch 
import h5py
import itertools
from operator import itemgetter 

import sys
import ipdb
from typing import List
import numpy as np
from dataclasses import dataclass

# custom imports
import core_mirror.util_skel as skel
from core_mirror.transforms import clean_function
from core_mirror.util_loading import run_DCPose
from core_mirror.transforms import baseline_to_h36m_uniform_3D
from extras.camera import project_point_radial, load_cameras,load_camera_params
from core_mirror.mirror_geometry import generate_p_data, mirror_calibrate, mirror_calibrate_batch, mirror_operation, mirror_operation_batch,  visualize_sim

# Annotation
@dataclass 
class video_loader(Dataset):
    """Video Loader - works for GT poses|  Detection poses | Calibration output
    
    args:
        video_urls: List of video urls of the same person.
        video_frames: List of folder paths to  video frames
        json_paths: List of file paths to  json files
        pseudo_2d_real_select: List of selected real poses, post-calibration
        pseudo_2d_virt_select: List of selected virtual poses, post-calibration
    """	
    
    def __init__(self, video_urls = None, video_frames=None, json_path=None, pseudo_2d_real_select=None, 
                pseudo_2d_virt_select=None, chosen_frames=None, dropped_frames=None,
                ankles=None, real_feets=None, virt_feets=None, save_folder=None, skel_type=None, use_mapper=False, mirror19=False, 
                comb_set=None, use_alpha_ankles=None, alpha_feet_weight=None):
        
        self.video_urls = video_urls
        self.video_frames = video_frames
        self.json_path = json_path
        self.save_folder = save_folder
        self.skel_type = skel_type
        self.use_mapper = use_mapper
 
        if comb_set is None:
            out = self.pre_process(pseudo_2d_real_select,pseudo_2d_virt_select,ankles,real_feets,virt_feets)
            pseudo_real_2dpose, pseudo_virt_2dpose, p, p_dash, image_urls, real_feets, virt_feets = out

            self.image_urls = image_urls
            self.p, self.p_dash = p, p_dash
            self.real_feets = real_feets
            self.virt_feets = virt_feets
    
            "this is done once, and not during optimization"
            # correct switched poses (expects raw alpha or mirror19 poses)
            pseudo_real_2dpose, pseudo_virt_2dpose, p, p_dash = clean_function(pseudo_real_2dpose, pseudo_virt_2dpose, p, p_dash, skel_type, mirror19)

        else:
            """seta: GT2D, setb: feet alphapose_detections """
            list_a, list_b = comb_set 
            pseudo_2d_real_select_a, pseudo_2d_virt_select_a, ankles_a, chosen_frames_a = list_a 
            pseudo_2d_real_select_b, pseudo_2d_virt_select_b, ankles_b, chosen_frames_b = list_b 

            out_a = self.pre_process(pseudo_2d_real_select_a, pseudo_2d_virt_select_a, ankles_a)
            out_b = self.pre_process(pseudo_2d_real_select_b, pseudo_2d_virt_select_b, ankles_b)

            pseudo_real_2dpose_a, pseudo_virt_2dpose_a, p_a, p_dash_a, image_urls_a, _, _ = out_a
            pseudo_real_2dpose_b, pseudo_virt_2dpose_b, p_b, p_dash_b, _, _, _ = out_b

            # find the common subset between a and b
            common_frames = list(set(chosen_frames_a).intersection(set(chosen_frames_b)))

            c_set = set(common_frames)
            chosen_index_a = [i for i, val in enumerate(chosen_frames_a) if val in c_set]
            chosen_index_b = [i for i, val in enumerate(chosen_frames_b) if val in c_set]
            chosen_frames = common_frames

            # mappings
            pseudo_real_2dpose_a, pseudo_virt_2dpose_a = pseudo_real_2dpose_a[chosen_index_a], pseudo_virt_2dpose_a[chosen_index_a]
            pseudo_real_2dpose_b, pseudo_virt_2dpose_b = pseudo_real_2dpose_b[chosen_index_b], pseudo_virt_2dpose_b[chosen_index_b]

            p_a, p_dash_a = p_a[chosen_index_a], p_dash_a[chosen_index_a]
            p_b, p_dash_b = p_b[chosen_index_b], p_dash_b[chosen_index_b]
            image_urls_a = np.array(image_urls_a)[chosen_index_a]

            
            if use_alpha_ankles:
                self.p, self.p_dash = p_b, p_dash_b
            else:
                self.p, self.p_dash = p_a, p_dash_a
            self.image_urls = image_urls_a#.tolist()
            # correct switched poses/assignments for detections
            pseudo_real_2dpose_b, pseudo_virt_2dpose_b, p_b, p_dash_b = clean_function(pseudo_real_2dpose_b, pseudo_virt_2dpose_b, p_b, p_dash_b, skel_type="alpha", mirror19=False)


        self.chosen_frames = chosen_frames

        if use_mapper:
            # still in raw format (not hip-first)
            if skel_type == "dcpose":
                self.real_common = pseudo_real_2dpose[:, skel.dc_to_dc_common ,0:2] # drop confidence score
                self.virt_common = pseudo_virt_2dpose[:, skel.dc_to_dc_common ,0:2] # drop confidence score
                self.c_score1_common = pseudo_real_2dpose[:, skel.dc_to_dc_common ,2:3]  
                self.c_score2_common = pseudo_virt_2dpose[:, skel.dc_to_dc_common ,2:3]  

            elif skel_type == "gt2d" and mirror19 and comb_set == None:
                # Mirror eval dataset authors initialized their GT annotation effort with (HRNET and OpenPose) detections, hence confidence was included
                # annotation was used for comparison in stage 2, not for the volumetric model (stage 3)
                self.real_common = pseudo_real_2dpose[:, skel.mirror_to_mirror19 ,0:2] # drop confidence score
                self.virt_common = pseudo_virt_2dpose[:, skel.mirror_to_mirror19 ,0:2] # drop confidence score
                self.c_score1_common = pseudo_real_2dpose[:, skel.mirror_to_mirror19 ,2:3]  
                self.c_score2_common = pseudo_virt_2dpose[:, skel.mirror_to_mirror19 ,2:3]  

            elif skel_type == "alpha" and comb_set == None:
                # now in common format
                self.real_common = pseudo_real_2dpose[:, skel.alphapose_to_mirror_25, 0:2]  # drop confidence score
                self.virt_common = pseudo_virt_2dpose[:, skel.alphapose_to_mirror_25, 0:2]  # drop confidence score
                self.c_score1_common = pseudo_real_2dpose[:, skel.alphapose_to_mirror_25, 2:3]  
                self.c_score2_common = pseudo_virt_2dpose[:,skel.alphapose_to_mirror_25, 2:3]  

            elif comb_set != None:
                print("stacking gt2D (19 joints) with 6 feet joints from alphapose detections")

                self.real_common_a = pseudo_real_2dpose_a[:, skel.mirror_to_mirror19 ,0:2] 
                self.virt_common_a = pseudo_virt_2dpose_a[:, skel.mirror_to_mirror19 ,0:2] 
                self.c_score1_common_a = pseudo_real_2dpose_a[:, skel.mirror_to_mirror19 ,2:3] 
                self.c_score2_common_a = pseudo_virt_2dpose_a[:, skel.mirror_to_mirror19 ,2:3] 

                self.real_common_b = pseudo_real_2dpose_b[:, skel.alphapose_to_mirror_25, 0:2]
                self.virt_common_b = pseudo_virt_2dpose_b[:, skel.alphapose_to_mirror_25, 0:2]
                self.c_score1_common_b = pseudo_real_2dpose_b[:, skel.alphapose_to_mirror_25, 2:3]
                self.c_score2_common_b = pseudo_virt_2dpose_b[:,skel.alphapose_to_mirror_25, 2:3]

                self.real_common_a = pseudo_real_2dpose_a[:, skel.mirror_to_mirror19 ,0:2] 
                self.virt_common_a = pseudo_virt_2dpose_a[:, skel.mirror_to_mirror19 ,0:2] 
                self.c_score1_common_a = pseudo_real_2dpose_a[:, skel.mirror_to_mirror19 ,2:3] 
                self.c_score2_common_a = pseudo_virt_2dpose_a[:, skel.mirror_to_mirror19 ,2:3] 

                self.real_common_b = pseudo_real_2dpose_b[:, skel.alphapose_to_mirror_25, 0:2]
                self.virt_common_b = pseudo_virt_2dpose_b[:, skel.alphapose_to_mirror_25, 0:2]
                self.c_score1_common_b = pseudo_real_2dpose_b[:, skel.alphapose_to_mirror_25, 2:3]
                self.c_score2_common_b = pseudo_virt_2dpose_b[:,skel.alphapose_to_mirror_25, 2:3]

                # base zero
                B = len(self.real_common_a)
                self.real_common = torch.zeros(B, 26, 2)  
                self.virt_common = torch.zeros(B, 26, 2) 
                self.c_score1_common = torch.zeros(B, 26, 1)  
                self.c_score2_common = torch.zeros(B, 26, 1)  

                # base: alphapose, update with GT2D 19 joints
                self.real_common[:, :19] += self.real_common_a[:, :19]
                self.virt_common[:, :19] += self.virt_common_a[:, :19]
                self.c_score1_common[:, :19] += self.c_score1_common_a[:, :19]
                self.c_score2_common[:, :19] += self.c_score2_common_a[:, :19]

                # update with alphapose feet joints
                self.real_common[:, 19:25] += self.real_common_b[:, 19:]
                self.virt_common[:, 19:25] += self.virt_common_b [:, 19:]

                if alpha_feet_weight !=None:
                    
                    self.c_score1_common[:, 19:25] += self.c_score1_common_b[:, 19:]*alpha_feet_weight
                    self.c_score2_common[:, 19:25] += self.c_score2_common_b[:, 19:]*alpha_feet_weight
                else:
                    self.c_score1_common[:, 19:25] += self.c_score1_common_b[:, 19:]
                    self.c_score2_common[:, 19:25] += self.c_score2_common_b[:, 19:]

                # update with alphapose head joint
                raw_alpha_head_idx = 17
                self.real_common[:, 25:26] += pseudo_real_2dpose_b[:, [raw_alpha_head_idx], 0:2]
                self.virt_common[:, 25:26] += pseudo_virt_2dpose_b[:, [raw_alpha_head_idx], 0:2]
                self.c_score1_common[:, 25:26] += pseudo_real_2dpose_b[:, [raw_alpha_head_idx], 2:3]
                self.c_score2_common[:, 25:26] += pseudo_virt_2dpose_b[:, [raw_alpha_head_idx], 2:3]

        self.length = len(self.real_common)
        print(".. done loading {}".format("video data"))

    def pre_process(self, pseudo_2d_real,pseudo_2d_virt,ankles,real_feets=None,virt_feets=None):
        # start from calibration file (use detections)
        if pseudo_2d_real is not None and pseudo_2d_virt is not None:

            # Images
            image_list = list(map(lambda x: x['image_path'], pseudo_2d_real)); 
            frames = [i for i in range(len(pseudo_2d_real))]
            image_urls = itemgetter(*frames)(image_list)

            '''both pseudo contain the info for the final ankles'''
            # Training setup
            pseudo_real_2dpose = torch.cat(list(map(lambda x: x['keypoints'].unsqueeze(0), pseudo_2d_real)), axis=0)[frames]
            pseudo_virt_2dpose = torch.cat(list(map(lambda x: x['keypoints'].unsqueeze(0), pseudo_2d_virt)), axis=0)[frames]  
            

        if ankles is not None:
            # calibration code does real, virtual, real, etc sequential order.
            p, p_dash = ankles[0::2][frames], ankles[1::2][frames]

        if real_feets != None and virt_feets != None:
            real_feets = torch.Tensor(real_feets)
            virt_feets = torch.Tensor(virt_feets)

        N = len(pseudo_real_2dpose)
        if len(pseudo_real_2dpose.shape) < 3:
            pseudo_real_2dpose = pseudo_real_2dpose.reshape(N,-1, 3)
            pseudo_virt_2dpose = pseudo_virt_2dpose.reshape(N,-1, 3)

        return pseudo_real_2dpose, pseudo_virt_2dpose, p, p_dash, image_urls, real_feets, virt_feets#, frames


    def __getitem__(self, idx):

        if self.use_mapper:
            output = {"real2d": self.real_common[idx],
                    "virt2d": self.virt_common[idx],
                    "image_urls": self.image_urls[idx],
                    "p": self.p[idx],
                    "p_dash": self.p_dash[idx],
                    "c_score1": self.c_score1_common[idx],
                    "c_score2": self.c_score2_common[idx], 
                    "chosen_frames" : self.chosen_frames[idx],
                    }
        return output

    def __len__(self):
        return self.length

    
class h36m_dataset(Dataset): 

    def __init__(self, folders = {}, split_flag = None, args=None):
        
        self.rcams = load_cameras(f"{args.h36m_data_dir}/metadata.xml") 
        self.main_url = args.h36m_data_dir 
        cam_path = f"{args.h36m_data_dir}/human36m-camera-parameters/" 
        
        cam_ids = {0:"54138969", 1:"55011271", 2:"58860488", 3:"60457274"}
        self.proj_mat = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.K_store = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.R_store = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        
        # new distortion (d) formulation
        self.R_d = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.T_d = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.f_d = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.c_d = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.k_d = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.p_d = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
         
        self.pose2d_siblings = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.pose3d_siblings = {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        self.ref_imgs_n_siblings =  {'54138969':[], '55011271':[], '58860488':[], '60457274':[]}
        
        self.ref_idx = [] 
        for subject in folders: 

            annot_path = self.main_url + '/{}/*/*.h5'.format(subject) 
            annots = sorted(glob(annot_path))

            for annot_per_action in annots:
                h5f= h5py.File(annot_per_action, 'r')

                # Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
                joint_indices1 = [0,1,2,3,6,7,8,12,13,14] # 17 joints
                joint_indices2 = [15,17,18,19,25,26,27]

#                 H36M_NAMES = ['']*32
#                 H36M_NAMES[0]  = 'Hip'
#                 H36M_NAMES[1]  = 'RHip'
#                 H36M_NAMES[2]  = 'RKnee'
#                 H36M_NAMES[3]  = 'RFoot'
#                 H36M_NAMES[6]  = 'LHip'
#                 H36M_NAMES[7]  = 'LKnee'
#                 H36M_NAMES[8]  = 'LFoot'
#                 H36M_NAMES[12] = 'Spine'
#                 H36M_NAMES[13] = 'Thorax' # included
#                 H36M_NAMES[14] = 'Neck/Nose' #included
#                 H36M_NAMES[15] = 'Head'
#                 H36M_NAMES[17] = 'LShoulder'
#                 H36M_NAMES[18] = 'LElbow'
#                 H36M_NAMES[19] = 'LWrist'
#                 H36M_NAMES[25] = 'RShoulder'
#                 H36M_NAMES[26] = 'RElbow'
#                 H36M_NAMES[27] = 'RWrist'
    
                
                subj_action_imgs = sorted(glob(self.main_url + '/{}/{}/*/*/*.jpg'.format(subject, annot_per_action.split('/')[-2])))
                if subject == "S7" and annot_per_action.split('/')[-2] == "Waiting-1":
                    subj_action_imgs = subj_action_imgs[:-5]
                    
                save_2d = torch.cat([torch.Tensor(h5f['pose']['2d'][:,joint_indices1]), torch.Tensor(h5f['pose']['2d'][:,joint_indices2])], dim=1)
                save_3d = torch.cat([torch.Tensor(h5f['pose']['3d'][:,joint_indices1]), torch.Tensor(h5f['pose']['3d'][:,joint_indices2])], dim=1)
                   
                len_save = len(save_2d) 
                save_imgs = subj_action_imgs 
            
                # get no of images per camera
                n_imgs_per_cam = len_save/4 # e.g 500
                n_imgs_per_cam = int(n_imgs_per_cam)

                # Build Projection Matrices
                with open(cam_path +'camera-parameters.json') as f:
                    d = json.load(f)
                    lis = [0,1,2,3]
                                     
                    ''' TASK: Refactor data loader ideology to exclude post-dataloader scatter function '''
                    i=0
                    # drop an id
                    new_lis = sorted(list(set(lis) - set([i])))
                    # moving reference camera, fixed storage
                    cnt = self.calc_Params(i, d,subject, annot_per_action, cam_ids[i], self.proj_mat[cam_ids[i]], self.K_store[cam_ids[i]], self.R_store[cam_ids[i]], self.R_d[cam_ids[i]], self.T_d[cam_ids[i]], self.f_d[cam_ids[i]],
                                          self.c_d[cam_ids[i]],self.k_d[cam_ids[i]],self.p_d[cam_ids[i]])


                    '''Overall we use ref id to get siblings pose, siblings projection matrix and siblings imgs'''
                    self.pose2d_siblings[cam_ids[i]].append(save_2d[(i*n_imgs_per_cam):(i+1)*n_imgs_per_cam])
                    self.pose3d_siblings[cam_ids[i]].append(save_3d[(i*n_imgs_per_cam):(i+1)*n_imgs_per_cam])
                    self.ref_imgs_n_siblings[cam_ids[i]].append(save_imgs[(i*n_imgs_per_cam):(i+1)*n_imgs_per_cam])

                    # store reference id
                    self.ref_idx = self.ref_idx + np.tile(i,(cnt)).tolist()

                    # in-parallel  |cam0|cam1|cam2|cam3|
                    if 0 in new_lis:
                        _ = self.calc_Params(0, d,subject, annot_per_action, cam_ids[0], self.proj_mat[cam_ids[0]], self.K_store[cam_ids[0]], self.R_store[cam_ids[0]], self.R_d[cam_ids[0]], self.T_d[cam_ids[0]], self.f_d[cam_ids[0]],
                                          self.c_d[cam_ids[0]],self.k_d[cam_ids[0]],self.p_d[cam_ids[0]])

                        #save e.g 500 poses for sibling cam 
                        self.pose2d_siblings[cam_ids[0]].append(save_2d[(0*n_imgs_per_cam):(0+1)*n_imgs_per_cam])
                        self.pose3d_siblings[cam_ids[0]].append(save_3d[(0*n_imgs_per_cam):(0+1)*n_imgs_per_cam])
                        self.ref_imgs_n_siblings[cam_ids[0]].append(save_imgs[(0*n_imgs_per_cam):(0+1)*n_imgs_per_cam])

                    if 1 in new_lis:
                        _ = self.calc_Params(1, d,subject, annot_per_action, cam_ids[1], self.proj_mat[cam_ids[1]], self.K_store[cam_ids[1]], self.R_store[cam_ids[1]], self.R_d[cam_ids[1]], self.T_d[cam_ids[1]], self.f_d[cam_ids[1]],
                                          self.c_d[cam_ids[1]],self.k_d[cam_ids[1]],self.p_d[cam_ids[1]])

                        #save e.g 500 poses for sibling cam 
                        self.pose2d_siblings[cam_ids[1]].append(save_2d[(1*n_imgs_per_cam):(1+1)*n_imgs_per_cam])
                        self.pose3d_siblings[cam_ids[1]].append(save_3d[(1*n_imgs_per_cam):(1+1)*n_imgs_per_cam])
                        self.ref_imgs_n_siblings[cam_ids[1]].append(save_imgs[(1*n_imgs_per_cam):(1+1)*n_imgs_per_cam])

                    if 2 in new_lis:
                        _ = self.calc_Params(2, d,subject, annot_per_action, cam_ids[2], self.proj_mat[cam_ids[2]], self.K_store[cam_ids[2]], self.R_store[cam_ids[2]], self.R_d[cam_ids[2]], self.T_d[cam_ids[2]], self.f_d[cam_ids[2]],
                                          self.c_d[cam_ids[2]],self.k_d[cam_ids[2]],self.p_d[cam_ids[2]])

                        #save e.g 500 poses for sibling cam 
                        self.pose2d_siblings[cam_ids[2]].append(save_2d[(2*n_imgs_per_cam):(2+1)*n_imgs_per_cam])
                        self.pose3d_siblings[cam_ids[2]].append(save_3d[(2*n_imgs_per_cam):(2+1)*n_imgs_per_cam])
                        self.ref_imgs_n_siblings[cam_ids[2]].append(save_imgs[(2*n_imgs_per_cam):(2+1)*n_imgs_per_cam])

                    if 3 in new_lis:
                        _ = self.calc_Params(3, d,subject, annot_per_action, cam_ids[3], self.proj_mat[cam_ids[3]], self.K_store[cam_ids[3]], self.R_store[cam_ids[3]], self.R_d[cam_ids[3]], self.T_d[cam_ids[3]], self.f_d[cam_ids[3]],
                                          self.c_d[cam_ids[3]],self.k_d[cam_ids[3]],self.p_d[cam_ids[3]])

                        #save e.g 500 poses for sibling cam 
                        self.pose2d_siblings[cam_ids[3]].append(save_2d[(3*n_imgs_per_cam):(3+1)*n_imgs_per_cam])
                        self.pose3d_siblings[cam_ids[3]].append(save_3d[(3*n_imgs_per_cam):(3+1)*n_imgs_per_cam])
                        self.ref_imgs_n_siblings[cam_ids[3]].append(save_imgs[(3*n_imgs_per_cam):(3+1)*n_imgs_per_cam])

        # stack tensors together 
        self.proj_mat = {k: torch.cat(v) for k,v in self.proj_mat.items()}
        self.K_store = {k: torch.cat(v) for k,v in self.K_store.items()}
        self.R_store = {k: torch.cat(v) for k,v in self.R_store.items()}
        
        # distortion params
        self.R_d = {k: torch.cat(v) for k,v in self.R_d.items()}
        self.T_d = {k: torch.cat(v) for k,v in self.T_d.items()}
        self.f_d = {k: torch.cat(v) for k,v in self.f_d.items()}
        self.c_d = {k: torch.cat(v) for k,v in self.c_d.items()}
        self.k_d = {k: torch.cat(v) for k,v in self.k_d.items()}
        self.p_d = {k: torch.cat(v) for k,v in self.p_d.items()}
        
        self.pose2d_siblings = {k: torch.cat(v) for k,v in self.pose2d_siblings.items()}
        self.pose3d_siblings = {k: torch.cat(v) for k,v in self.pose3d_siblings.items()}
       
        self.ref_imgs_n_siblings = {k: list(itertools.chain.from_iterable(v)) for k,v in self.ref_imgs_n_siblings.items()}

        self._length = len(self.ref_imgs_n_siblings['54138969'])
        self.transform = transforms.Compose([transforms.ToPILImage(),
                         transforms.CenterCrop((1000, 1000)),
                          transforms.ToTensor(),
                        ])
        
    def calc_Params(self,n_cam, d, subject,annot_per_action,cam_id,proj_mat, K_store, R_store, R_d_store, T_d_store,
                   f_d_store, c_d_store, k_d_store, p_d_store):
        
        # calibration matrix
        K = d["intrinsics"][cam_id]["calibration_matrix"] # (3,3)

        R = d["extrinsics"][subject][cam_id]["R"]
        t = d["extrinsics"][subject][cam_id]["t"]

        #To build projection matrix P simply multiply calibration K and extrinsics matricies
        '''Already in camera cords safe to exclude R and t'''
        P = K #@ np.hstack([R, t])
        '''to pad the size from (3,3)to (3,4) add 0,0,1'''
        P = np.hstack([np.array(P), np.array([[0],[0],[0]])])

        #------------Store K, R & T for siblings ----------------
        K = d["intrinsics"][cam_id]["calibration_matrix"] # (3,3)
        R = d["extrinsics"][subject][cam_id]["R"]
        t = d["extrinsics"][subject][cam_id]["t"]
        
        #  distortion
        S = int(subject[1:]) 
        R_d, T_d, f_d, c_d, k_d, p_d, cam_str = self.rcams[(S,n_cam+1)]
        
        #To build projection matrix P simply multiply calibration K and extrinsics matrices
        R_4x4 = np.vstack((np.hstack([R, t]), np.array([0,0,0,1]))) # (4x4)
        K_3x4 = np.hstack([K, np.array([[0],[0],[0]])])
 
        #------------------------------
        #Populate the P matrix
        cam_images = sorted(glob(self.main_url + '/{}/{}/*/{}/*.jpg'.format(subject, annot_per_action.split('/')[-2], cam_id)))
        # fixing a special issue here
        if subject == "S7" and annot_per_action.split('/')[-2] == "Waiting-1" and cam_id == "60457274":
            cam_images = cam_images[:-5]

        P_ = np.tile(P,(len(cam_images), 1,1))
        proj_mat.append(torch.Tensor(P_))
        
        K_ = np.tile(K_3x4,(len(cam_images), 1,1))
        R_ = np.tile(R_4x4,(len(cam_images), 1,1))
        
        #distortion params R_d, T_d, f_d, c_d, k_d, p_d
        R_d_ = np.tile(R_d,(len(cam_images), 1,1))
        T_d_ = np.tile(T_d,(len(cam_images), 1,1))
        f_d_ = np.tile(f_d,(len(cam_images), 1,1))
        c_d_ = np.tile(c_d,(len(cam_images), 1,1))
        k_d_ = np.tile(k_d,(len(cam_images), 1,1))
        p_d_ = np.tile(p_d,(len(cam_images), 1,1))
        
        
        K_store.append(torch.Tensor(K_))
        R_store.append(torch.Tensor(R_))
        
        R_d_store.append(torch.Tensor(R_d_))
        T_d_store.append(torch.Tensor(T_d_))
        f_d_store.append(torch.Tensor(f_d_))
        c_d_store.append(torch.Tensor(c_d_))
        k_d_store.append(torch.Tensor(k_d_))
        p_d_store.append(torch.Tensor(p_d_))
    
        
        return len(cam_images)


    def __getitem__(self, idx):
        print("idx investigation", idx)
         
        ref = self.ref_idx[idx] 
        # index out ; no need to re-arrange anymore (we are using fixed camera 0 as reference) 
        '''therefore note that K0 for different poses in a batch might come from diff camera views'''
        # distortion params
        R_d = [v[idx] for k,v in self.R_d.items()] 
        T_d = [v[idx] for k,v in self.T_d.items()]  
        f_d = [v[idx] for k,v in self.f_d.items()]  
        c_d = [v[idx] for k,v in self.c_d.items()]
        k_d = [v[idx] for k,v in self.k_d.items()] 
        p_d = [v[idx] for k,v in self.p_d.items()]  
        
        pose2d_siblings = [v[idx,:] for k,v in self.pose2d_siblings.items()]; 
        pose3d_siblings = [v[idx,:] for k,v in self.pose3d_siblings.items()]; 
        ref_img_urls_n_siblings = [v[idx] for k,v in self.ref_imgs_n_siblings.items()]
        ref_imgs_n_siblings = [self.transform(io.imread(v[idx])).permute(1,2,0) for k,v in self.ref_imgs_n_siblings.items()]
        
        frames = {            
            'ref_id': ref,
            'pose2d_siblings': pose2d_siblings,
            'pose3d_siblings': pose3d_siblings,
            'ref_img_urls_n_siblings': ref_img_urls_n_siblings,
            'ref_imgs_n_siblings': ref_imgs_n_siblings,
            'get_idx': idx,
            
            # distortion params
            'R_d': R_d,
            'T_d': T_d,
            'f_d': f_d,
            'c_d': c_d,
            'k_d': k_d,
            'p_d': p_d,
            } 

        return frames
    
    def __len__(self):
        return self._length







    
    
    
