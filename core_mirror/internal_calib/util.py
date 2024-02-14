
import numpy as np
from itertools import tee

eps = 1e-36
def determine_head(tmp, head_ind, lw, rw, head_up_region=50.0, special=False):
    if special:
        head_x = (tmp[head_ind[0]][0] + tmp[head_ind[1]][0]/2.0)
        head_y = (tmp[head_ind[0]][1] + tmp[head_ind[1]][1]/2.0)
        head = (head_x, head_y)
        # use ear1 score
        head_score = tmp[head_ind[0]][2]
    else:
        head = tmp[head_ind]
        head_x, head_y = head[0], head[1]
        head_score = head[2]

    lwrist = tmp[lw]
    rwrist = tmp[rw]
    head_left_bound, head_right_bound = max(head[0]-head_up_region, 0), head[0]+head_up_region
    
    # left hand within bound and above head 
    if head_left_bound < lwrist[0] < head_right_bound and lwrist[1] > head[1]:
        head_x = lwrist[0]
        head_y = lwrist[1]
        head_score = lwrist[2]

    # right hand within bound and above head
    elif head_left_bound < rwrist[0] < head_right_bound and rwrist[1] > head[1]:
        head_x = rwrist[0]
        head_y = rwrist[1]
        head_score = rwrist[2]
    
    elif head_left_bound < lwrist[0] < head_right_bound and lwrist[1] > head[1] \
        and head_left_bound < rwrist[0] < head_right_bound and rwrist[1] > head[1]:
        if lwrist[1] > rwrist[1]:
            head_x = lwrist[0]
            head_y = lwrist[1]
            head_score = lwrist[2]

        else:
            head_x = rwrist[0]
            head_y = rwrist[1]
            head_score = rwrist[2]
        
    # head is higher than both wrists
    else:
        return head_x, head_y, head_score
    return head_x, head_y, head_score


def determine_foot(tmp,rfoot,lfoot, hgt_threshold=80.0, wide_threshold=100.0, flag=None, decide=None, iter=None):

    dist_wide = abs(tmp[rfoot][0] - tmp[lfoot][0])
    dist_height = abs(tmp[rfoot][1] - tmp[lfoot][1])

    if flag == "real" or flag == None:
        if dist_height>hgt_threshold: # height
            if tmp[rfoot][1]> tmp[lfoot][1]:
                ankle_y = tmp[rfoot][1]
                ankle_x = tmp[rfoot][0]
                decide = 0
            else: 
                ankle_y = tmp[lfoot][1]
                ankle_x = tmp[lfoot][0]
                decide = 1
            
        elif dist_wide>wide_threshold: # wide 
            if tmp[rfoot][1] > tmp[lfoot][1]:
                ankle_y = tmp[rfoot][1]
                ankle_x = tmp[rfoot][0]
                decide = 0
            else:
                ankle_y = tmp[lfoot][1]
                ankle_x = tmp[lfoot][0]
                decide = 1

        else:
            # normal midpoint
            ankle_x = ((tmp[rfoot][0] + tmp[lfoot][0])/2.0)
            ankle_y = ((tmp[rfoot][1] + tmp[lfoot][1])/2.0)
            # for feet projection later
            if tmp[rfoot][1] > tmp[lfoot][1]:
                decide = 3
            else:
                decide = 4

    elif flag == "virt" and decide != None:
        if decide==1: 
            ankle_y = tmp[rfoot][1]
            ankle_x = tmp[rfoot][0]
            decide = 0

        elif decide==0: 
            ankle_y = tmp[lfoot][1]
            ankle_x = tmp[lfoot][0]
            decide = 1

        elif decide==3 or decide==4:
            # normal midpoint
            ankle_x = ((tmp[rfoot][0] + tmp[lfoot][0])/2.0)
            ankle_y = ((tmp[rfoot][1] + tmp[lfoot][1])/2.0)
             # for feet projection later
            if decide==3:
                decide=0
            else:
                decide=1

    return ankle_x, ankle_y, decide

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / (np.linalg.norm(vector) + eps)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def head_translate(hu, hv, au, av, percent):
    return [percent*(hu - au) + hu, percent*(hv - av) + hv] #percent*vect + np.array([hu, hv])
def foot_translate(hu, hv, au, av, percent):
    return [-percent*(hu - au) + au, -percent*(hv - av) + av]

def matrix_cosine(x, y):
    return np.einsum('ij,ij->i', x, y) / (
              np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    )

def str2bool(v):
    return v.lower() in ("True", "true", "yes", "t")

def list_frames(num):
    frames = []
    for i in range(0, num):
        frames.append(str(i).zfill(8) + '.jpg')
    return frames
    

def get_3d_coord_horiz_bottom(cam_inv, p_plane, init_point, normal, img_width, img_height):
    plane_horiz_world = plane_ray_intersection(img_width/2.0 + 1, img_height, cam_inv, normal, init_point)
    up_vector = np.array(plane_horiz_world) - np.array(init_point)
    up_vector = up_vector/(np.linalg.norm(up_vector) + eps)
    
    v = up_vector
    u = np.cross(v, normal)
    grid_location_3d_ij = plane_horiz_world + p_plane[0]*v + p_plane[1]*u  
    return grid_location_3d_ij

def plane_ray_intersection(x_imcoord, y_imcoord, cam_inv, normal, init_point):
    ray = np.array((cam_inv @ [x_imcoord, y_imcoord, 1.0]))        
    scale = np.dot(normal, init_point)/np.dot(normal, ray) 
    point_x = np.squeeze(ray[0]*scale)
    point_y = np.squeeze(ray[1]*scale)
    point_z = np.squeeze(ray[2]*scale)
    return np.array([point_x, point_y, point_z])

def plane_ray_intersection_np(x_imcoord, y_imcoord, cam_inv, normal, init_point):
    """init point is the ankleworld
    objective: get 3D point from 2D point and inverse focal length etc"""

    x_imcoord = np.array(x_imcoord)
    y_imcoord = np.array(y_imcoord)
    
    ones = np.ones(x_imcoord.shape[0])
    point_2d = np.stack((x_imcoord, y_imcoord, ones))
    # shoot ray
    ray = np.array((cam_inv @ point_2d))
    # intersect the ray with normal 
    normal_dot_ray = np.transpose(np.array(normal)) @ ray
    scale = np.divide(np.repeat(np.dot(normal, init_point), x_imcoord.shape[0]), normal_dot_ray)

    ray[0] = scale*ray[0]
    ray[1] = scale*ray[1]
    ray[2] = scale*ray[2]
    return ray

def basis_change_rotation_matrix(cam_matrix, cam_inv, init_point, normal, img_width, img_height):
    r = 0
    c = 0
    plane_world = plane_ray_intersection(img_width/2.0, img_height, cam_inv, normal, init_point)
    p00 = get_3d_coord_horiz_bottom(cam_inv, [(r+0),(c+0)], plane_world, normal, img_width, img_height)
    p01 = get_3d_coord_horiz_bottom(cam_inv, [(r+0),(c+1)], plane_world, normal, img_width, img_height)
    p10 = get_3d_coord_horiz_bottom(cam_inv, [(r+1),(c+0)], plane_world, normal, img_width, img_height)
    p11 = get_3d_coord_horiz_bottom(cam_inv, [(r+1),(c+1)], plane_world, normal, img_width, img_height)
    
    new_basis0 = p01 - p00
    new_basis1 = p10 - p00
    new_basis0 = new_basis0/(np.linalg.norm(new_basis0) + eps)
    new_basis1 = new_basis1/(np.linalg.norm(new_basis1) + eps)
    new_basis2 = normal
    
    old_basis0 = np.array([1, 0, 0])
    old_basis1 = np.array([0, 1, 0])
    old_basis2 = np.array([0, 0, 1])
    C = np.zeros([3,3], dtype = float)
    C[0] = [np.dot(new_basis0, old_basis0), np.dot(new_basis0, old_basis1), np.dot(new_basis0, old_basis2)]
    C[1] = [np.dot(new_basis1, old_basis0), np.dot(new_basis1, old_basis1), np.dot(new_basis1, old_basis2)]
    C[2] = [np.dot(new_basis2, old_basis0), np.dot(new_basis2, old_basis1), np.dot(new_basis2, old_basis2)]
    
    ########### DRAW AXIS
    x_basis = [plane_world[0] + new_basis0[0]*10, plane_world[1] + new_basis0[1]*10, plane_world[2] + new_basis0[2]*10]
    y_basis = [plane_world[0] + new_basis1[0]*10, plane_world[1] + new_basis1[1]*10, plane_world[2] + new_basis1[2]*10]
    z_basis = [plane_world[0] - new_basis2[0]*10, plane_world[1] - new_basis2[1]*10, plane_world[2] - new_basis2[2]*10]
    
    z_rotation = np.zeros((3,3))
    z_rotation[0] = [np.cos(np.pi/2.0), -1*np.sin(np.pi/2.0), 0]
    z_rotation[1] = [np.sin(np.pi/2.0), np.cos(np.pi/2.0), 0]
    z_rotation[2] = [0,0,1]
    
    flip = np.zeros((3,3))
    flip[0] = [-1, 0, 0]
    flip[1] = [0, 1, 0]
    flip[2] = [0, 0, 1,]
    return flip @ z_rotation @ C, plane_world, x_basis, y_basis, z_basis

def project_point_horiz_bottom(cam_matrix, cam_inv, p_plane, init_world, normal, img_width, img_height):
    plane_horiz_world = plane_ray_intersection(img_width/2.0 + 1, img_height, cam_inv, normal, init_world)
    up_vector = np.array(plane_horiz_world) - np.array(init_world)
    up_vector = up_vector/(np.linalg.norm(up_vector) + eps)
    v = up_vector
    u = np.cross(v, normal)
    
    # normalize
    v = v/(np.linalg.norm(v) + eps) 
    u = u/(np.linalg.norm(u) + eps)         
    grid_location_3d_ij = init_world + p_plane[0]*v + p_plane[1]*u
    p_cam = np.dot(cam_matrix, np.array(grid_location_3d_ij))

    p_px = p_cam
    if p_cam[2] < 0:
        return [None]
    if p_cam[2] != 0:
        p_px = p_cam/p_cam[2]
    return np.array(p_px[:2])

def find_nearest_point(pt, next_array):
    close_pt = None
    close_dist = np.Inf
    for i in range(len(next_array)):
        dist = np.linalg.norm(np.array(pt) - np.array(next_array[i]))
        
        if close_dist > dist:
            close_dist = dist
            close_pt = np.array(next_array[i])
    return close_pt, close_dist

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)