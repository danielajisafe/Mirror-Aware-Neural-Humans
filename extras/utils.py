from re import L
import torch
import numpy as np
import torch.nn.functional as F
#import pytorch3d.transforms.rotation_conversions as p3dr

def rot_to_axisang(rot):
    return p3dr.matrix_to_axis_angle(rot)

def to_degree(rad):
    deg = (rad*180)/torch.pi
    return sum(deg)

def rot_mat_2_axis_angle(matrix):
    """custom function"""
    # ref: https://www.mathworks.com/help/robotics/ref/rotm2axang.html#description

    n = len(matrix)
    for i in range(n):
        pre = np.zeros(3)
        pre[i] = 1
        if all(matrix[i] == pre):
            return pre
        

def constrain_rot(R):
    ''' Confirm that R*R^T=I holds for the rotation matrix R'''
    res = R@torch.transpose(R, dim0=2,dim1=3)
    return res

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotate_hip(theta, degree=None, **k_pipe_kwargs):
    """
    rotate skeleton by rotating the hip by 180 degree
    Use 3x3 homogneous matrix to perform 2D rotation about Y axis i.e keep Y, negate other axes
    ref: https://stackoverflow.com/questions/3455285/what-is-the-3-%C3%97-3-matrix-for-a-rotation-of-180-degrees

    Input:
        (B,J,6) Batch of 6-D rotation representations
    Output:
        (B,J,6) Batch of 6-D rotation representations
    """

    B,J,_ = theta.shape; #print("theta", theta.shape)
    
    '''Dan: formula is for rotation about the y-axis'''
    if degree is not None:
        if type(degree) == int:
            rot_180_matrix = rotation_formula(np.radians(degree)).view(1,1,3,3).to(k_pipe_kwargs["device"])
        else: # List of degrees
            pass
    else:
        # rotate by 180
        rot_180_matrix = torch.Tensor([[-1,0,0], [0,1,0], [0,0,-1]]).view(1,1,3,3).to(k_pipe_kwargs["device"])


    rot_mat = rot6d_to_rotmat(theta.view(-1,6)).view(B,J,3,3) 
    others = [i for i in range(J) if i!=0]; 

    hip = rot_mat[:,0:1,:,:] @ rot_180_matrix
    rot_mat = torch.cat([hip, rot_mat[:,others,:,:]], dim=1) 

    # dropped the last row
    rot6d = rot_mat[..., :3, :2].reshape(B, J, 6)
    return rot6d

def rotation_formula(radian):
    """produce rotation matrix given radian value from a degree [about y-axis?]
    The axis of rotation in R3 is the line which remains unchanged during rotation.
    ref:https://math.stackexchange.com/a/744743
    """

    rot_formula = np.round(np.array([[np.cos(radian), 0, np.sin(radian)], 
                                        [0,1,0], 
                                        [-np.sin(radian), 0, np.cos(radian)]]))
    rot_180_matrix = torch.Tensor(rot_formula.tolist())
    return rot_180_matrix


def axisang_to_rot(axisang):
    """
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py  # I like this :) 
    https://github.com/nkolot/SPIN/blob/5c796852ca7ca7373e104e8489aa5864323fbf84/utils/geometry.py#L9
    Args:
        The axis/rotation angle same as theta: size = [B, 3] in degree
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]    
    """

    # converts angle from degree to radian 
    angle = torch.norm(axisang + 1e-8, p=2, dim=-1)[..., None]
 
    axisang_norm = axisang / angle
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)

    quat = torch.cat([v_cos, v_sin * axisang_norm], dim=-1)
    rot = quat2mat(quat)
    
    return rot



def axisang_to_rot6d(axisang):
    B, N_J, _ = axisang.shape
    rot = axisang_to_rot(axisang.view(-1, 3))
    # dropped the last row
    rot6d = rot[..., :3, :2].reshape(B, N_J, 6)
    return rot6d

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)

    w = norm_quat[:, 0]
    x = norm_quat[:, 1]
    y = norm_quat[:, 2]
    z = norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 
        2 * xy - 2 * wz, 
        2 * wy + 2 * xz, 
        2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 
        2 * yz - 2 * wx, 
        2 * xz - 2 * wy, 
        2 * wx + 2 * yz, 
        w2 - x2 - y2 + z2 
    ], dim=1).view(batch_size, 3, 3)

    return rotMat

